import logging
import typing as t
from itertools import groupby
from pathlib import Path
from functools import lru_cache, cached_property

import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from scipy.sparse.base import isdense
from utils_find_1st import cmp_equal, find_1st


class Cells:
    """
    Extract information from an h5 file.

    Use output from BABY to find cells detected, get and fill edge masks,
    and retrieve mother-bud relationships.

    This class accesses in the h5 file:

    'cell_info', which contains 'angles', 'cell_label', 'centres',
    'edgemasks', 'ellipse_dims', 'mother_assign', 'mother_assign_dynamic',
    'radii', 'timepoint', and 'trap'. All of which except for 'edgemasks'
    are a 1D ndarray.

    'trap_info', which contains 'drifts', and 'trap_locations'.

    The "timepoint", "cell_label", and "trap" variables are mutually consistent
    1D lists.

    Examples are self["timepoint"][self.get_idx(1, 3)] to find the time points
    where cell 1 was present in trap 3.
    """

    def __init__(self, filename, path="cell_info"):
        """Initialise from a filename."""
        self.filename: t.Optional[t.Union[str, Path]] = filename
        self.cinfo_path: t.Optional[str] = path
        self._edgemasks: t.Optional[str] = None
        self._tile_size: t.Optional[int] = None

    def __getitem__(self, item):
        """
        Dynamically fetch data from the h5 file and save as an attribute.

        These attributes are accessed like dict keys.
        """
        assert item != "edgemasks", "Edgemasks must not be loaded as a whole"
        _item = "_" + item
        if not hasattr(self, _item):
            setattr(self, _item, self.fetch(item))
        return getattr(self, _item)

    def fetch(self, path):
        """Get data from the h5 file."""
        with h5py.File(self.filename, mode="r") as f:
            return f[self.cinfo_path][path][()]

    @classmethod
    def from_source(cls, source: t.Union[Path, str]):
        """Ensure initiating file is a Path object."""
        return cls(Path(source))

    def log(self, message: str, level: str = "warn"):
        """Log messages in the corresponding level."""
        logger = logging.getLogger("aliby")
        getattr(logger, level)(f"{self.__class__.__name__}: {message}")

    @staticmethod
    def asdense(array: np.ndarray):
        """Convert sparse array to dense array."""
        if not isdense(array):
            array = array.todense()
        return array

    @staticmethod
    def astype(array: np.ndarray, kind: str):
        """Convert sparse arrays if needed; if kind is 'mask' fill the outline."""
        array = Cells.asdense(array)
        if kind == "mask":
            array = ndimage.binary_fill_holes(array).astype(bool)
        return array

    def get_idx(self, cell_id: int, trap_id: int):
        """Return boolean array giving indices for a cell_id and trap_id."""
        return (self["cell_label"] == cell_id) & (self["trap"] == trap_id)

    @property
    def max_labels(self) -> t.List[int]:
        """Return the maximum cell label per tile."""
        return [
            max((0, *self.cell_labels_in_trap(i))) for i in range(self.ntraps)
        ]

    @property
    def max_label(self) -> int:
        """Return the maximum cell label over all tiles."""
        return sum(self.max_labels)

    @property
    def ntraps(self) -> int:
        """Find the number of tiles, or traps."""
        with h5py.File(self.filename, mode="r") as f:
            return len(f["trap_info/trap_locations"][()])

    @property
    def tinterval(self):
        """Return time interval in seconds."""
        with h5py.File(self.filename, mode="r") as f:
            return f.attrs["time_settings/timeinterval"]

    @property
    def traps(self) -> t.List[int]:
        """List unique tile, or trap, IDs."""
        return list(set(self["trap"]))

    @property
    def tile_size(self) -> t.Union[int, t.Tuple[int], None]:
        """Give the x- and y- sizes of a tile."""
        if self._tile_size is None:
            with h5py.File(self.filename, mode="r") as f:
                self._tile_size = f["cell_info/edgemasks"].shape[1:]
        return self._tile_size

    def nonempty_tp_in_trap(self, trap_id: int) -> set:
        """Given a tile, return time points for which cells are available."""
        return set(self["timepoint"][self["trap"] == trap_id])

    @property
    def edgemasks(self) -> t.List[np.ndarray]:
        """Return a list of masks for every cell at every trap and time point."""
        if self._edgemasks is None:
            edgem_path: str = "edgemasks"
            self._edgemasks = self.fetch(edgem_path)
        return self._edgemasks

    @property
    def labels(self) -> t.List[t.List[int]]:
        """Return all cell labels per tile as a set for all tiles."""
        return [self.cell_labels_in_trap(trap) for trap in range(self.ntraps)]

    def max_labels_in_frame(self, final_time_point: int) -> t.List[int]:
        """Get the maximal cell label for each tile within a frame of time."""
        max_labels = [
            self["cell_label"][
                (self["timepoint"] <= final_time_point)
                & (self["trap"] == trap_id)
            ]
            for trap_id in range(self.ntraps)
        ]
        return [max([0, *labels]) for labels in max_labels]

    def where(self, cell_id: int, trap_id: int):
        """Return time points, indices, and edge masks for a cell and trap."""
        idx = self.get_idx(cell_id, trap_id)
        return (
            self["timepoint"][idx],
            idx,
            self.edgemasks_where(cell_id, trap_id),
        )

    def mask(self, cell_id, trap_id):
        """Return the times and the filled edge masks for a cell and trap."""
        times, outlines = self.outline(cell_id, trap_id)
        return times, np.array(
            [ndimage.morphology.binary_fill_holes(o) for o in outlines]
        )

    def at_time(
        self, timepoint: int, kind="mask"
    ) -> t.List[t.List[np.ndarray]]:
        """Return a dict with traps as keys and cell masks as values for a time point."""
        idx = self["timepoint"] == timepoint
        traps = self["trap"][idx]
        edgemasks = self.edgemasks_from_idx(idx)
        masks = [
            Cells.astype(edgemask, kind)
            for edgemask in edgemasks
            if edgemask.any()
        ]
        return self.group_by_traps(traps, masks)

    def at_times(
        self, timepoints: t.Iterable[int], kind="mask"
    ) -> t.List[t.List[np.ndarray]]:
        """Return a list of lists of cell masks one for specified time point."""
        return [
            [
                np.stack(tile_masks) if len(tile_masks) else []
                for tile_masks in self.at_time(tp, kind=kind).values()
            ]
            for tp in timepoints
        ]

    def group_by_traps(
        self, traps: t.Collection, cell_labels: t.Collection
    ) -> t.Dict[int, t.List[int]]:
        """Return a dict with traps as keys and a list of labels as values."""
        iterator = groupby(zip(traps, cell_labels), lambda x: x[0])
        d = {key: [x[1] for x in group] for key, group in iterator}
        d = {i: d.get(i, []) for i in self.traps}
        return d

    def cell_labels_in_trap(self, trap_id: int) -> t.Set[int]:
        """Return unique cell labels for a given trap."""
        return set((self["cell_label"][self["trap"] == trap_id]))

    def labels_at_time(self, timepoint: int) -> t.Dict[int, t.List[int]]:
        """Return a dict with traps as keys and cell labels as values for a time point."""
        labels = self["cell_label"][self["timepoint"] == timepoint]
        traps = self["trap"][self["timepoint"] == timepoint]
        return self.group_by_traps(traps, labels)

    def edgemasks_from_idx(self, idx):
        """Get edge masks from the h5 file."""
        with h5py.File(self.filename, mode="r") as f:
            edgem = f[self.cinfo_path + "/edgemasks"][idx, ...]
        return edgem

    def edgemasks_where(self, cell_id, trap_id):
        """Get the edge masks for a given cell and trap for all time points."""
        idx = self.get_idx(cell_id, trap_id)
        edgemasks = self.edgemasks_from_idx(idx)
        return edgemasks

    def outline(self, cell_id: int, trap_id: int):
        """Get times and edge masks for a given cell and trap."""
        idx = self.get_idx(cell_id, trap_id)
        times = self["timepoint"][idx]
        return times, self.edgemasks_from_idx(idx)

    @property
    def ntimepoints(self) -> int:
        """Return total number of time points in the experiment."""
        return self["timepoint"].max() + 1

    @cached_property
    def cells_vs_tps(self):
        """Boolean matrix showing when cells are present for all time points."""
        total_ncells = sum([len(x) for x in self.labels])
        cells_vs_tps = np.zeros((total_ncells, self.ntimepoints), dtype=bool)
        cells_vs_tps[
            self.cell_cumlsum[self["trap"]] + self["cell_label"] - 1,
            self["timepoint"],
        ] = True
        return cells_vs_tps

    @cached_property
    def cell_cumlsum(self):
        """Find cumulative sum over tiles of the number of cells present."""
        ncells_per_tile = [len(x) for x in self.labels]
        cumsum = np.roll(np.cumsum(ncells_per_tile), shift=1)
        cumsum[0] = 0
        return cumsum

    def index_to_tile_and_cell(self, idx: int) -> t.Tuple[int, int]:
        """Convert an index to the equivalent pair of tile and cell IDs."""
        tile_id = int(np.where(idx + 1 > self.cell_cumlsum)[0][-1])
        cell_label = idx - self.cell_cumlsum[tile_id] + 1
        return tile_id, cell_label

    @property
    def tiles_vs_cells_vs_tps(self):
        """
        Boolean matrix showing if a cell is present.

        The matrix is indexed by trap, cell label, and time point.
        """
        ncells_mat = np.zeros(
            (self.ntraps, self["cell_label"].max(), self.ntimepoints),
            dtype=bool,
        )
        ncells_mat[self["trap"], self["cell_label"] - 1, self["timepoint"]] = (
            True
        )
        return ncells_mat

    def cell_tp_where(
        self,
        min_consecutive_tps: int = 15,
        interval: None or t.Tuple[int, int] = None,
    ):
        """
        Find cells present for all time points in a sliding window of time.

        The result can be restricted to a particular interval of time.
        """
        window = sliding_window_view(
            self.cells_vs_tps, min_consecutive_tps, axis=1
        )
        tp_min = window.sum(axis=-1) == min_consecutive_tps
        # apply a filter to restrict to an interval of time
        if interval is not None:
            interval = tuple(np.array(interval))
        else:
            interval = (0, window.shape[1])
        low_boundary, high_boundary = interval
        tp_min[:, :low_boundary] = False
        tp_min[:, high_boundary:] = False
        return tp_min

    @lru_cache(20)
    def mothers_in_trap(self, trap_id: int):
        """Return mothers at a trap."""
        return self.mothers[trap_id]

    @cached_property
    def mothers(self):
        """
        Return a list of mother IDs for each cell in each tile.

        Use Baby's "mother_assign_dynamic".
        An ID of zero implies that no mother was assigned.
        """
        return self.mother_assign_from_dynamic(
            self["mother_assign_dynamic"],
            self["cell_label"],
            self["trap"],
            self.ntraps,
        )

    @cached_property
    def mothers_daughters(self) -> np.ndarray:
        """
        Return mother-daughter relationships for all tiles.

        Returns
        -------
        mothers_daughters: np.ndarray
            An array with shape (n, 3) where n is the number of mother-daughter
            pairs found. The first column is the tile_id for the tile where the
            mother cell is located. The second column is the cell index of a
            mother cell in the tile. The third column is the index of the
            corresponding daughter cell.
        """
        # list of arrays, one per tile, giving mothers of each cell in each tile
        mothers = self.mothers
        if sum([x for y in mothers for x in y]):
            mothers_daughters = np.array(
                [
                    (trap_id, mother, bud)
                    for trap_id, trapcells in enumerate(mothers)
                    for bud, mother in enumerate(trapcells, start=1)
                    if mother
                ],
                dtype=np.uint16,
            )
        else:
            mothers_daughters = np.array([])
            self.log("No mother-daughters assigned")
        return mothers_daughters

    @staticmethod
    def mother_assign_to_mb_matrix(ma: t.List[np.array]):
        """
        Convert a list of mother-daughters into a boolean sparse matrix.

        Each row in the matrix correspond to daughter buds.
        If an entry is True, a given cell is a mother cell and a given
        daughter bud is assigned to the mother cell in the next time point.

        Parameters:
        -----------
        ma : list of lists of integers
            A list of lists of mother-bud assignments.
            The i-th sublist contains the bud assignments for the i-th tile.
            The integers in each sublist represent the mother label, with zero
            implying no mother found.

        Returns:
        --------
        mb_matrix : boolean numpy array of shape (n, m)
            An n x m array where n is the total number of cells (sum
            of the lengths of all sublists in ma) and m is the maximum
            number of buds assigned to any mother cell in ma.
            The value at (i, j) is True if cell i is a daughter cell and
            cell j is its assigned mother.

        Examples:
        --------
        >>> ma = [[0, 0, 1], [0, 1, 0]]
        >>> Cells(None).mother_assign_to_mb_matrix(ma)
        >>> array([[False, False, False, False, False, False],
                   [False, False, False, False, False, False],
                   [ True, False, False, False, False, False],
                   [False, False, False, False, False, False],
                   [False, False, False,  True, False, False],
                   [False, False, False, False, False, False]])
        """
        ncells = sum([len(t) for t in ma])
        mb_matrix = np.zeros((ncells, ncells), dtype=bool)
        c = 0
        for cells in ma:
            for d, m in enumerate(cells):
                if m:
                    mb_matrix[c + d, c + m - 1] = True
            c += len(cells)
        return mb_matrix

    @staticmethod
    def mother_assign_from_dynamic(
        ma: np.ndarray,
        cell_label: t.List[int],
        trap: t.List[int],
        ntraps: int,
    ) -> t.List[t.List[int]]:
        """
        Find mothers from Baby's 'mother_assign_dynamic' variable.

        Parameters
        ----------
        ma: np.ndarray
            An array with of length number of time points times number of cells
            containing the 'mother_assign_dynamic' produced by Baby.
        cell_label: List[int]
            A list of cell labels.
        trap: List[int]
            A list of trap labels.
        ntraps: int
            The total number of traps.

        Returns
        -------
        List[List[int]]
            A list giving the mothers for each cell at each trap.
        """
        ids = np.unique(list(zip(trap, cell_label)), axis=0)
        # find when each cell last appeared at its trap
        last_lin_preds = [
            find_1st(
                (
                    (cell_label[::-1] == cell_label_id)
                    & (trap[::-1] == trap_id)
                ),
                True,
                cmp_equal,
            )
            for trap_id, cell_label_id in ids
        ]
        # find the cell's mother using the latest prediction from Baby
        mother_assign_sorted = ma[::-1][last_lin_preds]
        # rearrange as a list of mother IDs for each cell in each tile
        traps = ids[:, 0]
        iterator = groupby(zip(traps, mother_assign_sorted), lambda x: x[0])
        d = {trap: [x[1] for x in mothers] for trap, mothers in iterator}
        mothers = [d.get(i, []) for i in range(ntraps)]
        return mothers

    ###############################################################################
    # Apparently unused below here
    ###############################################################################

    @lru_cache(maxsize=200)
    def labelled_in_frame(
        self, frame: int, global_id: bool = False
    ) -> np.ndarray:
        """
        Return labels in a 4D ndarray with potentially global ids.

        Use lru_cache to cache the results for speed.

        Parameters
        ----------
        frame : int
            The frame number (time point).
        global_id : bool, optional
            If True, the returned array contains global ids, otherwise only
            the local ids of the labels.

        Returns
        -------
        np.ndarray
            A 4D numpy array containing the labels in the given frame.
            The array has dimensions (ntraps, max_nlabels, ysize, xsize),
            where max_nlabels is specific for this frame, not the entire
            experiment.
        """
        labels_in_frame = self.labels_at_time(frame)
        n_labels = [
            len(labels_in_frame.get(trap_id, []))
            for trap_id in range(self.ntraps)
        ]
        stacks_in_frame = self.get_stacks_in_frame(frame, self.tile_size)
        first_id = np.cumsum([0, *n_labels])
        labels_mat = np.zeros(
            (
                self.ntraps,
                max(n_labels),
                *self.tile_size,
            ),
            dtype=int,
        )
        for trap_id, masks in enumerate(stacks_in_frame):  # new_axis = np.pad(
            if trap_id in labels_in_frame:
                new_axis = np.array(labels_in_frame[trap_id], dtype=int)[
                    :, np.newaxis, np.newaxis
                ]
                global_id_masks = new_axis * masks
                if global_id:
                    global_id_masks += first_id[trap_id] * masks
                global_id_masks = np.pad(
                    global_id_masks,
                    pad_width=(
                        (0, labels_mat.shape[1] - global_id_masks.shape[0]),
                        (0, 0),
                        (0, 0),
                    ),
                )
                labels_mat[trap_id] += global_id_masks
        return labels_mat

    def get_stacks_in_frame(
        self, frame: int, tile_shape: t.Tuple[int]
    ) -> t.List[np.ndarray]:
        """
        Return a list of stacked masks.

        Each corresponds to a tile at a given time point.

        Parameters
        ----------
        frame : int
            Frame for which to obtain the stacked masks.
        tile_shape : Tuple[int]
            Shape of a tile to stack the masks into.

        Returns
        -------
        List[np.ndarray]
            List of stacked masks for each tile at the given time point.
        """
        masks = self.at_time(frame)
        return [
            stack_masks_in_tile(
                masks.get(trap_id, np.array([], dtype=bool)), tile_shape
            )
            for trap_id in range(self.ntraps)
        ]

    def sample_tiles_tps(
        self,
        size=1,
        min_consecutive_ntps: int = 15,
        seed: int = 0,
        interval=None,
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Sample tiles that have a minimum number of cells and are occupied for at least a minimum number of consecutive time points.

        Parameters
        ----------
        size: int, optional (default=1)
            The number of tiles to sample.
        min_ncells: int, optional (default=2)
            The minimum number of cells per tile.
        min_consecutive_ntps: int, optional (default=5)
            The minimum number of consecutive timep oints a cell must be present in a trap.
        seed: int, optional (default=0)
            Random seed value for reproducibility.
        interval: None or Tuple(int,int), optional (default=None)
            Random seed value for reproducibility.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray,np.ndarray]
            A tuple of 1D numpy arrays containing the indices of the sampled tiles and the corresponding timepoints.
        """
        # cell_availability_matrix = self.matrix_trap_tp_where(
        #     min_ncells=min_ncells, min_consecutive_tps=min_consecutive_ntps
        # )

        # # Find all valid tiles with min_ncells for at least min_tps
        # tile_ids, tps = np.where(cell_availability_matrix)
        cell_availability_matrix = self.cell_tp_where(
            min_consecutive_tps=min_consecutive_ntps,
            interval=interval,
        )
        # Find all valid tiles with min_ncells for at least min_tps
        index_id, tps = np.where(cell_availability_matrix)
        if interval is None:  # Limit search
            interval = (0, cell_availability_matrix.shape[1])
        np.random.seed(seed)
        choices = np.random.randint(len(index_id), size=size)
        linear_indices = np.zeros_like(self["cell_label"], dtype=bool)
        for cell_index_flat, tp in zip(index_id[choices], tps[choices]):
            tile_id, cell_label = self.index_to_tile_and_cell(cell_index_flat)
            linear_indices[
                (
                    (self["cell_label"] == cell_label)
                    & (self["trap"] == tile_id)
                    & (self["timepoint"] == tp)
                )
            ] = True
        return linear_indices

    def sample_masks(
        self,
        size: int = 1,
        min_consecutive_ntps: int = 15,
        interval: t.Union[None, t.Tuple[int, int]] = None,
        seed: int = 0,
    ) -> t.Tuple[t.Tuple[t.List[int], t.List[int], t.List[int]], np.ndarray]:
        """
        Sample a number of cells from within an interval.

        Parameters
        ----------
        size: int, optional (default=1)
            The number of cells to sample.
        min_ncells: int, optional (default=2)
            The minimum number of cells per tile.
        min_consecutive_ntps: int, optional (default=5)
            The minimum number of consecutive timepoints a cell must be present in a trap.
        seed: int, optional (default=0)
            Random seed value for reproducibility.

        Returns
        -------
        Tuple[Tuple[np.ndarray, np.ndarray, List[int]], List[np.ndarray]]
            Two tuples are returned. The first tuple contains:
            - `tile_ids`: A 1D numpy array of the tile ids that correspond to the tile identifier.
            - `tps`: A 1D numpy array of the timepoints at which the cells were sampled.
            - `cell_ids`: A list of integers that correspond to the local id of the sampled cells.
            The second tuple contains:
            - `masks`: A list of 2D numpy arrays representing the binary masks of the sampled cells at each timepoint.
        """
        sampled_bitmask = self.sample_tiles_tps(
            size=size,
            min_consecutive_ntps=min_consecutive_ntps,
            seed=seed,
            interval=interval,
        )
        #  Sort sampled tiles to use automatic cache when possible
        tile_ids = self["trap"][sampled_bitmask]
        cell_labels = self["cell_label"][sampled_bitmask]
        tps = self["timepoint"][sampled_bitmask]
        masks = []
        for tile_id, cell_label, tp in zip(tile_ids, cell_labels, tps):
            local_idx = self.labels_at_time(tp)[tile_id].index(cell_label)
            tile_mask = self.at_time(tp)[tile_id][local_idx]
            masks.append(tile_mask)
        return (tile_ids, cell_labels, tps), np.stack(masks)

    def matrix_trap_tp_where(
        self, min_ncells: int = 2, min_consecutive_tps: int = 5
    ):
        """
        NOTE CURRENTLY UNUSED BUT USEFUL.

        Return a matrix of shape (ntraps x ntps - min_consecutive_tps) to
        indicate traps and time-points where min_ncells are available for at least min_consecutive_tps

        Parameters
        ---------
            min_ncells: int Minimum number of cells
            min_consecutive_tps: int
                Minimum number of time-points a

        Returns
        ---------
            (ntraps x ( ntps-min_consecutive_tps )) 2D boolean numpy array where rows are trap ids and columns are timepoint windows.
            If the value in a cell is true its corresponding trap and timepoint contains more than min_ncells for at least min_consecutive time-points.
        """
        window = sliding_window_view(
            self.tiles_vs_cells_vs_tps, min_consecutive_tps, axis=2
        )
        tp_min = window.sum(axis=-1) == min_consecutive_tps
        ncells_tp_min = tp_min.sum(axis=1) >= min_ncells
        return ncells_tp_min


def stack_masks_in_tile(
    masks: t.List[np.ndarray], tile_shape: t.Tuple[int]
) -> np.ndarray:
    """Stack all masks in a trap, padding accordingly if no outlines found."""
    result = np.zeros((0, *tile_shape), dtype=bool)
    if len(masks):
        result = np.stack(masks)
    return result
