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
    Extracts information from an h5 file. This class accesses:

    'cell_info', which contains 'angles', 'cell_label', 'centres',
    'edgemasks', 'ellipse_dims', 'mother_assign', 'mother_assign_dynamic',
    'radii', 'timepoint', 'trap'.
    All of these except for 'edgemasks' are a 1D ndarray.

    'trap_info', which contains 'drifts', 'trap_locations'

    """

    def __init__(self, filename, path="cell_info"):
        self.filename: t.Optional[t.Union[str, Path]] = filename
        self.cinfo_path: t.Optional[str] = path
        self._edgemasks: t.Optional[str] = None
        self._tile_size: t.Optional[int] = None

    @classmethod
    def from_source(cls, source: t.Union[Path, str]):
        return cls(Path(source))

    def _log(self, message: str, level: str = "warn"):
        # Log messages in the corresponding level
        logger = logging.getLogger("aliby")
        getattr(logger, level)(f"{self.__class__.__name__}: {message}")

    @staticmethod
    def _asdense(array: np.ndarray):
        if not isdense(array):
            array = array.todense()
        return array

    @staticmethod
    def _astype(array: np.ndarray, kind: str):
        # Convert sparse arrays if needed and if kind is 'mask' it fills the outline
        array = Cells._asdense(array)
        if kind == "mask":
            array = ndimage.binary_fill_holes(array).astype(bool)
        return array

    def _get_idx(self, cell_id: int, trap_id: int):
        # returns boolean array of time points where both the cell with cell_id and the trap with trap_id exist
        return (self["cell_label"] == cell_id) & (self["trap"] == trap_id)

    @property
    def max_labels(self) -> t.List[int]:
        return [max((0, *self.labels_in_trap(i))) for i in range(self.ntraps)]

    @property
    def max_label(self) -> int:
        return sum(self.max_labels)

    @property
    def ntraps(self) -> int:
        # find the number of traps from the h5 file
        with h5py.File(self.filename, mode="r") as f:
            return len(f["trap_info/trap_locations"][()])

    @property
    def tinterval(self):
        with h5py.File(self.filename, mode="r") as f:
            return f.attrs["time_settings/timeinterval"]

    @property
    def traps(self) -> t.List[int]:
        # returns a list of traps
        return list(set(self["trap"]))

    @property
    def tile_size(self) -> t.Union[int, t.Tuple[int], None]:
        if self._tile_size is None:
            with h5py.File(self.filename, mode="r") as f:
                # self._tile_size = f["trap_info/tile_size"][0]
                self._tile_size = f["cell_info/edgemasks"].shape[1:]
        return self._tile_size

    def nonempty_tp_in_trap(self, trap_id: int) -> set:
        # given a trap_id returns time points in which cells are available
        return set(self["timepoint"][self["trap"] == trap_id])

    @property
    def edgemasks(self) -> t.List[np.ndarray]:
        # returns the masks per tile
        if self._edgemasks is None:
            edgem_path: str = "edgemasks"
            self._edgemasks = self._fetch(edgem_path)
        return self._edgemasks

    @property
    def labels(self) -> t.List[t.List[int]]:
        """
        Return all cell labels in object
        We use mother_assign to list traps because it is the only property that appears even
        when no cells are found
        """
        return [self.labels_in_trap(trap) for trap in range(self.ntraps)]

    def max_labels_in_frame(self, frame: int) -> t.List[int]:
        # Return the maximum label for each trap in the given frame
        max_labels = [
            self["cell_label"][
                (self["timepoint"] <= frame) & (self["trap"] == trap_id)
            ]
            for trap_id in range(self.ntraps)
        ]
        return [max([0, *labels]) for labels in max_labels]

    def where(self, cell_id: int, trap_id: int):
        """
        Parameters
        ----------
            cell_id: int
                Cell index
            trap_id: int
                Trap index

        Returns
        ----------
            indices int array
            boolean mask array
            edge_ix int array
        """
        indices = self._get_idx(cell_id, trap_id)
        edgem_ix = self._edgem_where(cell_id, trap_id)
        return (
            self["timepoint"][indices],
            indices,
            edgem_ix,
        )

    def mask(self, cell_id, trap_id):
        """
        Returns the times and the binary masks of a given cell in a given tile.

        Parameters
        ----------
        cell_id : int
            The unique ID of the cell.
        tile_id : int
            The unique ID of the tile.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The times when the binary masks were taken and the binary masks of the given cell in the given tile.

        """
        times, outlines = self.outline(cell_id, trap_id)
        return times, np.array(
            [ndimage.morphology.binary_fill_holes(o) for o in outlines]
        )

    def at_time(
        self, timepoint: t.Iterable[int], kind="mask"
    ) -> t.List[t.List[np.ndarray]]:
        """
        Returns a list of lists of binary masks in a given list of time points.

        Parameters
        ----------
        timepoints : Iterable[int]
            The list of time points for which to return the binary masks.
        kind : str, optional
            The type of binary masks to return, by default "mask".

        Returns
        -------
        List[List[np.ndarray]]
            A list of lists with binary masks grouped by tile IDs.

        """

        ix = self["timepoint"] == timepoint
        traps = self["trap"][ix]
        edgemasks = self._edgem_from_masking(ix)
        masks = [
            self._astype(edgemask, kind)
            for edgemask in edgemasks
            if edgemask.any()
        ]
        return self.group_by_traps(traps, masks)

    def at_times(
        self, timepoints: t.Iterable[int], kind="mask"
    ) -> t.List[t.List[np.ndarray]]:
        """
        Returns a list of lists of binary masks for a given list of time points.

        Parameters
        ----------
        timepoints : Iterable[int]
            The list of time points for which to return the binary masks.
        kind : str, optional
            The type of binary masks to return, by default "mask".

        Returns
        -------
        List[List[np.ndarray]]
            A list of lists with binary masks grouped by tile IDs.

        """
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
        """
        Returns a dict with traps as keys and list of labels as value.
        Note that the total number of traps are calculated from Cells.traps.

        """
        iterator = groupby(zip(traps, cell_labels), lambda x: x[0])
        d = {key: [x[1] for x in group] for key, group in iterator}
        d = {i: d.get(i, []) for i in self.traps}
        return d

    def labels_in_trap(self, trap_id: int) -> t.Set[int]:
        # return set of cell ids for a given trap
        return set((self["cell_label"][self["trap"] == trap_id]))

    def labels_at_time(self, timepoint: int) -> t.Dict[int, t.List[int]]:
        labels = self["cell_label"][self["timepoint"] == timepoint]
        traps = self["trap"][self["timepoint"] == timepoint]
        return self.group_by_traps(traps, labels)

    def __getitem__(self, item):
        assert item != "edgemasks", "Edgemasks must not be loaded as a whole"

        _item = "_" + item
        if not hasattr(self, _item):
            setattr(self, _item, self._fetch(item))
        return getattr(self, _item)

    def _fetch(self, path):
        with h5py.File(self.filename, mode="r") as f:
            return f[self.cinfo_path][path][()]

    def _edgem_from_masking(self, mask):
        with h5py.File(self.filename, mode="r") as f:
            edgem = f[self.cinfo_path + "/edgemasks"][mask, ...]
        return edgem

    def _edgem_where(self, cell_id, trap_id):
        id_mask = self._get_idx(cell_id, trap_id)
        edgem = self._edgem_from_masking(id_mask)

        return edgem

    def outline(self, cell_id: int, trap_id: int):
        id_mask = self._get_idx(cell_id, trap_id)
        times = self["timepoint"][id_mask]

        return times, self._edgem_from_masking(id_mask)

    @property
    def ntimepoints(self) -> int:
        return self["timepoint"].max() + 1

    @cached_property
    def _cells_vs_tps(self):
        # Binary matrix showing the presence of all cells in all time points
        ncells_per_tile = [len(x) for x in self.labels]
        cells_vs_tps = np.zeros(
            (sum(ncells_per_tile), self.ntimepoints), dtype=bool
        )

        cells_vs_tps[
            self._cell_cumsum[self["trap"]] + self["cell_label"] - 1,
            self["timepoint"],
        ] = True
        return cells_vs_tps

    @cached_property
    def _cell_cumsum(self):
        # Cumulative sum indicating the number of cells per tile
        ncells_per_tile = [len(x) for x in self.labels]
        cumsum = np.roll(np.cumsum(ncells_per_tile), shift=1)
        cumsum[0] = 0
        return cumsum

    def _flat_index_to_tuple_location(self, idx: int) -> t.Tuple[int, int]:
        # Convert a cell index to a tuple
        # Note that it assumes tiles and cell labels are flattened, but
        # it is agnostic to tps

        tile_id = int(np.where(idx + 1 > self._cell_cumsum)[0][-1])
        cell_label = idx - self._cell_cumsum[tile_id] + 1
        return tile_id, cell_label

    @property
    def _tiles_vs_cells_vs_tps(self):
        ncells_mat = np.zeros(
            (self.ntraps, self["cell_label"].max(), self.ntimepoints),
            dtype=bool,
        )
        ncells_mat[
            self["trap"], self["cell_label"] - 1, self["timepoint"]
        ] = True
        return ncells_mat

    def cell_tp_where(
        self,
        min_consecutive_tps: int = 15,
        interval: None or t.Tuple[int, int] = None,
    ):
        window = sliding_window_view(
            self._cells_vs_tps, min_consecutive_tps, axis=1
        )

        tp_min = window.sum(axis=-1) == min_consecutive_tps

        # Apply an interval filter to focucs on a slice
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
        return self.mothers[trap_id]

    @cached_property
    def mothers(self):
        """
        Return nested list with final prediction of mother id for each cell
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
        Return a single array with three columns, containing information about
        the mother-daughter relationships: tile, mothers and daughters.

        Returns
        -------
        np.ndarray
            An array with shape (n, 3) where n is the number of mother-daughter pairs found.
            The columns contain:
            - tile: the tile where the mother cell is located.
            - mothers: the index of the mother cell within the tile.
            - daughters: the index of the daughter cell within the tile.
        """
        nested_massign = self.mothers

        if sum([x for y in nested_massign for x in y]):
            mothers_daughters = np.array(
                [
                    (tid, m, d)
                    for tid, trapcells in enumerate(nested_massign)
                    for d, m in enumerate(trapcells, 1)
                    if m
                ],
                dtype=np.uint16,
            )
        else:
            mothers_daughters = np.array([])
            self._log("No mother-daughters assigned")

        return mothers_daughters

    @staticmethod
    def mother_assign_to_mb_matrix(ma: t.List[np.array]):
        """
        Convert from a list of lists of mother-bud paired assignments to a
        sparse matrix with a boolean dtype. The rows correspond to
        to daughter buds. The values are boolean and indicate whether a
        given cell is a mother cell and a given daughter bud is assigned
        to the mother cell in the next timepoint.

        Parameters:
        -----------
        ma : list of lists of integers
            A list of lists of mother-bud assignments. The i-th sublist contains the
            bud assignments for the i-th tile. The integers in each sublist
            represent the mother label, if it is zero no mother was found.

        Returns:
        --------
        mb_matrix : boolean numpy array of shape (n, m)
            An n x m boolean numpy array where n is the total number of cells (sum
            of the lengths of all sublists in ma) and m is the maximum number of buds
            assigned to any mother cell in ma. The value at (i, j) is True if cell i
            is a daughter cell and cell j is its mother assigned to i.

        Examples:
        --------
        ma = [[0, 0, 1], [0, 1, 0]]
        Cells(None).mother_assign_to_mb_matrix(ma)
        # array([[False, False, False, False, False, False],
        #        [False, False, False, False, False, False],
        #        [ True, False, False, False, False, False],
        #        [False, False, False, False, False, False],
        #        [False, False, False,  True, False, False],
        #        [False, False, False, False, False, False]])

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
        ma: np.ndarray, cell_label: t.List[int], trap: t.List[int], ntraps: int
    ) -> t.List[t.List[int]]:
        """
        Interpolate the associated mothers from the 'mother_assign_dynamic' feature.

        Parameters
        ----------
        ma: np.ndarray
            An array with shape (n_t, n_c) containing the 'mother_assign_dynamic' feature.
        cell_label: List[int]
            A list containing the cell labels.
        trap: List[int]
            A list containing the trap labels.
        ntraps: int
            The total number of traps.

        Returns
        -------
        List[List[int]]
            A list of lists containing the interpolated mother assignment for each cell in each trap.
        """
        idlist = list(zip(trap, cell_label))
        cell_gid = np.unique(idlist, axis=0)

        last_lin_preds = [
            find_1st(
                ((cell_label[::-1] == lbl) & (trap[::-1] == tr)),
                True,
                cmp_equal,
            )
            for tr, lbl in cell_gid
        ]
        mother_assign_sorted = ma[::-1][last_lin_preds]

        traps = cell_gid[:, 0]
        iterator = groupby(zip(traps, mother_assign_sorted), lambda x: x[0])
        d = {key: [x[1] for x in group] for key, group in iterator}
        nested_massign = [d.get(i, []) for i in range(ntraps)]

        return nested_massign

    @lru_cache(maxsize=200)
    def labelled_in_frame(
        self, frame: int, global_id: bool = False
    ) -> np.ndarray:
        """
        Returns labels in a 4D ndarray with the global ids with shape
        (ntraps, max_nlabels, ysize, xsize) at a given frame.

        Parameters
        ----------
        frame : int
            The frame number.
        global_id : bool, optional
            If True, the returned array contains global ids, otherwise it
            contains only the local ids of the labels. Default is False.

        Returns
        -------
        np.ndarray
            A 4D numpy array containing the labels in the given frame.
            The array has dimensions (ntraps, max_nlabels, ysize, xsize),
            where max_nlabels is specific for this frame, not the entire
            experiment.

        Notes
        -----
        This method uses lru_cache to cache the results for faster access.

        """
        labels_in_frame = self.labels_at_time(frame)
        n_labels = [
            len(labels_in_frame.get(trap_id, []))
            for trap_id in range(self.ntraps)
        ]
        # maxes = self.max_labels_in_frame(frame)
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
        Returns a list of stacked masks, each corresponding to a tile at a given timepoint.

        Parameters
        ----------
        frame : int
            Frame for which to obtain the stacked masks.
        tile_shape : Tuple[int]
            Shape of a tile to stack the masks into.

        Returns
        -------
        List[np.ndarray]
            List of stacked masks for each tile at the given timepoint.
        """
        masks = self.at_time(frame)
        return [
            stack_masks_in_tile(
                masks.get(trap_id, np.array([], dtype=bool)), tile_shape
            )
            for trap_id in range(self.ntraps)
        ]

    def _sample_tiles_tps(
        self,
        size=1,
        min_consecutive_ntps: int = 15,
        seed: int = 0,
        interval=None,
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Sample tiles that have a minimum number of cells and are occupied for at least a minimum number of consecutive timepoints.

        Parameters
        ----------
        size: int, optional (default=1)
            The number of tiles to sample.
        min_ncells: int, optional (default=2)
            The minimum number of cells per tile.
        min_consecutive_ntps: int, optional (default=5)
            The minimum number of consecutive timepoints a cell must be present in a trap.
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
            tile_id, cell_label = self._flat_index_to_tuple_location(
                cell_index_flat
            )
            linear_indices[
                (
                    (self["cell_label"] == cell_label)
                    & (self["trap"] == tile_id)
                    & (self["timepoint"] == tp)
                )
            ] = True

        return linear_indices

    def _sample_masks(
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
        sampled_bitmask = self._sample_tiles_tps(
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
        NOTE CURRENLTY UNUSED WITHIN ALIBY THE MOMENT. MAY BE USEFUL IN THE FUTURE.

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
            self._tiles_vs_cells_vs_tps, min_consecutive_tps, axis=2
        )
        tp_min = window.sum(axis=-1) == min_consecutive_tps
        ncells_tp_min = tp_min.sum(axis=1) >= min_ncells
        return ncells_tp_min


def stack_masks_in_tile(
    masks: t.List[np.ndarray], tile_shape: t.Tuple[int]
) -> np.ndarray:
    # Stack all masks in a trap padding accordingly if no outlines found
    result = np.zeros((0, *tile_shape), dtype=bool)
    if len(masks):
        result = np.stack(masks)
    return result
