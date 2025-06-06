"""Define from an h5 file a class containing data on the segmented cells."""

import logging
import typing as t
from functools import cached_property, lru_cache
from itertools import groupby
from pathlib import Path

import h5py
import numpy as np
from agora.utils.indexing import find_1st_equal
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from scipy.sparse.base import isdense


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

    The "timepoint", "cell_label", and "trap" variables are mutually
    consistent 1D lists.

    Examples are self["timepoint"][self.get_idx(1, 3)] to find the time
    points where cell 1 was present in trap 3.
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
            find_1st_equal(
                (
                    (cell_label[::-1] == cell_label_id)
                    & (trap[::-1] == trap_id)
                ),
                True,
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
