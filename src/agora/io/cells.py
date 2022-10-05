import logging
import typing as t
from collections.abc import Iterable
from itertools import groupby
from pathlib import Path, PosixPath
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
        self.filename: t.Optional[t.Union[str, PosixPath]] = filename
        self.cinfo_path: t.Optional[str] = path
        self._edgemasks: t.Optional[str] = None
        self._tile_size: t.Optional[int] = None

    @classmethod
    def from_source(cls, source: t.Union[PosixPath, str]):
        return cls(Path(source))

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
        times, outlines = self.outline(cell_id, trap_id)
        return times, np.array(
            [ndimage.morphology.binary_fill_holes(o) for o in outlines]
        )

    def at_time(self, timepoint, kind="mask"):
        ix = self["timepoint"] == timepoint
        traps = self["trap"][ix]
        edgemasks = self._edgem_from_masking(ix)
        masks = [
            self._astype(edgemask, kind)
            for edgemask in edgemasks
            if edgemask.any()
        ]
        return self.group_by_traps(traps, masks)

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

    @property
    def ncells_matrix(self):
        ncells_mat = np.zeros(
            (self.ntraps, self["cell_label"].max(), self.ntimepoints),
            dtype=bool,
        )
        ncells_mat[
            self["trap"], self["cell_label"] - 1, self["timepoint"]
        ] = True
        return ncells_mat

    def matrix_trap_tp_where(
        self, min_ncells: int = None, min_consecutive_tps: int = None
    ):
        """
        Return a matrix of shape (ntraps x ntps - min_consecutive_tps to
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
        if min_ncells is None:
            min_ncells = 2
        if min_consecutive_tps is None:
            min_consecutive_tps = 5

        window = sliding_window_view(
            self.ncells_matrix, min_consecutive_tps, axis=2
        )
        tp_min = window.sum(axis=-1) == min_consecutive_tps
        ncells_tp_min = tp_min.sum(axis=1) >= min_ncells
        return ncells_tp_min

    def random_valid_trap_tp(
        self, min_ncells: int = None, min_consecutive_tps: int = None
    ):
        # Return a randomly-selected pair of trap_id and timepoints
        mat = self.matrix_trap_tp_where(
            min_ncells=min_ncells, min_consecutive_tps=min_consecutive_tps
        )
        traps, tps = np.where(mat)
        rand = np.random.randint(mat.sum())
        return (traps[rand], tps[rand])

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
        Return mothers and daugters as a single array with three columns:
        trap, mothers and daughters
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
            # print("Warning:Cells: No mother-daughters assigned")

        return mothers_daughters

    @staticmethod
    def mother_assign_to_mb_matrix(ma: t.List[np.array]):
        # Convert from list of lists to mother_bud sparse matrix
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
        ma, cell_label: t.List[int], trap: t.List[int], ntraps: int
    ):
        """
        Interpolate the list of lists containing the associated mothers from the mother_assign_dynamic feature
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
    def labelled_in_frame(self, frame: int, global_id=False) -> np.ndarray:
        """
        Return labels in a ndarray with the global ids
        with shape (ntraps, max_nlabels, ysize, xsize)
        at a given frame.

        max_nlabels is specific for this frame, not
        the entire experiment.
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

    def get_stacks_in_frame(self, frame: int, tile_shape: t.Tuple[int]):
        # Stack all cells in a trap-wise manner
        masks = self.at_time(frame)
        return [
            stack_masks_in_trap(
                masks.get(trap_id, np.array([], dtype=bool)), tile_shape
            )
            for trap_id in range(self.ntraps)
        ]


def stack_masks_in_trap(
    masks: t.List[np.ndarray], tile_shape: t.Tuple[int]
) -> np.ndarray:
    # Stack all masks in a trap padding accordingly if no outlines found
    result = np.zeros((0, *tile_shape), dtype=bool)
    if len(masks):
        result = np.array(masks)
    return result
