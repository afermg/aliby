import logging
import typing as t
from collections.abc import Iterable
from itertools import groupby
from pathlib import Path, PosixPath

import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from scipy.sparse.base import isdense
from utils_find_1st import cmp_equal, find_1st

from agora.io.writer import load_complex


class Cells:
    """An object that gathers information about all the cells in a given
    trap.
    This is the abstract object, used for type testing
    """

    @classmethod
    def from_source(cls, source: t.Union[PosixPath, str]):
        return cls(Path(source))

    @staticmethod
    def _asdense(array):
        if not isdense(array):
            array = array.todense()
        return array

    @staticmethod
    def _astype(array, kind):
        # Convert sparse arrays if needed and if kind is 'mask' it fills the outline
        array = Cells._asdense(array)
        if kind == "mask":
            array = ndimage.binary_fill_holes(array).astype(bool)
        return array

    @classmethod
    def hdf(cls, fpath):
        return CellsHDF(fpath)


class CellsHDF(Cells):
    def __init__(
        self, filename: t.Union[str, PosixPath], path: str = "cell_info"
    ):
        self.filename = filename
        self.cinfo_path = path
        self._edgem_indices = None
        self._edgemasks = None
        self._tile_size = None

    def __getitem__(self, item: str) -> np.ndarray:
        if item == "edgemasks":
            return self.edgemasks
        _item = "_" + item
        if not hasattr(self, _item):
            setattr(self, _item, self._fetch(item))
        return getattr(self, _item)

    def _get_idx(self, cell_id: int, trap_id: int) -> t.List[bool]:
        return (self["cell_label"] == cell_id) & (self["trap"] == trap_id)

    def _fetch(self, path: str) -> t.List[t.Union[np.ndarray, int]]:
        with h5py.File(self.filename, mode="r") as f:
            return f[self.cinfo_path][path][()]

    @property
    def max_labels(self) -> t.List[int]:
        return [max(self.labels_in_trap(i)) for i in range(self.ntraps)]

    @property
    def ntraps(self) -> int:
        with h5py.File(self.filename, mode="r") as f:
            return len(f["/trap_info/trap_locations"][()])

    @property
    def tinterval(self):
        with h5py.File(self.filename, mode="r") as f:
            return f.attrs["time_settings/timeinterval"]

    @property
    def traps(self) -> t.List[int]:
        return list(set(self["trap"]))

    @property
    def ntimepoints(self) -> int:
        return self["timepoint"].max() + 1

    @property
    def tile_size(self) -> t.Union[int, t.Tuple[int], None]:
        if self._tile_size is None:
            with h5py.File(self.filename, mode="r") as f:
                self._tile_size == f["trap_info/tile_size"][0]
        return self._tile_size

    @property
    def edgem_indices(self) -> t.Union[np.ndarray, None]:
        if self._edgem_indices is None:
            edgem_path = "edgemasks/indices"
            self._edgem_indices = load_complex(self._fetch(edgem_path))
        return self._edgem_indices

    def nonempty_tp_in_trap(self, trap_id: int) -> t.Set[bool]:
        # Returns time-points in which cells are available
        return set(self["timepoint"][self["trap"] == trap_id])

    @property
    def edgemasks(self) -> t.List[np.ndarray]:
        if self._edgemasks is None:
            edgem_path = "edgemasks/values"
            self._edgemasks = self._fetch(edgem_path)

        return self._edgemasks

    def _edgem_where(self, cell_id: int, trap_id: int):
        ix = trap_id + 1j * cell_id
        return find_1st(self.edgem_indices == ix, True, cmp_equal)

    @property
    def labels(self) -> t.List[t.List[int]]:
        """
        Return all cell labels in object
        We use mother_assign to list traps because it is the only propriety that appears even
        when no cells are found"""
        return [self.labels_in_trap(trap) for trap in self.traps]

    def where(self, cell_id: int, trap_id: int):
        """
        Returns
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

    def outline(
        self, cell_id: int, trap_id: int
    ) -> t.Tuple[t.List, np.ndarray]:
        times, indices, cell_ix = self.where(cell_id, trap_id)
        return times, self["edgemasks"][cell_ix, times]

    def mask(self, cell_id, trap_id):
        times, outlines = self.outline(cell_id, trap_id)
        return times, np.array(
            [ndimage.morphology.binary_fill_holes(o) for o in outlines]
        )

    def at_time(
        self, timepoint: int, kind: str = "mask"
    ) -> t.Dict[int, np.ndarray]:
        ix = self["timepoint"] == timepoint
        cell_ix = self["cell_label"][ix]
        traps = self["trap"][ix]
        indices = traps + 1j * cell_ix
        choose = np.in1d(self.edgem_indices, indices)
        edgemasks = self["edgemasks"][choose, timepoint]
        masks = [
            self._astype(edgemask, kind)
            for edgemask in edgemasks
            if edgemask.any()
        ]
        return self.group_by_traps(traps, masks)

    def group_by_traps(self, traps, data):
        # returns a dict with traps as keys and labels as value
        iterator = groupby(zip(traps, data), lambda x: x[0])
        d = {key: [x[1] for x in group] for key, group in iterator}
        d = {i: d.get(i, []) for i in self.traps}
        return d

    def labels_in_trap(self, trap_id):
        # Return set of cell ids in a trap.
        return set((self["cell_label"][self["trap"] == trap_id]))

    def labels_at_time(self, timepoint):
        labels = self["cell_label"][self["timepoint"] == timepoint]
        traps = self["trap"][self["timepoint"] == timepoint]
        return self.group_by_traps(traps, labels)


class CellsLinear(CellsHDF):
    """
    Reimplement functions from CellsHDF to save edgemasks in a (N,tile_size, tile_size) array

    This overrides the previous implementation of at_time.
    """

    def __init__(self, filename, path="cell_info"):
        super().__init__(filename, path=path)

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

    def outline(self, cell_id, trap_id):
        id_mask = self._get_idx(cell_id, trap_id)
        times = self["timepoint"][id_mask]

        return times, self.edgem_from_masking(id_mask)

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

    def mothers_in_trap(self, trap_id: int):
        return self.mothers[trap_id]

    @property
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

    @property
    def mothers_daughters(self):
        nested_massign = self.mothers

        if sum([x for y in nested_massign for x in y]):
            mothers, daughters = zip(
                *[
                    ((tid, m), (tid, d))
                    for tid, trapcells in enumerate(nested_massign)
                    for d, m in enumerate(trapcells, 1)
                    if m
                ]
            )
        else:
            mothers, daughters = ([], [])
            # print("Warning:Cells: No mother-daughters assigned")

        return mothers, daughters

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
