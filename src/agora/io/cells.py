"""Define from an h5 file a class containing data on the segmented cells."""

import logging
import typing as t
from functools import cached_property
from itertools import groupby
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage
from scipy.sparse import issparse


def fill_outline(outline: np.ndarray) -> np.ndarray:
    """
    Fill a cell's outline to give its mask.

    Fill only the box that bounds the outline: filling the whole tile
    gives the same mask and took four times as long.
    """
    outline = np.asarray(outline, dtype=bool)
    mask = np.zeros_like(outline)
    rows = np.flatnonzero(outline.any(axis=1))
    if len(rows):
        columns = np.flatnonzero(outline.any(axis=0))
        box = (
            slice(rows[0], rows[-1] + 1),
            slice(columns[0], columns[-1] + 1),
        )
        mask[box] = ndimage.binary_fill_holes(outline[box])
    return mask


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

    An example is
    self["timepoint"][(self["cell_label"] == 1) & (self["trap"] == 3)]
    to find the time points where cell 1 was present in trap 3.
    """

    def __init__(self, filename, path="cell_info"):
        """Initialise from a filename."""
        self.filename: t.Optional[t.Union[str, Path]] = filename
        self.cinfo_path: t.Optional[str] = path
        self._tile_size: tuple[int, int] | None = None

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

    def log(self, message: str, level: str = "warning"):
        """Log messages in the corresponding level."""
        logger = logging.getLogger("aliby")
        getattr(logger, level)(f"{self.__class__.__name__}: {message}")

    @staticmethod
    def asdense(array: np.ndarray):
        """Convert sparse array to dense array."""
        if issparse(array):
            array = array.todense()
        return array

    @staticmethod
    def astype(array: np.ndarray, kind: str):
        """Convert a sparse array and, if kind is 'mask', fill the outline."""
        array = Cells.asdense(array)
        if kind == "mask":
            array = fill_outline(array)
        return array

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
    def tile_size(self) -> tuple[int, int] | None:
        """
        Give a tile's height and width.

        Read off the stored edge masks rather than the tile_size attribute,
        so it is the shape the masks actually have. A tile need not be
        square: a mother machine's channel is tall and narrow.
        """
        if self._tile_size is None:
            with h5py.File(self.filename, mode="r") as f:
                self._tile_size = f["cell_info/edgemasks"].shape[1:]
        return self._tile_size

    def nonempty_tp_in_trap(self, trap_id: int) -> set:
        """Given a tile, return time points for which cells are available."""
        return set(self["timepoint"][self["trap"] == trap_id])

    def at_time(
        self, timepoint: int, kind="mask"
    ) -> t.List[t.List[np.ndarray]]:
        """
        Return a dict of the cell masks in each trap at a time point.

        Return one mask per cell the h5 records, an empty one included.
        The masks are matched to the traps by position here, and to the
        labels from labels_at_time by position downstream.
        """
        idx = self["timepoint"] == timepoint
        traps = self["trap"][idx]
        edgemasks = self.edgemasks_from_idx(idx)
        masks = [Cells.astype(edgemask, kind) for edgemask in edgemasks]
        return self.group_by_traps(traps, masks)

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
        """Return a dict of the cell labels in each trap at a time point."""
        labels = self["cell_label"][self["timepoint"] == timepoint]
        traps = self["trap"][self["timepoint"] == timepoint]
        return self.group_by_traps(traps, labels)

    def edgemasks_from_idx(self, idx):
        """Get edge masks from the h5 file."""
        with h5py.File(self.filename, mode="r") as f:
            edgem = f[self.cinfo_path + "/edgemasks"][idx, ...]
        return edgem

    @property
    def ntimepoints(self) -> int:
        """
        Return the number of time points up to the last with a cell.

        Zero for a position where no cell was ever found.
        """
        timepoints = self["timepoint"]
        return int(timepoints.max()) + 1 if len(timepoints) else 0

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

        A daughter is named by its cell label. The mothers of a tile are
        listed in the order of the tile's labels, so a daughter's place
        in that list is its label only when the labels run 1 to N with
        no gaps. Cells are lost and labels are skipped: in a tile whose
        labels are 1, 3 and 7, naming the daughter by its place called
        cell 7 cell 3, which is a real cell and the wrong one.
        """
        # a list for each tile of the mother of each of its cells
        mothers = self.mothers
        # not their sum, which is a uint16 and wraps, on occasion to zero
        if any(x for y in mothers for x in y):
            ids = np.unique(
                np.stack((self["trap"], self["cell_label"]), axis=1), axis=0
            )
            # cell labels for each trap
            cell_labels = {
                trap_id: ids[ids[:, 0] == trap_id, 1]
                for trap_id in np.unique(ids[:, 0])
            }
            # a cell with a mother is a daughter
            mothers_daughters = np.array(
                [
                    (trap_id, mother, cell_label)
                    for trap_id, trapmothers in enumerate(mothers)
                    for cell_label, mother in zip(
                        cell_labels.get(trap_id, []), trapmothers
                    )
                    if mother
                ],
                dtype=np.uint16,
            )
        else:
            mothers_daughters = np.array([])
            self.log("No mother-daughters assigned")
        return mothers_daughters

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
        # find when each cell last appeared at its trap: its first row
        # with the rows reversed, which np.unique gives for every cell at
        # once
        ids, last_lin_preds = np.unique(
            np.stack((trap, cell_label), axis=1)[::-1],
            axis=0,
            return_index=True,
        )
        # find the cell's mother using the latest prediction from Baby
        mother_assign_sorted = ma[::-1][last_lin_preds]
        # rearrange as a list of mother IDs for each cell in each tile
        traps = ids[:, 0]
        iterator = groupby(zip(traps, mother_assign_sorted), lambda x: x[0])
        d = {trap: [x[1] for x in mothers] for trap, mothers in iterator}
        mothers = [d.get(i, []) for i in range(ntraps)]
        return mothers
