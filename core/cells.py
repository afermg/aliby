from pathlib import Path, PosixPath
from typing import Union
from itertools import groupby

import h5py
import numpy as np
from scipy import ndimage
from scipy.sparse.base import isdense

from core.io.matlab import matObject


def cell_factory(store, type="matlab"):
    if isinstance(store, matObject):
        return CellsMat(store)
    if type == "matlab":
        mat_object = matObject(store)
        return CellsMat(mat_object)
    elif type == "hdf5":
        file = h5py.File(store)
        return CellsHDF(file)
    else:
        raise TypeError(
            "Could not get cells for type {}:" "valid types are matlab and hdf5"
        )


class Cells:
    """An object that gathers information about all the cells in a given
    trap.
    This is the abstract object, used for type testing
    """

    def __init__(self):
        pass

    @staticmethod
    def from_source(source: Union[PosixPath, str], kind: str = None):
        if isinstance(source, str):
            source = Path(source)
        if kind is None:  # Infer kind from filename
            kind = "matlab" if source.suffix == ".mat" else "hdf5"
        return cell_factory(source, kind)

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
            array = ndimage.binary_fill_holes(array).astype(int)
        return array


# def is_or_in(item, arr): #TODO CLEAN if not being used
#     if isinstance(arr, (list, np.ndarray)):
#         return item in arr
#     else:
#         return item == arr


from core.io.hdf5 import hdf_dict


from functools import wraps


class CellsHDF(Cells):
    # DONE implement cells information from HDF5 file format
    # TODO combine all the cells of one strain into a cellResults?
    # TODO filtering
    def __init__(self, file):
        self._file = file
        self._info = hdf_dict(self._file.get("/cell_info"))

    def __getitem__(self, item):
        _item = "_" + item
        if not hasattr(self, _item):
            setattr(self, _item, self._info[item][()])
        return getattr(self, _item)

    def _get_idx(self, cell_id, trap_id):
        return (self["cell_label"] == cell_id) & (self["trap"] == trap_id)

    @property
    def traps(self):
        return list(set(self["trap"]))

    @property
    def labels(self):
        """
        Return all cell labels in object
        We use mother_assign to list traps because it is the only propriety that appears even
        when no cells are found"""
        return [self.labels_in_trap(trap) for trap in self.traps]

    def where(self, cell_id, trap_id):
        indices = self._get_idx(cell_id, trap_id)
        return self["timepoints"][indices], indices

    def outline(self, cell_id, trap_id):
        times, indices = self.where(cell_id, trap_id)
        return times, self["edgemasks"][indices]

    def mask(self, cell_id, trap_id):
        times, outlines = self.outline(cell_id, trap_id)
        return times, np.array(
            [ndimage.morphology.binary_fill_holes(o) for o in outlines]
        )

    def at_time(self, timepoint, kind="mask"):
        edgemasks = self["edgemasks"][self["timepoint"] == timepoint]
        traps = self["trap"][self["timepoint"] == timepoint]

        masks = [self._astype(edgemask, kind) for edgemask in edgemasks]

        return self.group_by_traps(traps, masks)

    def group_by_traps(self, traps, data):
        # returns a dict with traps as keys and labels as value
        iterator = groupby(zip(traps, data), lambda x: x[0])
        d = {key: [x[1] for x in group] for key, group in iterator}
        d = {i: d.get(i, []) for i in self.traps}
        return list(d.values())

    def labels_in_trap(self, trap_id):
        # Return set of cell ids in a trap.
        return set((self["cell_label"][self["trap"] == trap_id]))

    def labels_at_time(self, timepoint):
        labels = self["cell_label"][self["timepoint"] == timepoint]
        traps = self["trap"][self["timepoint"] == timepoint]
        return self.group_by_traps(traps, labels)

    @property
    def close(self):
        self._file.close()


class CellsMat(Cells):
    def __init__(self, mat_object):
        super(CellsMat, self).__init__()
        # TODO add __contains__ to the matObject
        timelapse_traps = mat_object.get(
            "timelapseTrapsOmero", mat_object.get("timelapseTraps", None)
        )
        if timelapse_traps is None:
            raise NotImplementedError(
                "Could not find a timelapseTraps or "
                "timelapseTrapsOmero object. Cells "
                "from cellResults not implemented"
            )
        else:
            self.trap_info = timelapse_traps["cTimepoint"]["trapInfo"]
            if isinstance(self.trap_info, list):
                self.trap_info = {
                    k: list([res.get(k, []) for res in self.trap_info])
                    for k in self.trap_info[0].keys()
                }

    def where(self, cell_id, trap_id):
        times, indices = zip(
            *[
                (tp, np.where(cell_id == x)[0][0])
                for tp, x in enumerate(self.trap_info["cellLabel"][:, trap_id].tolist())
                if np.any(cell_id == x)
            ]
        )
        return times, indices

    def outline(self, cell_id, trap_id):
        times, indices = self.where(cell_id, trap_id)
        info = self.trap_info["cell"][times, trap_id]

        def get_segmented(cell, index):
            if cell["segmented"].ndim == 0:
                return cell["segmented"][()].todense()
            else:
                return cell["segmented"][index].todense()

        segmentation_outline = [
            get_segmented(cell, idx) for idx, cell in zip(indices, info)
        ]
        return times, np.array(segmentation_outline)

    def mask(self, cell_id, trap_id):
        times, outlines = self.outline(cell_id, trap_id)
        return times, np.array(
            [ndimage.morphology.binary_fill_holes(o) for o in outlines]
        )

    def at_time(self, timepoint, kind="outline"):
        """Returns the segmentations for all the cells at a given timepoint.

        FIXME: this is extremely hacky and accounts for differently saved
            results in the matlab object. Deprecate ASAP.
        """
        # Case 1: only one cell per trap: trap_info['cell'][timepoint] is a
        # structured array
        if isinstance(self.trap_info["cell"][timepoint], dict):
            segmentations = [
                self._astype(x, "outline")
                for x in self.trap_info["cell"][timepoint]["segmented"]
            ]
        # Case 2: Multiple cells per trap: it becomes a list of arrays or
        # dictionaries,  one for each trap
        # Case 2.1 : it's a dictionary
        elif isinstance(self.trap_info["cell"][timepoint][0], dict):
            segmentations = []
            for x in self.trap_info["cell"][timepoint]:
                seg = x["segmented"]
                if not isinstance(seg, np.ndarray):
                    seg = [seg]
                segmentations.append([self._astype(y, "outline") for y in seg])
        # Case 2.2 : it's an array
        else:
            segmentations = [
                [self._astype(y, type) for y in x["segmented"]] if x.ndim != 0 else []
                for x in self.trap_info["cell"][timepoint]
            ]
        return segmentations

    def to_hdf(self):
        pass


class ExtractionRunner:
    """An object to run extraction of fluorescence, and general data out of
    segmented data.

    Configure with what extraction we want to run.
    Cell selection criteria.
    Filtering criteria.
    """

    def __init__(self, tiler, cells):
        pass

    def run(self, keys, store, **kwargs):
        pass
