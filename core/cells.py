from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

from core.io.matlab import matObject


def cell_factory(store, type='matlab'):
    if isinstance(store, matObject):
        return CellsMat(store)
    if type == 'matlab':
        mat_object = matObject(store)
        return CellsMat(mat_object)
    elif type == 'hdf5':
        file = h5py.File(store)
        return CellsHDF(file)
    else:
        raise TypeError("Could not get cells for type {}:"
                        "valid types are matlab and hdf5")


class Cells:
    """An object that gathers information about all the cells in a given
    trap.
    This is the abstract object, used for type testing
    """

    def __init__(self):
        pass

    @staticmethod
    def from_source(source, type=None):
        if isinstance(source, str):
            source = Path(source)
            if type is None:
                # Infer type from filename
                type = 'matlab' if source.suffix == '.mat' else 'hdf5'
        return cell_factory(source, type)


def is_or_in(item, arr):
    if isinstance(arr, (list, np.ndarray)):
        return item in arr
    else:
        return item == arr


class CellsHDF(Cells):
    # TODO implement cells information from HDF5 file format
    # TODO combine all the cells of one strain into a cellResults?
    # TODO filtering
    def __init__(self, file):
        pass

    def where(self, cell_id, trap_id):
        pass

    def outline(self, cell_id, trap_id):
        pass

    def mask(self, cell_id, trap_id):
        pass


class CellsMat(Cells):
    def __init__(self, mat_object):
        super(CellsMat, self).__init__()
        # TODO add __contains__ to the matObject
        if 'timelapseTrapsOmero' in mat_object.attrs:
            self.trap_info = mat_object['timelapseTrapsOmero'][
                'cTimepoint']['trapInfo']
            if isinstance(self.trap_info, list):
                self.trap_info = {k: list([res.get(k, [])
                                           for res in self.trap_info])
                                        for k in self.trap_info[0].keys()}
        else:
            raise NotImplementedError('Cells from Cell Results not yet '
                                      'implemented')

    def where(self, cell_id, trap_id):
        times, indices = zip(*[(tp, np.where(cell_id == x)[0][0])
                               for tp, x in
                               enumerate(self.trap_info['cellLabel'][:,
                                         trap_id].tolist())
                               if np.any(cell_id == x)])
        return times, indices

    def outline(self, cell_id, trap_id):
        times, indices = self.where(cell_id, trap_id)
        info = self.trap_info['cell'][times, trap_id]

        def get_segmented(cell, index):
            if cell['segmented'].ndim == 0:
                return cell['segmented'][()].todense()
            else:
                return cell['segmented'][index].todense()

        segmentation_outline = [get_segmented(cell, idx)
                                for idx, cell in zip(indices, info)]
        return times, np.array(segmentation_outline)

    def mask(self, cell_id, trap_id):
        times, outlines = self.outline(cell_id, trap_id)
        return times, np.array([ndimage.morphology.binary_fill_holes(o) for
                                o in outlines])

    def _astype(self, array, type):
        if type == 'outline':
            return np.array(array.todense())
        elif type == 'mask':
            arr = np.array(array.todense())
            return ndimage.binary_fill_holes(arr).astype(int)
        else:
            return array

    def at_time(self, timepoint, type='outline'):
        """Returns the segmentations for all the cells at a given timepoint.

        FIXME: this is extremely hacky and accounts for differently saved
            results in the matlab object. Deprecate ASAP.
        """
        if isinstance(self.trap_info['cell'][timepoint][0], dict):
            segmentations = []
            for x in self.trap_info['cell'][timepoint]:
                seg = x['segmented']
                if not isinstance(seg, np.ndarray):
                    seg = [seg]
                segmentations.append(
                    [self._astype(y, 'outline') for y in seg])
        else:
            segmentations = [[self._astype(y, type) for y in x['segmented']]
                         if x.ndim != 0 else []
                         for x in self.trap_info['cell'][timepoint]]
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
