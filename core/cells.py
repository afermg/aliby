import h5py
import numpy as np
import scipy
from scipy import ndimage

from core.io.matlab import matObject


def cell_factory(store, type='matlab'):
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


def is_or_in(item, arr):
    if isinstance(arr, (list, np.ndarray)):
        return item in arr
    else:
        return item == arr


class CellsHDF(Cells):
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
