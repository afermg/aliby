"""
HDF5 input and output from pre-defined formats.

Includes reading Mat Objects that were saved as HDF5.
"""
import io

import h5py
from core.io.matlab import describe

class hdf_object:
    def __init__(self, filename):
        f = h5py.File(filename, 'r')

    def __getitem__(self, item):
        pass


class hdf_dict:
    def __init__(self, group):
        self.group = group

    def __getitem__(self, item):
        try:  # Get a dataset
            value = self.group[item]
            if isinstance(value, h5py.Group):
                return hdf_dict(self.group[item])
            else:
                return self.group[item]
        except KeyError:
            pass  # Default to attributes instead
        try:
            return self.group.attrs[item]
        except KeyError:
            raise KeyError(f'The object does not have attribute {item}')

    def keys(self):
        return list(self.group.keys()) + list(self.group.attrs.keys())

    def items(self):
        for k in self.keys():
            yield k, self[k]

