from typing import Union
from itertools import groupby

import h5py


class BridgeH5:
    def __init__(self, file):
        self._hdf = h5py.File(file, "r")

        self._filecheck()

    def _filecheck(self):
        assert "cell_info" in self._hdf, "Invalid file. No 'cell_info' found."

    def close(self):
        self._hdf.close()

    def max_ncellpairs(self, nstepsback):
        """
        Get maximum number of cell pairs to be calculated
        """

        dset = self._hdf["cell_info"][()]
        # attrs = self._hdf[dataset].attrs
        pass

    @property
    def cell_tree(self):
        return self.get_info_tree()

    def get_info_tree(
        self, fields: Union[tuple, list] = ("trap", "timepoint", "cell_label")
    ):
        """
        Returns traps, time points and labels for this position in form of a tree
        in the hierarchy determined by the argument fields. Note that it is
        compressed to non-empty elements and timepoints.

        Default hierarchy is:
        - trap
         - time point
          - cell label

        This function currently produces trees of depth 3, but it can easily be
        extended for deeper trees if needed (e.g. considering groups,
        chambers and/or positions).

        input
        :fields: Fields to fetch from 'cell_info' inside the hdf5 storage

        returns
        :tree: Nested dictionary where keys (or branches) are the upper levels
             and the leaves are the last element of :fields:.
        """
        zipped_info = (*zip(*[self._hdf["cell_info"][f][()] for f in fields]),)

        return recursive_groupsort(zipped_info)


def groupsort(iterable: Union[tuple, list]):
    # Groups a list or tuple by the first element and returns
    # a dictionary that follows {v[0]:sorted(v[1:]) for v in iterable}.
    # Sorted by the first element in the remaining values

    return {k: [x[1:] for x in v] for k, v in groupby(iterable, lambda x: x[0])}


def recursive_groupsort(iterable):
    # Recursive extension of groupsort
    if len(iterable[0]) > 1:
        return {k: recursive_groupsort(v) for k, v in groupsort(iterable).items()}
    else:  # Only two elements in list
        return [x[0] for x in iterable]
