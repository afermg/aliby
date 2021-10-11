from typing import Union
import collections
from itertools import groupby, chain, product

import numpy as np
import h5py


class BridgeH5:
    """
    Base class to interact with h5 data stores.
    It also contains functions useful to predict how long should segmentation take.
    """

    def __init__(self, filename, flag="r"):
        self.filename = filename
        if flag is not None:
            self._hdf = h5py.File(filename, flag)

            self._filecheck

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

    def get_n_cellpairs(self, nstepsback=2):
        cell_tree = self.cell_tree
        # get pair of consecutive trap-time points
        pass

    @staticmethod
    def get_consecutives(tree, nstepsback):
        # Receives a sorted tree and returns the keys of consecutive elements
        vals = {k: np.array(list(v)) for k, v in tree.items()}  # get tp level
        where_consec = [
            {
                k: np.where(np.subtract(v[n + 1 :], v[: -n - 1]) == n + 1)[0]
                for k, v in vals.items()
            }
            for n in range(nstepsback)
        ]  # get indices of consecutive elements
        return where_consec

    def get_npairs(self, nstepsback=2, tree=None):
        if tree is None:
            tree = self.cell_tree

        consecutive = self.get_consecutives(tree, nstepsback=nstepsback)
        flat_tree = flatten(tree)

        n_predictions = 0
        for i, d in enumerate(consecutive, 1):
            flat = list(chain(*[product([k], list(v)) for k, v in d.items()]))
            pairs = [(f, (f[0], f[1] + i)) for f in flat]
            for p in pairs:
                n_predictions += len(flat_tree.get(p[0], [])) * len(
                    flat_tree.get(p[1], [])
                )

        return n_predictions

    def get_npairs_over_time(self, nstepsback=2):
        tree = self.cell_tree
        npairs = []
        for t in self._hdf["cell_info"]["processed_timepoints"][()]:
            tmp_tree = {
                k: {k2: v2 for k2, v2 in v.items() if k2 <= t} for k, v in tree.items()
            }
            npairs.append(self.get_npairs(tree=tmp_tree))

        return np.diff(npairs)

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
    # Sorts iterable and returns a dictionary where the values are grouped by the first element.

    iterable = sorted(iterable, key=lambda x: x[0])
    grouped = {k: [x[1:] for x in v] for k, v in groupby(iterable, lambda x: x[0])}
    return grouped


def recursive_groupsort(iterable):
    # Recursive extension of groupsort
    if len(iterable[0]) > 1:
        return {k: recursive_groupsort(v) for k, v in groupsort(iterable).items()}
    else:  # Only two elements in list
        return [x[0] for x in iterable]


def flatten(d, parent_key="", sep="_"):
    """Flatten nested dict. Adapted from https://stackoverflow.com/a/6027615"""
    items = []
    for k, v in d.items():
        new_key = parent_key + (k,) if parent_key else (k,)
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
