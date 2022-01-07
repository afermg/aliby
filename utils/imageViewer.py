"""This is an example module to show the structure."""
from typing import Union

import numpy as np
from PIL import Image


class localImageViewer:
    """
    This class is used to quickly access position images without tiling
    from image.h5 objects.
    """

    def __init__(self, h5file):
        """This class takes one parameter and is used to add one to that
        parameter.

        :param parameter: The parameter for this class
        """
        self._hdf = h5py.File(h5file)
        self.positions = list(self._hdf.keys())
        self.current_position = self.positions[0]
        self.parameter = parameter

    def plot_position(channel=0, tp=0, z=0, stretch=True):
        pixvals = self._hdf[self.current_position][channel, tp, ..., z]
        if stretch:
            minval = np.percentile(pixvals, 0.5)
            maxval = np.percentile(pixvals, 99.5)
            pixvals = np.clip(pixvals, minval, maxval)
            pixvals = ((pixvals - minval) / (maxval - minval)) * 255

        Image.fromarray(pixvals.astype(np.uint8))
