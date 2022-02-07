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


from aliby.tile.tiler import Tiler, TilerParameters, TrapLocations
from agora.io.writer import load_attributes


import json

with open("/home/alan/Documents/dev/skeletons/server_info.json", "r") as f:
    # json.dump(
    #     {
    #         "host": "islay.bio.ed.ac.uk",
    #         "username": "upload",
    #         "password": "",
    #     },
    #     f,
    # )
    server_info = json.load(f)


import h5py
from aliby.io.omero import Image

from agora.io.cells import Cells


class remoteImageViewer:
    def __init__(self, fpath):
        with h5py.File(fpath, "r") as f:
            self.image_id = f.attrs.get("image_id", None) or 105146
        # trap_locs = TrapLocations.from_source(fpath)
        with Image(self.image_id, **server_info) as image:
            self.tiler = Tiler.from_hdf5(image, fpath)
        self.cells = Cells.from_source(fpath)
        # if parameters is None:
        #     parameters = TilerParameters.default()

        # with h5py.File(hdf, "r") as f:
        #     # image_id = f.attrs["omero_id"]
        #     image_id = 16543

    def get_position(self):
        pass

    def get_position_timelapse(self):
        pass

    @property
    def full(self):
        if not hasattr(self, "_full"):
            self._full = {}
        return self._full

    def get_tc(self, tp):
        with Image(self.image_id, **server_info) as image:
            self.tiler.image = image.data
            return self.tiler.get_tc(tp, riv.tiler.ref_channel)

    def get_trap_timepoints(self, trap_id, tps):
        with Image(self.image_id, **server_info) as image:
            self.tiler.image = image.data
            if set(tps).difference(self.full.keys()):
                tps = set(tps).difference(self.full.keys())
                for tp in tps:
                    self.full[tp] = self.tiler.get_traps_timepoint(
                        tp, channels=[self.tiler.ref_channel], z=[0]
                    )[:, 0, 0, ..., 0]
            requested_trap = {tp: self.full[tp] for tp in tps}

            return requested_trap

    def get_labeled_trap(self, trap_id, tps):
        imgs = self.get_trap_timepoints(trap_id, tps)
        imgs_list = [x[trap_id] for x in imgs.values()]
        outlines = [
            riv.cells.at_time(tp, kind="edgemask").get(trap_id, []) for tp in tps
        ]
        lbls = [riv.cells.labels_at_time(tp).get(trap_id, []) for tp in tps]
        lbld_outlines = [
            np.dstack([mask * lbl for mask, lbl in zip(maskset, lblset)]).max(axis=2)
            if len(lblset)
            else np.zeros_like(imgs_list[0]).astype(bool)
            for maskset, lblset in zip(outlines, lbls)
        ]
        outline_concat = np.concatenate(lbld_outlines, axis=1)
        img_concat = np.concatenate(imgs_list, axis=1)
        return outline_concat, img_concat


import matplotlib.pyplot as plt

# fpath = "/home/alan/Documents/dev/skeletons/data/2019_07_16_aggregates_CTP_switch_2_0glu_0_0glu_URA7young_URA8young_URA8old_01/2019_07_16_aggregates_CTP_switch_2_0glu_0_0glu_URA7young_URA8young_URA8old_01/URA8_old007.h5"
fpath = "/home/alan/Documents/dev/skeletons/data/2021_11_01_01_Raf_00/2021_11_01_01_Raf_00/d1134002.h5"
riv = remoteImageViewer(fpath)
# pos = riv.get_tc(0)
out, img = riv.get_labeled_trap(9, list(range(0, 30)))
out_bak = out
out = dilation(out).astype(float)
out[out == 0] = np.nan
plt.imshow(
    np.concatenate(np.array_split(img, 6, axis=1)),
    interpolation=None,
    cmap="Greys_r",
)
plt.imshow(
    np.concatenate(np.array_split(out, 6, axis=1)),
    cmap="Set1",
    interpolation=None,
)
plt.show()

concat = lambda a: np.concatenate([x for x in a])
add = lambda a: np.sum(a, axis=0)
# plt.imshow(add(roll(tmp[0], 10), np.roll(roll(tmp[1], 11), 6, axis=0)))
# plt.show()
