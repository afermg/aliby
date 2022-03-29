"""
ImageViewer class, used to look at individual or multiple traps over time.


Example of usage:

fpath = "/home/alan/Documents/dev/skeletons/scripts/data/16543_2019_07_16_aggregates_CTP_switch_2_0glu_0_0glu_URA7young_URA8young_URA8old_01/URA8_young018.h5"

trap_id = 9
trange = list(range(0, 30))
ncols = 8

riv = remoteImageViewer(fpath)
riv.plot_labeled_traps(trap_id, trange, ncols)

"""

import yaml
import numpy as np
from PIL import Image
from skimage.morphology import dilation

from agora.io.cells import CellsLinear as Cells
from agora.io.writer import load_attributes
from aliby.io.omero import Image as OImage
from aliby.tile.tiler import Tiler

import matplotlib.pyplot as plt


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


class remoteImageViewer:
    def __init__(self, fpath, server_info=None):
        attrs = load_attributes(fpath)

        self.image_id = attrs.get("image_id")
        assert self.image_id is not None, "No valid image_id found in metadata"

        if server_info is None:
            server_info = yaml.safe_load(attrs["parameters"])["general"]["server_info"]
        self.server_info = server_info

        with OImage(self.image_id, **self.server_info) as image:
            self.tiler = Tiler.from_hdf5(image, fpath)

        self.cells = Cells.from_source(fpath)

    def get_position(self):
        raise (NotImplementedError)

    def get_position_timelapse(self):
        raise (NotImplementedError)

    @property
    def full(self):
        if not hasattr(self, "_full"):
            self._full = {}
        return self._full

    def get_tc(self, tp, channel=None, server_info=None):
        server_info = server_info or self.server_info
        channel = channel or self.tiler.ref_channel

        with OImage(self.image_id, **server_info) as image:
            self.tiler.image = image.data
            return self.tiler.get_tc(tp, channel)

    def find_channels(self, channels):
        channels = channels or self.tiler.ref_channel
        if isinstance(channels, int):
            channels = [channels]
        elif isinstance(channels, list) and isinstance(channels[0], str):
            channels = [self.tiler.channels.index(ch) for ch in channels]

        return channels

    def get_trap_timepoints(self, trap_id, tps, channels=None, server_info=None):
        server_info = server_info or self.server_info
        channels = self.find_channels(channels)

        ch_tps = set([(channels[0], tp) for tp in tps])
        with OImage(self.image_id, **server_info) as image:
            self.tiler.image = image.data
            if ch_tps.difference(self.full.keys()):
                tps = set(tps).difference(self.full.keys())
                for ch, tp in ch_tps:
                    self.full[(ch, tp)] = self.tiler.get_traps_timepoint(
                        tp, channels=[ch], z=[0]
                    )[:, 0, 0, ..., 0]
            requested_trap = {tp: self.full[(ch, tp)] for ch, tp in ch_tps}

            return requested_trap

    def get_labeled_trap(self, trap_id, tps, **kwargs):
        imgs = self.get_trap_timepoints(trap_id, tps, **kwargs)
        imgs_list = [x[trap_id] for x in imgs.values()]
        outlines = [
            self.cells.at_time(tp, kind="edgemask").get(trap_id, []) for tp in tps
        ]
        lbls = [self.cells.labels_at_time(tp).get(trap_id, []) for tp in tps]
        lbld_outlines = [
            np.dstack([mask * lbl for mask, lbl in zip(maskset, lblset)]).max(axis=2)
            if len(lblset)
            else np.zeros_like(imgs_list[0]).astype(bool)
            for maskset, lblset in zip(outlines, lbls)
        ]
        outline_concat = np.concatenate(lbld_outlines, axis=1)
        img_concat = np.concatenate(imgs_list, axis=1)
        return outline_concat, img_concat

    def plot_labeled_channelrows(self, trap_id, channels, trange, **kwargs):
        imgs = {}
        for ch in self.find_channels(channels):
            out, imgs[ch] = self.get_labeled_trap(
                trap_id, trange, channels=[ch], **kwargs
            )

        # dilation makes outlines easier to see
        out = dilation(out).astype(float)
        out[out == 0] = np.nan

        img_set = np.concatenate([v for v in imgs.values()], axis=0)
        tiled_out = np.tile(out, (len(imgs), 1))
        plt.imshow(
            img_set,
            interpolation=None,
            cmap="Greys_r",
        )
        plt.imshow(
            tiled_out,
            cmap="Set1",
            interpolation=None,
        )
        plt.yticks(
            ticks=[self.tiler.tile_size * (i + 0.5) for i in range(len(channels))],
            labels=[
                self.tiler.channels[ch] if isinstance(ch, int) else ch
                for ch in channels
            ],
        )
        plt.xticks(
            ticks=[self.tiler.tile_size * (i + 0.5) for i in range(len(trange))],
            labels=[t for t in trange],
        )
        plt.show()

    def plot_labeled_traps(self, trap_id, trange, ncols, **kwargs):
        """
        Wrapper to plot a single trap over time

        Parameters
        ---------
        :trap_id: int trap identification
        :trange: list list of time points to fetch
        """
        nrows = len(trange) // ncols
        width = self.tiler.tile_size * ncols
        out, img = self.get_labeled_trap(trap_id, trange, **kwargs)

        # dilation makes outlines easier to see
        out = dilation(out).astype(float)
        out[out == 0] = np.nan

        def concat_pad(array):
            return np.concatenate(
                np.array_split(
                    np.pad(
                        array,
                        ((0, 0), (0, array.shape[1] % width)),
                        constant_values=np.nan,
                    ),
                    nrows,
                    axis=1,
                )
            )

        plt.imshow(
            self.concat_pad(img),
            interpolation=None,
            cmap="Greys_r",
        )
        plt.imshow(
            self.concat_pad(out),
            # concat_pad(mask),
            cmap="Set1",
            interpolation=None,
        )
        plt.yticks(
            ticks=[self.tiler.tile_size * (i + 0.5) for i in range(nrows)],
            labels=[trange[0] + ncols * i for i in range(nrows)],
        )
        plt.show()
