"""
ImageViewer class, used to look at individual or multiple traps over time.


Example of usage:

fpath = "/home/alan/Documents/dev/skeletons/scripts/data/16543_2019_07_16_aggregates_CTP_switch_2_0glu_0_0glu_URA7young_URA8young_URA8old_01/URA8_young018.h5"

trap_id = 9
trange = list(range(0, 30))
ncols = 8

riv = remoteImageViewer(fpath)
riv.plot_labelled_trap(trap_id, trange, [0], ncols=ncols)

"""

import re
import typing as t

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from agora.io.cells import Cells
from agora.io.writer import load_attributes
from PIL import Image
from skimage.morphology import dilation

from aliby.io.image import Image as OImage
from aliby.tile.tiler import Tiler
from aliby.tile.traps import stretch_image

default_colours = {
    "Brightfield": "Greys_r",
    "GFP": "Greens_r",
    "mCherry": "Reds_r",
    "cell_label": sns.color_palette("Paired", as_cmap=True),
}


def custom_imshow(a, norm=None, cmap=None, *args, **kwargs):
    """
    Wrapper on plt.imshow function.
    """
    if cmap is None:
        cmap = "Greys_r"
    return plt.imshow(
        a,
        *args,
        cmap=cmap,
        interpolation=None,
        interpolation_stage="rgba",
        **kwargs,
    )


class localImageViewer:
    """
    Fast access to Images segmented locally without tiling
    from image.h5 objects.
    """

    def __init__(self, h5file):
        self._hdf = h5py.File(h5file)
        self.positions = list(self._hdf.keys())
        self.current_position = self.positions[0]

    def plot_position(self, channel=0, tp=0, z=0, stretch=True):
        pixvals = self._hdf[self.current_position][channel, tp, ..., z]
        if stretch:
            minval = np.percentile(pixvals, 0.5)
            maxval = np.percentile(pixvals, 99.5)
            pixvals = np.clip(pixvals, minval, maxval)
            pixvals = ((pixvals - minval) / (maxval - minval)) * 255

        Image.fromarray(pixvals.astype(np.uint8))


class remoteImageViewer:
    """
    This ImageViewer combines fetching remote images with tiling and outline display.
    """

    def __init__(self, fpath, server_info=None):
        self._fpath = fpath
        attrs = load_attributes(fpath)

        self.image_id = attrs.get("image_id")

        assert self.image_id is not None, "No valid image_id found in metadata"

        if server_info is None:
            server_info = attrs["parameters"]["general"]["server_info"]
        self.server_info = server_info

        with OImage(self.image_id, **self.server_info) as image:
            self.tiler = Tiler.from_hdf5(image, fpath)

        self.cells = Cells.from_source(fpath)

    @property
    def shape(self):
        return self.tiler.image.shape

    @property
    def ntraps(self):
        return self.cells.ntraps

    @property
    def max_labels(self):
        # Print max cell label in whole experiment
        return [max(x) for x in self.cells.labels]

    def labels_at_time(self, tp: int):
        # Print  cell label at a given time-point
        return self.cells.labels_at_time(tp)

    def random_valid_trap_tp(
        self,
        min_ncells: int = None,
        min_consecutive_tps: int = None,
        label_modulo: int = None,
    ):
        # Call Cells convenience function to pick a random trap and tp
        # containing cells for x cells for y
        return self.cells.random_valid_trap_tp(
            min_ncells=min_ncells,
            min_consecutive_tps=min_consecutive_tps,
            # label_modulo=label_modulo,
        )

    def get_entire_position(self):
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

    def _find_channels(self, channels: str, guess: bool = True):
        channels = channels or self.tiler.ref_channel
        if isinstance(channels, (int, str)):
            channels = [channels]
        if isinstance(channels[0], str):
            if guess:
                channels = [self.tiler.channels.index(ch) for ch in channels]
            else:
                channels = [
                    re.search(ch, tiler_channels)
                    for ch in channels
                    for tiler_channels in self.tiler.channels
                ]

        return channels

    def get_pos_timepoints(
        self,
        tps: t.Union[int, t.Collection[int]],
        channels: t.Union[str, t.Collection[str]] = None,
        z: int = None,
        server_info=None,
        **kwargs,
    ):

        if tps and not isinstance(tps, t.Collection):
            tps = range(tps)

        # TODO add support for multiple channels or refactor
        if channels and not isinstance(channels, t.Collection):
            channels = [channels]

        if z is None:
            z = 0

        server_info = server_info or self.server_info
        channels = 0 or self._find_channels(channels)
        z = z or self.tiler.ref_z

        ch_tps = [(channels[0], tp) for tp in tps]
        with OImage(self.image_id, **server_info) as image:
            self.tiler.image = image.data
            # if ch_tps.difference(self.full.keys()):
            # tps = set(tps).difference(self.full.keys())
            for ch, tp in ch_tps:
                if (ch, tp) not in self.full:
                    self.full[(ch, tp)] = self.tiler.get_tiles_timepoint(
                        tp, channels=[ch], z=[z]
                    )[:, 0, 0, ..., z]
            requested_trap = {tp: self.full[(ch, tp)] for ch, tp in ch_tps}

            return requested_trap

    def get_labelled_trap(
        self,
        trap_id: int,
        tps: t.Union[range, t.Collection[int]],
        channels=None,
        concatenate=True,
        **kwargs,
    ) -> t.Tuple[np.array]:
        """
        Core method to fetch traps and labels together
        """
        imgs = self.get_pos_timepoints(tps, channels=channels, **kwargs)
        imgs_list = [x[trap_id] for x in imgs.values()]
        outlines = [
            self.cells.at_time(tp, kind="edgemask").get(trap_id, [])
            for tp in tps
        ]
        lbls = [self.cells.labels_at_time(tp).get(trap_id, []) for tp in tps]
        lbld_outlines = [
            np.dstack([mask * lbl for mask, lbl in zip(maskset, lblset)]).max(
                axis=2
            )
            if len(lblset)
            else np.zeros_like(imgs_list[0]).astype(bool)
            for maskset, lblset in zip(outlines, lbls)
        ]
        if concatenate:
            lbld_outlines = np.concatenate(lbld_outlines, axis=1)
            imgs_list = np.concatenate(imgs_list, axis=1)
        return lbld_outlines, imgs_list

    def get_images(self, trap_id, trange, channels, **kwargs):
        """
        Wrapper to fetch images
        """
        out = None
        imgs = {}

        for ch in self._find_channels(channels):
            out, imgs[ch] = self.get_labelled_trap(
                trap_id, trange, channels=[ch], **kwargs
            )
        return out, imgs

    def plot_labelled_trap(
        self,
        trap_id: int,
        channels,
        trange: t.Union[range, t.Collection[int]],
        remove_axis: bool = False,
        savefile: str = None,
        skip_outlines: bool = False,
        norm: str = None,
        ncols: int = None,
        local_colours: bool = True,
        img_plot_kwargs: dict = {},
        lbl_plot_kwargs: dict = {"alpha": 0.8},
        **kwargs,
    ):
        """Wrapper to plot time-lapses of individual traps

        Use Cells and Tiler to generate images of cells with their resulting
        outlines.

        Parameters
        ----------
        trap_id : int
            Identifier of trap
        channels : Union[str, int]
            Channels to use
        trange : t.Union[range, t.Collection[int]]
            Range or collection indicating the time-points to use.
        remove_axis : bool
            None, "off", or "x". Determines whether to remove the x-axis, both
            axes or none.
        savefile : str
            Saves file to a location.
        skip_outlines : bool
            Do not add overlay with outlines
        norm : str
            Normalise signals
        ncols : int
            Number of columns to plot.
        local_colours : bool
            Bypass label indicators to guarantee that colours are not repeated
            (TODO implement)
        img_plot_kwargs : dict
            Arguments to pass to plt.imshow used for images.
        lbl_plot_kwargs : dict
            Keyword arguments to pass to label plots.
        **kwargs : dict
            Additional keyword arguments passed to ImageViewer.get_images.

        Examples
        --------
        FIXME: Add docs.

        """
        if ncols is None:
            ncols = len(trange)
        nrows = int(np.ceil(len(trange) / ncols))
        width = self.tiler.tile_size * ncols

        out, images = self.get_images(trap_id, trange, channels, **kwargs)

        # dilation makes outlines easier to see
        out = dilation(out).astype(float)
        out[out == 0] = np.nan

        channel_labels = [
            self.tiler.channels[ch] if isinstance(ch, int) else ch
            for ch in channels
        ]

        assert not norm or norm in (
            "l1",
            "l2",
            "max",
        ), "Invalid norm argument."

        if norm and norm in ("l1", "l2", "max"):
            images = {k: stretch_image(v) for k, v in images.items()}

        images = [concat_pad(img, width, nrows) for img in images.values()]
        # TODO convert to RGB to draw fluorescence with colour
        tiled_imgs = {}
        tiled_imgs["img"] = np.concatenate(images, axis=0)
        tiled_imgs["cell_labels"] = np.concatenate(
            [concat_pad(out, width, nrows) for _ in images], axis=0
        )

        custom_imshow(
            tiled_imgs["img"],
            **img_plot_kwargs,
        )
        custom_imshow(
            tiled_imgs["cell_labels"],
            cmap=sns.color_palette("Paired", as_cmap=True),
            **lbl_plot_kwargs,
        )

        if remove_axis is True:
            plt.axis("off")
        elif remove_axis == "x":
            plt.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )

        if remove_axis != "True":
            plt.yticks(
                ticks=[
                    (i * self.tiler.tile_size * nrows)
                    + self.tiler.tile_size * nrows / 2
                    for i in range(len(channels))
                ],
                labels=channel_labels,
            )

        if not remove_axis:
            xlabels = (
                ["+ {} ".format(i) for i in range(ncols)]
                if nrows > 1
                else list(trange)
            )
            plt.xlabel("Time-point")

            plt.xticks(
                ticks=[self.tiler.tile_size * (i + 0.5) for i in range(ncols)],
                labels=xlabels,
            )

        if not np.any(out):
            print("ImageViewer:Warning:No cell outlines found")

        if savefile:
            plt.savefig(savefile, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()


def concat_pad(a: np.array, width, nrows):
    """
    Melt an array into having multiple blocks as rows
    """
    return np.concatenate(
        np.array_split(
            np.pad(
                a,
                # ((0, 0), (0, width - (a.shape[1] % width))),
                ((0, 0), (0, a.shape[1] % width)),
                constant_values=np.nan,
            ),
            nrows,
            axis=1,
        )
    )
