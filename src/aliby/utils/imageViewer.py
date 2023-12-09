import re
import typing as t
from abc import ABC
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.morphology import dilation

from agora.io.cells import Cells
from agora.io.metadata import dispatch_metadata_parser
from aliby.io.image import dispatch_image

from aliby.tile.tiler import Tiler
from aliby.utils.plot import stretch_clip

default_colours = {
    "Brightfield": "Greys_r",
    "GFP": "Greens_r",
    "mCherry": "Reds_r",
    "cell_label": sns.color_palette("Paired", as_cmap=True),
}


def custom_imshow(a, norm=None, cmap=None, *args, **kwargs):
    """Wrap plt.imshow."""
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


class BaseImageViewer(ABC):
    """Base class with routines common to all ImageViewers."""

    def __init__(self, h5file_path):
        """Initialise from a Path to a h5 file."""
        self.h5file_path = h5file_path
        self.logfiles_meta = dispatch_metadata_parser(h5file_path.parent)
        self.image_id = self.logfiles_meta.get("image_id")
        if self.image_id is None:
            with h5py.File(h5file_path, "r") as f:
                self.image_id = f.attrs.get("image_id")
        if self.image_id is None:
            raise ("No valid image_id found in metadata.")
        self.full = {}

    @property
    def shape(self):
        """Return shape of image array."""
        return self.tiler.image.shape

    @property
    def ntraps(self):
        """Find the number of traps available."""
        return self.cells.ntraps

    @property
    def max_labels(self):
        """Find maximum cell label in whole experiment."""
        return [max(x) for x in self.cells.labels]

    def labels_at_time(self, tp: int):
        """Find cell label at a given time point."""
        return self.cells.labels_at_time(tp)

    def find_channel_indices(
        self, channels: t.Union[str, t.Collection[str]], guess=True
    ):
        """Find index for particular channels."""
        channels = channels or self.tiler.ref_channel
        if isinstance(channels, (int, str)):
            channels = [channels]
        if isinstance(channels[0], str):
            if guess:
                indices = [self.tiler.channels.index(ch) for ch in channels]
            else:
                indices = [
                    re.search(ch, tiler_channels)
                    for ch in channels
                    for tiler_channels in self.tiler.channels
                ]
            return indices
        else:
            return channels

    def get_outlines_tiles_dict(self, tile_id, trange, channels):
        """Get outlines and dict of tiles with channel indices as keys."""
        outlines = None
        tile_dict = {}
        for ch in self.find_channel_indices(channels):
            outlines, tile_dict[ch] = self.get_outlines_tiles(
                tile_id, trange, channels=[ch]
            )
        return outlines, tile_dict

    def get_outlines_tiles(
        self,
        tile_id: int,
        tps: t.Union[range, t.Collection[int]],
        channels=None,
        concatenate=True,
        **kwargs,
    ) -> t.Tuple[np.array]:
        """
        Get masks uniquely labelled for each cell with the corresponding tiles.

        Returns a list of masks, each an array with distinct masks for each cell,
        and an array of tiles for the given channel.
        """
        tile_dict = self.get_tiles(tps, channels=channels, **kwargs)
        # get tiles of interest
        tiles = [x[tile_id] for x in tile_dict.values()]
        # get outlines for each time point
        outlines = [
            self.cells.at_time(tp, kind="edgemask").get(tile_id, []) for tp in tps
        ]
        # get cell labels for each time point
        cell_labels = [self.cells.labels_at_time(tp).get(tile_id, []) for tp in tps]
        # generate one image with all cell outlines uniquely labelled per tile
        labelled_outlines = [
            np.stack(
                [outline * label for outline, label in zip(outlines_tp, labels_tp)]
            ).max(axis=0)
            if len(labels_tp)
            else np.zeros_like(tiles[0]).astype(bool)
            for outlines_tp, labels_tp in zip(outlines, cell_labels)
        ]
        if concatenate:
            # concatenate to allow potential image processing
            labelled_outlines = np.concatenate(labelled_outlines, axis=1)
            tiles = np.concatenate(tiles, axis=1)
        return labelled_outlines, tiles

    def get_tiles(
        self,
        tps: t.Union[int, t.Collection[int]],
        channels: None,
        z: int = None,
    ):
        """Get dict with time points as keys and all available tiles as values."""
        if tps and not isinstance(tps, t.Collection):
            tps = range(tps)
        if z is None:
            z = 0
        z = z or self.tiler.ref_z
        channel_indices = self.find_channel_indices(channels)
        ch_tps = [(channel_indices[0], tp) for tp in tps]
        for ch, tp in ch_tps:
            if (ch, tp) not in self.full:
                self.full[(ch, tp)] = self.tiler.get_tiles_timepoint(
                    tp, channels=[ch], z=[z]
                )[:, 0, 0, z, ...]
        tile_dict = {tp: self.full[(ch, tp)] for ch, tp in ch_tps}
        return tile_dict

    def plot_labelled_trap(
        self,
        tile_id,
        channels,
        trange: t.Union[range, t.Collection[int]],
        remove_axis: bool = False,
        skip_outlines: bool = False,
        norm=True,
        ncols: int = None,
        local_colours: bool = True,
        img_plot_kwargs: dict = {},
        lbl_plot_kwargs: dict = {"alpha": 0.8},
        **kwargs,
    ):
        """
        Plot time-lapses of individual tiles.

        Use Cells and Tiler to generate images of cells with their resulting
        outlines.

        Parameters
        ----------
        tile_id : int
            Identifier of trap
        channel : Union[str, int]
            Channels to use
        trange : t.Union[range, t.Collection[int]]
            Range or collection indicating the time-points to use.
        remove_axis : bool
            None, "off", or "x". Determines whether to remove the x-axis, both
            axes or none.
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
        """
        # set up for plotting
        if ncols is None:
            ncols = len(trange)
        nrows = int(np.ceil(len(trange) / ncols))
        width = self.tiler.tile_size * ncols
        outlines, tiles_dict = self.get_outlines_tiles_dict(tile_id, trange, channels)
        channel_labels = [
            self.tiler.channels[ch] if isinstance(ch, int) else ch for ch in channels
        ]
        # dilate to make outlines easier to see
        outlines = dilation(outlines).astype(float)
        outlines[outlines == 0] = np.nan
        # split concatenated tiles into one tile per time point in a row
        tiles = [
            into_image_time_series(tile, width, nrows) for tile in tiles_dict.values()
        ]
        # TODO convert to RGB to draw fluorescence with colour
        res = {}
        # concatenate different channels vertically for display
        res["tiles"] = np.concatenate(tiles, axis=0)
        res["cell_labels"] = np.concatenate(
            [into_image_time_series(outlines, width, nrows) for _ in tiles], axis=0
        )
        custom_imshow(res["tiles"], **img_plot_kwargs)
        custom_imshow(
            res["cell_labels"], cmap=default_colours["cell_label"], **lbl_plot_kwargs
        )
        if remove_axis is True:
            plt.axis("off")
        elif remove_axis == "x":
            plt.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
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
                ["+ {} ".format(i) for i in range(ncols)] if nrows > 1 else list(trange)
            )
            plt.xlabel("Time-point")
            plt.xticks(
                ticks=[self.tiler.tile_size * (i + 0.5) for i in range(ncols)],
                labels=xlabels,
            )
        if not np.any(outlines):
            print("ImageViewer:Warning: No cell outlines found.")
        plt.tight_layout()
        plt.show(block=False)


class LocalImageViewer(BaseImageViewer):
    """
    View images from local files.

    File are either zarr or organised in directories.
    """

    def __init__(self, h5file: str, image_direc: str):
        """Initialise using a h5file and a local directory of images."""
        h5file_path = Path(h5file)
        image_direc_path = Path(image_direc)
        super().__init__(h5file_path)
        with dispatch_image(image_direc_path)(image_direc_path) as image:
            self.tiler = Tiler.from_h5(image, h5file_path)
        self.cells = Cells.from_source(h5file_path)


class RemoteImageViewer(BaseImageViewer):
    """Fetching remote images with tiling and outline display."""

    credentials = ("host", "username", "password")

    def __init__(self, h5file: str, server_info: t.Dict[str, str]):
        """Initialise using a h5file and importing aliby.io.omero."""
        from aliby.io.omero import UnsafeImage as OImage

        h5file_path = Path(h5file)
        super().__init__(h5file_path)
        self.server_info = server_info or {
            k: self.attrs["parameters"]["general"][k] for k in self.credentials
        }
        image = OImage(self.image_id, **self._server_info)
        self.tiler = Tiler.from_h5(image, h5file_path)
        self.cells = Cells.from_source(h5file_path)


def into_image_time_series(a: np.array, width, nrows):
    """Split into sub-arrays and then concatenate into one."""
    return np.concatenate(
        np.array_split(
            np.pad(
                a,
                ((0, 0), (0, a.shape[1] % width)),
                constant_values=np.nan,
            ),
            nrows,
            axis=1,
        )
    )
