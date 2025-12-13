"""
Tiler: Divides images into smaller tiles.

The tasks of the Tiler are selecting regions of interest, or tiles, of
images - with one trap per tile, correcting for the drift of the microscope
stage over time, and handling errors and bridging between the image data
and Aliby’s image-processing steps.

Tiler subclasses deal with either network connections or local files.
There is a special MonoTile class for the case when the whole
image is to be processed.

To find tiles, we use a two-step process: we analyse the bright-field image
to produce the template of a trap, and we fit this template to the image to
find the tiles' centres.

We use texture-based segmentation (entropy) to split the image into
foreground -- cells and traps -- and background, which we then identify with
an Otsu filter. Two methods are used to produce a template trap from these
regions: pick the trap with the smallest minor axis length and average over
all validated traps.

A peak-identifying algorithm recovers the x and y-axis location of traps in
the original image, and we choose the approach to template that identifies
the most tiles.

The experiment is stored as an array with our standard indexing order of
(Time, Channels, Z-stack, X, Y).
"""

import logging
import re
import typing as t
import warnings
from functools import lru_cache, partial
from itertools import product
from typing import Callable

import dask.array as da
import numpy as np
from skimage.registration import phase_cross_correlation

from agora.abc import ParametersABC, StepABC
from aliby.tile.process_traps import segment_traps
from aliby.tile.tiles import TileLocations


class TilerParameters(ParametersABC):
    """Define default values for tile size and the reference channels."""

    _defaults = {
        "tile_size": 117,
        "ref_channel": 0,
        "ref_z": 0,
        "track_drift": True,
    }


def dispatch_tiler(kind: str, kwargs: dict) -> Callable:
    """Returns a Tiler class constructor that requires an Image (class)."""
    match kind:
        case "crop":
            tiler = CropTiler
        case _:
            tiler = Tiler
    return partial(tiler.from_image, parameters=TilerParameters(**kwargs))


def tile_last2(pix: np.ndarray, tile_size: int):
    """
    Parameters
    ----------
    pix:  any numpy array with ndim ≥ 2
    tile_size : extent of the tiles in both x and y-axes.
    returns: Array with one extra dimension
        shape == pix.shape[:-2] + (n_tiles, tile_h, tile_w)
    """

    *lead, H, W = pix.shape

    # number of tiles along each axis
    n_th = (H - tile_size) // tile_size + 1
    n_tw = (W - tile_size) // tile_size + 1
    n_tiles = n_th * n_tw

    # pre-allocate output
    tiles = np.empty((n_tiles, *lead, tile_size, tile_size), dtype=pix.dtype)

    # fill slices
    idx = 0
    for i in range(0, H - tile_size + 1, tile_size):
        for j in range(0, W - tile_size + 1, tile_size):
            sl = (slice(None),) * (pix.ndim - 2) + (
                slice(i, i + tile_size),
                slice(j, j + tile_size),
            )
            tiles[idx] = pix[sl]
            idx += 1
    return tiles


class CropTiler(StepABC):
    """Tiler that crops the input image."""

    def __init__(self, pixels: da.core.Array, tile_size: int, **kwargs):
        self.pixels = pixels
        self.tile_size = tile_size

    @classmethod
    def from_image(cls, image, parameters):
        return cls(image.data, **parameters.to_dict())

    def get_fczyx(self, tp: int, tile_size: int) -> np.ndarray:
        """
        Load multidimensional image for a given time point.
        Note that this one does not apply image tracking.
        """
        pix = self.pixels[tp]
        if hasattr(pix, "compute"):
            # if using dask fetch images
            pix = pix.compute(scheduler="synchronous")
        tiles = tile_last2(pix, tile_size)

        return tiles

    def _run_tp(self, tp: int):
        return {
            "pixels": self.get_fczyx(tp, tile_size=self.tile_size),
        }


class Tiler(StepABC):
    """
    Divide images into smaller tiles for faster processing.

    Find tiles and re-register images if they drift.
    Fetch images from an OMERO server if necessary.

    Uses an Image instance, which lazily provides the pixel data,
    and, as an independent argument, meta.
    """

    def __init__(
        self,
        pixels: da.core.Array,
        meta: dict,
        parameters: TilerParameters,
        tile_locations=None,
        **kwargs,
    ):
        """
        Initialise.

        Parameters
        ----------
        pixels: Numerical values of image
        meta: dictionary
        parameters: an instance of TilerParameters
        tile_locs: (optional)
        **kwargs
        """
        super().__init__(parameters)
        self.pixels = pixels

        params_d = parameters.to_dict()

        # if "position_name" in params_d:  # SwainLab Experiment
        # from aliby.tile.meta import find_channel_swainlab

        # self.channels = find_channel_swainlab(meta, params_d["position_name"])
        # self.position_name = params_d["position_name"]
        # # get reference channel - used for segmentation
        # if "zsections" in meta:
        #     self.z_perchannel = {
        #         ch: zsect for ch, zsect in zip(self.channels, meta["zsections"])
        #     }
        # else:  # Data with little metadata
        # self.channels = meta.get("channels", list(range(pixels.shape[-4])))
        self.channels = list(range(pixels.shape[-4]))
        self.tile_size = self.tile_size or self.pixels.shape[-2:]
        # get reference channel - used for segmentation
        ref_channel_index = parameters.ref_channel
        if isinstance(ref_channel_index, str):
            ref_channel_index = self.channels.index(parameters.ref_channel)
        self.ref_channel_index = ref_channel_index
        self.tile_locs = tile_locations

    @classmethod
    def from_image(
        cls,
        image,
        parameters: TilerParameters,
        **kwargs,
    ):
        """
        Instantiate from an Image instance.

        Parameters
        ----------
        image: an instance of Image
        parameters: an instance of TilerPameters
        """
        return cls(
            image.data,
            image.meta,
            parameters,
            **kwargs,
        )

    @property
    def no_processed(self) -> int:
        """Return the number of processed images."""
        if not hasattr(self, "_no_processed"):
            self._no_processed = 0
        return self._no_processed

    @no_processed.setter
    def no_processed(self, value):
        self._no_processed = value

    def find_drift(self, tp: int):
        """
        Find any translational drift between two images.

        Use cross correlation between two consecutive images.

        Arguments
        ---------
        tp: integer
            Index for a time point.
        """
        ref_z = getattr(self, "ref_z", 0)
        prev_tp = max(0, tp - 1)

        # cross-correlate
        drift, _, _ = phase_cross_correlation(
            self.pixels[prev_tp, self.ref_channel_index, ref_z],
            self.pixels[tp, self.ref_channel_index, ref_z],
        )
        # store drift
        if 0 < tp < len(self.tile_locs.drifts):
            self.tile_locs.drifts[tp] = drift.tolist()
        else:
            self.tile_locs.drifts.append(drift.tolist())

    def get_fczyx(
        self,
        tp: int,
        drift: bool = True,
    ) -> np.ndarray:
        """
        Return all tiles corrected for drift.

        Parameters
        ----------
        tp: integer
            An index for a time point

        Returns
        ----------
        Numpy ndarray of tiles with shape (#tiles,channels, z-sections, y, x)
        """
        channels = []
        nch = range(self.pixels.shape[-4])
        for ch in nch:
            tiles = self.get_tp_data(tp, ch)
            channels.append(tiles)

        cfzyx = np.array(channels)
        return np.swapaxes(cfzyx, 0, 1)

    def get_tp_data(
        self,
        tp: int,
        c: int,
        drift: bool = True,
    ) -> np.ndarray:
        """
        Return all tiles corrected for drift.

        Parameters
        ----------
        tp: integer
            An index for a time point
        c: integer
            An index for a channel

        Returns
        ----------
        Numpy ndarray of tiles with shape (no tiles, z-sections, y, x)
        """
        tiles = []
        full = self.load_image(tp, c)
        for tile in self.tile_locs:
            if drift:
                # pad tile if necessary
                tiled_pixels = if_out_of_bounds_pad(full, tile.as_range(tp))

            tiles.append(tiled_pixels)
        return np.stack(tiles)

    def get_tile_data(self, tile_id: int, tp: int, c: int) -> np.ndarray:
        """
        Return a tile corrected for drift and padding.

        Parameters
        ----------
        tile_id: integer
            Index of tile.
        tp: integer
            Index of time points.
        c: integer
            Index of channel.

        Returns
        -------
        ndtile: array
            An array of (x, y) arrays, one for each z stack
        """
        full = self.load_image(tp, c)
        tile = self.tile_locs.tiles[tile_id]

        ndtile = if_out_of_bounds_pad(full, tile.as_range(tp))

        return ndtile

    def _run_tp(self, tp: int):
        """
        Find tiles for a given time point.

        Determine any translational drift of the current image from the
        previous one.

        Arguments
        ---------
        tp: integer
            The time point to tile.
        """
        ref_z = getattr(self, "ref_z", 0)
        if self.no_processed == 0:
            initial_image = self.pixels[0, self.ref_channel_index, ref_z]
            self.tile_locs = set_areas_of_interest(
                initial_image,
                self.tile_size,
            )

        if hasattr(self.tile_locs, "drifts"):
            drift_len = len(self.tile_locs.drifts)
            if self.no_processed != drift_len:
                warnings.warn(
                    "Tiler: the number of processed tiles and the number of drifts"
                    " calculated do not match."
                )
                self.no_processed = drift_len

        if not hasattr(self, "calculate_drift"):
            self.calculate_drift = False

        if self.calculate_drift:
            # determine drift for this time point and update tile_locs.drifts
            self.find_drift(tp)
        else:
            drift = [0.0, 0.0]
            # store drift
            if 0 < tp < len(self.tile_locs.drifts):
                self.tile_locs.drifts[tp] = drift
            else:
                self.tile_locs.drifts.append(drift)

        # update no_processed
        self.no_processed = tp + 1
        # return result for writer
        return {
            "drift": self.tile_locs.to_dict(tp),
            "pixels": self.get_fczyx(tp),
        }

    def get_pixels(self, tp: int) -> np.ndarray:
        """
        Load multidimensional image for a given time point.
        Note that this one does not apply image tracking.
        """
        # full = self.pixels[tp]
        tiles = self.get_tp_data(tp)
        if hasattr(tiles, "compute"):
            # if using dask fetch images
            tiles = tiles.compute(scheduler="synchronous")
        return tiles

    @lru_cache(maxsize=2)
    def load_image(self, tp: int, c: int) -> np.ndarray:
        """
        Load image using dask.

        Assumes the image is arranged as
            no of time points
            no of channels
            no of z stacks
            no of pixels in y direction
            no of pixels in x direction

        Parameters
        ----------
        tp: integer
            An index for a time point
        c: integer
            An index for a channel

        Returns
        -------
        full: an array of images
        """
        full = self.pixels[tp, c]
        if hasattr(full, "compute"):
            # if using dask fetch images
            full = full.compute(scheduler="synchronous")
        return full

    @property
    def shape(self):
        """
        Return the shape of the image array.

        The image array is arranged as number of images, number of channels,
        number of z sections, and size of the image in y and x.
        """
        return self.pixels.shape

    def get_tiles_timepoint(self, tp: int, channels=None, z: int = 0) -> np.ndarray:
        """
        Get a multidimensional array with all tiles for a set of channels
        and z-stacks.

        Used by extractor.

        Parameters
        ---------
        tp: int
            Index of time point
        tile_shape: int or tuple of two ints
            Size of tile in x and y dimensions
        channels: string or list of strings
            Names of channels of interest
        z: int
            Index of z-channel of interest

        Returns
        -------
        res: array
            Data arranged as (tiles, channels, Z, X, Y)
        """
        if channels is None:
            channels = [0]
        elif isinstance(channels, str):
            channels = [channels]
        # convert to indices
        channels = [
            (self.channels.index(channel) if isinstance(channel, str) else channel)
            for channel in channels
        ]
        # get the data as a list of length of the number of channels
        res = []
        for c in channels:
            # only return requested z
            tiles = self.get_tp_data(tp, c)[:, z]
            # insert new axis at index 1 for missing time point
            tiles = np.expand_dims(tiles, axis=1)
            res.append(tiles)
        # stack at time-point axis if more than one channel
        tiles_tp = np.stack(res, axis=1)
        return tiles_tp


def run(self, time_dim: int or None = None):
    """
    Tile all time points in an experiment at once.
    If no time dimension is provided it assumes it is the first one.
    """
    if time_dim is None:
        time_dim = 0
    for frame in range(self.pixels.shape[time_dim]):
        self.run_tp(frame)
    return None


def get_channel_index(self, channel: str or int) -> int or None:
    """
    Find index for channel using regex.

    If channels are strings, return the first matched string.
    If channels are integers, return channel unchanged if it is
    an integer.

    Parameters
    ----------
    channel: string or int
        The channel or index to be used.
    """
    if isinstance(channel, int) and all(
        map(lambda x: isinstance(x, int), self.channels)
    ):
        return channel
    elif isinstance(channel, str):
        return find_channel_index(self.channels, channel)
    else:
        return None


def find_channel_index(image_channels: t.List[str], channel_regex: str):
    """Use a regex to find the index of a channel."""
    for index, ch in enumerate(image_channels):
        found = re.match(channel_regex, ch, re.IGNORECASE)
        if found:
            if len(found.string) - (found.endpos - found.start()):
                logging.getLogger("aliby").log(
                    logging.WARNING,
                    f"Channel {channel_regex} matched {ch} using regex",
                )
            return index


def find_channel_name(image_channels: t.List[str], channel_regex: str):
    """Find the name of the channel using regex."""
    index = find_channel_index(image_channels, channel_regex)
    if index is not None:
        return image_channels[index]


def if_out_of_bounds_pad(
    pixels: np.ndarray, slices: tuple[slice], max_padding: float = 0.25
):
    """
    Pad slices if out of bounds.

    Parameters
    ----------
    pixels: array
        Slice of image (zstacks, x, y) - the entire position
        with zstacks as first axis
    slices: tuple of two slices
        Delineates indices for the x- and y- ranges of the tile.
    max_padding: float
        Maximum fraction of `pixels` in any dimension that must not be padded.
        If more than `max_padding` must be padded the entire tile is converted to NaNs.

    Returns
    -------
    tile: array
        A tile with all z stacks for the given slices.
        If some padding is needed, the median of the image is used.
        If much padding is needed, a tile of NaN is returned.
    """
    # number of pixels in the yx axes
    max_yx = pixels.shape[-2:]
    # ignore parts of the tile outside of the image
    y, x = [
        slice(max(0, s.start), min(upper_bound, s.stop))
        for s, upper_bound in zip(slices, max_yx)
    ]
    # find extent of padding needed in x and y
    padding = np.array([
        (-min(0, s.start), -min(0, upper_bound - s.stop))
        for s, upper_bound in zip(slices, max_yx)
    ])

    # get the tile including all z stacks
    tile = pixels[:, y, x]
    if padding.any():
        tile_shape = [x.stop - x.start for x in slices]
        if (padding / 0.25 > tile_shape).any():
            # fill with NaN because too much of the tile is outside of the image
            tile = np.full((pixels.shape[0], *tile_shape), np.nan)
        else:
            # pad tile with median value of the tile, ignore first (z) axis
            tile = np.pad(tile, [[0, 0]] + padding.tolist(), "median")
    return tile


def set_areas_of_interest(
    pixels: np.ndarray,
    tile_size: list[int] = None,
) -> tuple[tuple[int]]:
    """
    Find initial positions of tiles, or determine that the entire image is
    an area of interest.

    Remove tiles that are too close to the edge of the image
    so no padding is necessary.

    Parameters
    ----------
    tile_size: list[integer]
        The size of a tile
    """
    shape = pixels.shape
    # only tile if the image fits more than one non-overlaping tile
    if tile_size is not None and min(shape) // 2 > min(tile_size) // 2:
        half_tile = min(tile_size) // 2
        # max_size is the minimum of the numbers of x and y pixels
        max_size = min(shape[-2:])
        # find the tiles
        min_tile_size = min(tile_size)  # Use smaller end for trap segmentation
        tile_locs = segment_traps(pixels, min_tile_size)
        # keep only tiles that are not near an edge
        tile_locs = [
            [x, y]
            for x, y in tile_locs
            if half_tile < x < max_size - half_tile
            and half_tile < y < max_size - half_tile
        ]
        # store tiles in an instance of TileLocations
        tile_locs = TileLocations.from_tiler_init(tile_locs, tile_size)
    else:
        # one tile with its centre at the image's centre
        yx_shape = shape[-2:]
        tile_locs = (tuple(x // 2 for x in yx_shape),)
        tile_locs = TileLocations.from_tiler_init(tile_locs, max_size=yx_shape)
    return tile_locs
