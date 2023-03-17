"""
Tiler: Divides images into smaller tiles.

The tasks of the Tiler are selecting regions of interest, or tiles, of images - with one trap per tile, correcting for the drift of the microscope stage over time, and handling errors and bridging between the image data and Aliby’s image-processing steps.

Tiler subclasses deal with either network connections or local files.

To find tiles, we use a two-step process: we analyse the bright-field image to produce the template of a trap, and we fit this template to the image to find the tiles' centres.

We use texture-based segmentation (entropy) to split the image into foreground -- cells and traps -- and background, which we then identify with an Otsu filter. Two methods are used to produce a template trap from these regions: pick the trap with the smallest minor axis length and average over all validated traps.

A peak-identifying algorithm recovers the x and y-axis location of traps in the original image, and we choose the approach to template that identifies the most tiles.

The experiment is stored as an array with a standard indexing order of (Time, Channels, Z-stack, X, Y).
"""
import logging
import re
import typing as t
import warnings
from functools import lru_cache
from pathlib import Path

import dask.array as da
import h5py
import numpy as np
from skimage.registration import phase_cross_correlation

from agora.abc import ParametersABC, StepABC
from agora.io.writer import BridgeH5
from aliby.io.image import ImageDummy
from aliby.tile.traps import segment_traps


class Tile:
    """
    Store a tile's location and size.

    Checks to see if the tile should be padded.
    Can export the tile either in OMERO or numpy formats.
    """

    def __init__(self, centre, parent, size, max_size):
        self.centre = centre
        self.parent = parent  # used to access drifts
        self.size = size
        self.half_size = size // 2
        self.max_size = max_size

    def at_time(self, tp: int) -> t.List[int]:
        """
        Return tile's centre by applying drifts.

        Parameters
        ----------
        tp: integer
            Index for the time point of interest.
        """
        drifts = self.parent.drifts
        tile_centre = self.centre - np.sum(drifts[: tp + 1], axis=0)
        return list(tile_centre.astype(int))

    def as_tile(self, tp: int):
        """
        Return tile in the OMERO tile format of x, y, w, h.

        Here x, y are at the bottom left corner of the tile
        and w and h are the tile width and height.

        Parameters
        ----------
        tp: integer
            Index for the time point of interest.

        Returns
        -------
        x: int
            x-coordinate of bottom left corner of tile
        y: int
            y-coordinate of bottom left corner of tile
        w: int
            Width of tile
        h: int
            Height of tile
        """
        x, y = self.at_time(tp)
        # tile bottom corner
        x = int(x - self.half_size)
        y = int(y - self.half_size)
        return x, y, self.size, self.size

    def as_range(self, tp: int):
        """
        Return tile in a range format: two slice objects that can
        be used in arrays.

        Parameters
        ----------
        tp: integer
            Index for a time point

        Returns
        -------
        A slice of x coordinates from left to right
        A slice of y coordinates from top to bottom
        """
        x, y, w, h = self.as_tile(tp)
        return slice(x, x + w), slice(y, y + h)


class TileLocations:
    """Store each tile as an instance of Tile."""

    def __init__(
        self,
        initial_location: np.array,
        tile_size: int = None,
        max_size: int = 1200,
        drifts: np.array = None,
    ):
        if drifts is None:
            drifts = []
        self.tile_size = tile_size
        self.max_size = max_size
        self.initial_location = initial_location
        self.tiles = [
            Tile(centre, self, tile_size or max_size, max_size)
            for centre in initial_location
        ]
        self.drifts = drifts

    def __len__(self):
        return len(self.tiles)

    def __iter__(self):
        yield from self.tiles

    @property
    def shape(self):
        """Return numbers of tiles and drifts."""
        return len(self.tiles), len(self.drifts)

    def to_dict(self, tp: int):
        """
        Export initial locations, tile_size, max_size, and drifts
        as a dictionary.

        Parameters
        ----------
        tp: integer
            An index for a time point
        """
        res = dict()
        if tp == 0:
            res["trap_locations"] = self.initial_location
            res["attrs/tile_size"] = self.tile_size
            res["attrs/max_size"] = self.max_size
        res["drifts"] = np.expand_dims(self.drifts[tp], axis=0)
        return res

    def at_time(self, tp: int) -> np.ndarray:
        """Return an array of tile centres (x- and y-coords)."""
        return np.array([tile.at_time(tp) for tile in self.tiles])

    @classmethod
    def from_tiler_init(
        cls, initial_location, tile_size: int = None, max_size: int = 1200
    ):
        """Instantiate from a Tiler."""
        return cls(initial_location, tile_size, max_size, drifts=[])

    @classmethod
    def read_hdf5(cls, file):
        """Instantiate from a h5 file."""
        with h5py.File(file, "r") as hfile:
            tile_info = hfile["trap_info"]
            initial_locations = tile_info["trap_locations"][()]
            drifts = tile_info["drifts"][()].tolist()
            max_size = tile_info.attrs["max_size"]
            tile_size = tile_info.attrs["tile_size"]
        tile_loc_cls = cls(initial_locations, tile_size, max_size=max_size)
        tile_loc_cls.drifts = drifts
        return tile_loc_cls


class TilerParameters(ParametersABC):
    """
    tile_size: int
    ref_channel: str or int
    ref_z: int
    backup_ref_channel int or None, if int indicates the index for reference channel. Used when image does not include metadata, ref_channel is a string and channel names are included in parsed logfiles.

    """

    _defaults = {
        "tile_size": 117,
        "ref_channel": "Brightfield",
        "ref_z": 0,
        "backup_ref_channel": None,
    }


class Tiler(StepABC):
    """
    Divide images into smaller tiles for faster processing.

    Finds tiles and re-registers images if they drift.
    Fetch images from an OMERO server if necessary.

    Uses an Image instance, which lazily provides the data on pixels,
    and, as an independent argument, metadata.
    """

    def __init__(
        self,
        image: da.core.Array,
        metadata: dict,
        parameters: TilerParameters,
        tile_locs=None,
    ):
        """
        Initialise.

        Parameters
        ----------
        image: an instance of Image
        metadata: dictionary
        parameters: an instance of TilerParameters
        tile_locs: (optional)
        """
        super().__init__(parameters)
        self.image = image
        self._metadata = metadata
        self.channels = metadata.get(
            "channels",
            list(range(metadata.get("size_c", 0))),
        )

        self.ref_channel = self.get_channel_index(parameters.ref_channel)
        if self.ref_channel is None:
            self.ref_channel = self.backup_ref_channel

        self.ref_channel = self.get_channel_index(parameters.ref_channel)
        self.tile_locs = tile_locs
        try:
            self.z_perchannel = {
                ch: zsect
                for ch, zsect in zip(self.channels, metadata["zsections"])
            }
        except Exception as e:
            self._log(f"No z_perchannel data: {e}")
        self.tile_size = self.tile_size or min(self.image.shape[-2:])

    @classmethod
    def dummy(cls, parameters: dict):
        """
        Instantiate dummy Tiler from dummy image.

        If image.dimorder exists dimensions are saved in that order.
        Otherwise default to "tczyx".

        Parameters
        ----------
        parameters: dict
            An instance of TilerParameters converted to a dict.
        """
        imgdmy_obj = ImageDummy(parameters)
        dummy_image = imgdmy_obj.get_data_lazy()
        # default to "tczyx" if image.dimorder is None
        dummy_omero_metadata = {
            f"size_{dim}": dim_size
            for dim, dim_size in zip(
                imgdmy_obj.dimorder or "tczyx", dummy_image.shape
            )
        }
        dummy_omero_metadata.update(
            {
                "channels": [
                    parameters["ref_channel"],
                    *(["nil"] * (dummy_omero_metadata["size_c"] - 1)),
                ],
                "name": "",
            }
        )
        return cls(
            imgdmy_obj.data,
            dummy_omero_metadata,
            TilerParameters.from_dict(parameters),
        )

    @classmethod
    def from_image(cls, image, parameters: TilerParameters):
        """
        Instantiate from an Image instance.

        Parameters
        ----------
        image: an instance of Image
        parameters: an instance of TilerPameters
        """
        return cls(image.data, image.metadata, parameters)

    @classmethod
    def from_h5(
        cls,
        image,
        filepath: t.Union[str, Path],
        parameters: t.Optional[TilerParameters] = None,
    ):
        """
        Instantiate from h5 files.

        Parameters
        ----------
        image: an instance of Image
        filepath: Path instance
            Path to a directory of h5 files
        parameters: an instance of TileParameters (optional)
        """
        tile_locs = TileLocations.read_hdf5(filepath)
        metadata = BridgeH5(filepath).meta_h5
        metadata["channels"] = image.metadata["channels"]
        if parameters is None:
            parameters = TilerParameters.default()
        tiler = cls(
            image.data,
            metadata,
            parameters,
            tile_locs=tile_locs,
        )
        if hasattr(tile_locs, "drifts"):
            tiler.n_processed = len(tile_locs.drifts)
        return tiler

    @lru_cache(maxsize=2)
    def get_tc(self, t: int, c: int) -> np.ndarray:
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
        t: integer
            An index for a time point
        c: integer
            An index for a channel

        Returns
        -------
        full: an array of images
        """
        full = self.image[t, c]
        if hasattr(full, "compute"):  # If using dask fetch images here
            full = full.compute(scheduler="synchronous")

        return full

    @property
    def shape(self):
        """
        Return properties of the time-lapse as shown by self.image.shape
        """
        return self.image.shape

    @property
    def n_processed(self):
        """Return the number of processed images."""
        if not hasattr(self, "_n_processed"):
            self._n_processed = 0
        return self._n_processed

    @n_processed.setter
    def n_processed(self, value):
        self._n_processed = value

    @property
    def n_tiles(self):
        """Return number of tiles."""
        return len(self.tile_locs)

    def initialise_tiles(self, tile_size: int = None):
        """
        Find initial positions of tiles.

        Remove tiles that are too close to the edge of the image
        so no padding is necessary.

        Parameters
        ----------
        tile_size: integer
            The size of a tile.
        """
        initial_image = self.image[0, self.ref_channel, self.ref_z]
        if tile_size:
            half_tile = tile_size // 2
            # max_size is the minimal number of x or y pixels
            max_size = min(self.image.shape[-2:])
            # first time point, reference channel, reference z-position
            # find the tiles
            tile_locs = segment_traps(initial_image, tile_size)
            # keep only tiles that are not near an edge
            tile_locs = [
                [x, y]
                for x, y in tile_locs
                if half_tile < x < max_size - half_tile
                and half_tile < y < max_size - half_tile
            ]
            # store tiles in an instance of TileLocations
            self.tile_locs = TileLocations.from_tiler_init(
                tile_locs, tile_size
            )
        else:
            yx_shape = self.image.shape[-2:]
            tile_locs = [[x // 2 for x in yx_shape]]
            self.tile_locs = TileLocations.from_tiler_init(
                tile_locs, max_size=min(yx_shape)
            )

    def find_drift(self, tp: int):
        """
        Find any translational drift between two images at consecutive
        time points using cross correlation.

        Arguments
        ---------
        tp: integer
            Index for a time point.
        """
        prev_tp = max(0, tp - 1)
        # cross-correlate
        drift, _, _ = phase_cross_correlation(
            self.image[prev_tp, self.ref_channel, self.ref_z],
            self.image[tp, self.ref_channel, self.ref_z],
        )
        # store drift
        if 0 < tp < len(self.tile_locs.drifts):
            self.tile_locs.drifts[tp] = drift.tolist()
        else:
            self.tile_locs.drifts.append(drift.tolist())

    def get_tp_data(self, tp, c) -> np.ndarray:
        """
        Returns all tiles corrected for drift.

        Parameters
        ----------
        tp: integer
            An index for a time point
        c: integer
            An index for a channel

        Returns
        ----------
        Numpy ndarray of tiles with shape (tile, z, y, x)
        """
        tiles = []
        # get OMERO image
        full = self.get_tc(tp, c)
        for tile in self.tile_locs:
            # pad tile if necessary
            ndtile = self.ifoob_pad(full, tile.as_range(tp))
            tiles.append(ndtile)
        return np.stack(tiles)

    def get_tile_data(self, tile_id: int, tp: int, c: int):
        """
        Return a particular tile corrected for drift and padding.

        Parameters
        ----------
        tile_id: integer
            Number of tile.
        tp: integer
            Index of time points.
        c: integer
            Index of channel.

        Returns
        -------
        ndtile: array
            An array of (x, y) arrays, one for each z stack
        """
        full = self.get_tc(tp, c)
        tile = self.tile_locs.tiles[tile_id]
        ndtile = self.ifoob_pad(full, tile.as_range(tp))
        return ndtile

    def _run_tp(self, tp: int):
        """
        Find tiles if they have not yet been found.

        Determine any translational drift of the current image from the
        previous one.

        Arguments
        ---------
        tp: integer
            The time point to tile.
        """
        # assert tp >= self.n_processed, "Time point already processed"
        # TODO check contiguity?
        if self.n_processed == 0 or not hasattr(self.tile_locs, "drifts"):
            self.initialise_tiles(self.tile_size)
        if hasattr(self.tile_locs, "drifts"):
            drift_len = len(self.tile_locs.drifts)
            if self.n_processed != drift_len:
                warnings.warn("Tiler:n_processed and ndrifts don't match")
                self.n_processed = drift_len
        # determine drift
        self.find_drift(tp)
        # update n_processed
        self.n_processed = tp + 1
        # return result for writer
        return self.tile_locs.to_dict(tp)

    def run(self, time_dim=None):
        """
        Tile all time points in an experiment at once.
        """
        if time_dim is None:
            time_dim = 0
        for frame in range(self.image.shape[time_dim]):
            self.run_tp(frame)
        return None

    def get_traps_timepoint(self, *args, **kwargs):
        self._log(
            "get_traps_timepoint is deprecated; get_tiles_timepoint instead."
        )
        return self.get_tiles_timepoint(*args, **kwargs)

    # The next set of functions are necessary for the extraction object
    def get_tiles_timepoint(
        self, tp: int, tile_shape=None, channels=None, z: int = 0
    ) -> np.ndarray:
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
            Data arranged as (tiles, channels, time points, X, Y, Z)
        """
        # FIXME add support for sub-tiling a tile
        # FIXME can we ignore z
        if channels is None:
            channels = [0]
        elif isinstance(channels, str):
            channels = [channels]
        # get the data
        res = []
        for c in channels:
            # only return requested z
            val = self.get_tp_data(tp, c)[:, z]
            # starts with the order: tiles, z, y, x
            # returns the order: tiles, C, T, Z, X, Y
            val = np.expand_dims(val, axis=1)
            res.append(val)
        if tile_shape is not None:
            if isinstance(tile_shape, int):
                tile_shape = (tile_shape, tile_shape)
            assert np.all(
                [
                    (tile_size - ax) > -1
                    for tile_size, ax in zip(tile_shape, res[0].shape[-3:-2])
                ]
            )
        return np.stack(res, axis=1)

    @property
    def ref_channel_index(self):
        """Return index of reference channel."""
        return self.get_channel_index(self.parameters.ref_channel)

    def get_channel_index(self, channel: str or int) -> int or None:
        """
        Find index for channel using regex. Returns the first matched string.
        If self.channels is integers (no image metadata) it returns None.
        If channel is integer

        Parameters
        ----------
        channel: string or int
            The channel or index to be used.
        """

        if all(map(lambda x: isinstance(x, int), self.channels)):
            channel = channel if isinstance(channel, int) else None

        if isinstance(channel, str):
            channel = find_channel_index(self.channels, channel)
        return channel

    @staticmethod
    def ifoob_pad(full, slices):
        """
        Return the slices padded if out of bounds.

        Parameters
        ----------
        full: array
            Slice of OMERO image (zstacks, x, y) - the entire position
            with zstacks as first axis
        slices: tuple of two slices
            Delineates indices for the x- and y- ranges of the tile.

        Returns
        -------
        tile: array
            A tile with all z stacks for the given slices.
            If some padding is needed, the median of the image is used.
            If much padding is needed, a tile of NaN is returned.
        """
        # number of pixels in the y direction
        max_size = full.shape[-1]
        # ignore parts of the tile outside of the image
        y, x = [slice(max(0, s.start), min(max_size, s.stop)) for s in slices]
        # get the tile including all z stacks
        tile = full[:, y, x]
        # find extent of padding needed in x and y
        padding = np.array(
            [(-min(0, s.start), -min(0, max_size - s.stop)) for s in slices]
        )
        if padding.any():
            tile_size = slices[0].stop - slices[0].start
            if (padding > tile_size / 4).any():
                # too much of the tile is outside of the image
                # fill with NaN
                tile = np.full((full.shape[0], tile_size, tile_size), np.nan)
            else:
                # pad tile with median value of the tile
                tile = np.pad(tile, [[0, 0]] + padding.tolist(), "median")
        return tile


# FIXME: Refactor to support both channel or index
# self._log below is not defined
def find_channel_index(image_channels: t.List[str], channel: str):
    """
    Access
    """
    for i, ch in enumerate(image_channels):
        found = re.match(channel, ch, re.IGNORECASE)
        if found:
            if len(found.string) - (found.endpos - found.start()):
                logging.getLogger("aliby").log(
                    logging.WARNING,
                    f"Channel {channel} matched {ch} using regex",
                )
            return i


def find_channel_name(image_channels: t.List[str], channel: str):
    """
    Find the name of the channel using regex.

    Parameters
    ----------
    image_channels: list of str
        Channels.
    channel: str
        A regular expression.
    """
    index = find_channel_index(image_channels, channel)
    if index is not None:
        return image_channels[index]
