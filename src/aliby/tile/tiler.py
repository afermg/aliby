"""
Tiler: Divides images into smaller tiles.

The tasks of the Tiler are selecting regions of interest, or tiles, of
images - with one trap per tile, correcting for the drift of the microscope
stage over time, and handling errors and bridging between the image data
and Aliby’s image-processing steps.

Tiler subclasses deal with either network connections or local files.

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

The experiment is stored as an array with a standard indexing order of
(Time, Channels, Z-stack, X, Y).
"""

import logging
import re
import typing as t
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
from agora.abc import ParametersABC, StepABC
from agora.io.bridge import BridgeH5
from aliby.global_settings import global_settings
from aliby.tile.tiles import TileLocations
from tiler.crop import cut_tile
from tiler.detect import keep_away_from_edges, segment_traps, whole_image
from tiler.drift import drift_at

if t.TYPE_CHECKING:
    # omero is optional and needed here only to annotate
    from omero.gateway import ImageWrapper


class TilerParameters(ParametersABC):
    """
    Define default values for tile size and the reference channels.

    initial_processing_tp is the first image to process. Time points are
    always the images' own indices, so processing from image 3 writes time
    point 3 first, and nothing before it.
    """

    _defaults = {
        "tile_size": global_settings.imaging_specifications["tile_size"],
        "ref_channel": "Brightfield",
        "ref_z": global_settings.imaging_specifications["ref_z"],
        "position_name": None,
        "magnification": global_settings.imaging_specifications[
            "magnification"
        ],
        "initial_processing_tp": 0,
        # register each image to the first ("first") or to the one before
        # ("previous", aliby's method until 2026)
        "drift_reference": "first",
        # warn when the two registrations disagree by more pixels than this
        "drift_check_px": 3,
    }

    def __init__(self, **kwargs):
        """Refuse initial_tp, whose meaning has changed."""
        if "initial_tp" in kwargs:
            raise ValueError(
                "initial_tp is now initial_processing_tp and no longer "
                "renumbers time points: processing from image k writes time "
                "point k, not time point 0."
            )
        super().__init__(**kwargs)


class Tiler(StepABC):
    """
    Divide images into smaller tiles for faster processing.

    Find tiles and re-register images if they drift.
    Fetch images from an OMERO server if necessary.

    Uses an Image instance, which lazily provides the pixel data,
    and, as an independent argument, image_metadata.
    """

    def __init__(
        self,
        image: da.core.Array,
        image_metadata: t.Dict,
        parameters: TilerParameters,
        tile_locations=None,
        microscopy_metadata: t.Dict = None,
    ):
        """
        Initialise.

        Parameters
        ----------
        image: an instance of Image
        image_metadata: dictionary
        parameters: an instance of TilerParameters
        tile_locs: (optional)
        microscopy_metadata: optional
        """
        super().__init__(parameters)
        self.image = image
        self.position_name = parameters.to_dict()["position_name"]
        if "channels" in image_metadata and isinstance(image, da.Array):
            # information for a particular image
            self.channels = image_metadata["channels"]
        elif (
            microscopy_metadata is not None
            and "channels_by_position" in microscopy_metadata["full"]
        ):
            # likely zarr array
            self.channels = microscopy_metadata["full"][
                "channels_by_position"
            ][self.position_name]
        elif "channels" in image_metadata:
            # an image whose data is not a dask array, such as a zarr array
            self.channels = image_metadata["channels"]
        else:
            self.channels = list(range(image_metadata.get("size_c", 0)))
        # get spatial location of position
        if (
            microscopy_metadata is not None
            and "spatial_locations" in microscopy_metadata["full"]
        ):
            self.spatial_location = microscopy_metadata["full"][
                "spatial_locations"
            ][self.position_name]
        else:
            self.spatial_location = None
        # get reference channel - used for segmentation
        self.ref_channel_index = self.channels.index(parameters.ref_channel)
        self.tile_locs = tile_locations
        # adjust for non-standard magnification
        if (
            self.tile_size
            != global_settings.imaging_specifications["tile_size"]
        ):
            print(
                "Warning: tile_size has been changed."
                "\nConsider changing magnification instead."
            )
        else:
            # adjust tile_size for any magnification differing from default
            self.tile_size = int(
                self.tile_size
                * self.magnification
                / global_settings.imaging_specifications["magnification"]
            )

    @classmethod
    def from_image(
        cls,
        image: "ImageWrapper",
        parameters: TilerParameters,
        microscopy_metadata: t.Dict = None,
    ):
        """
        Instantiate from an Image instance.

        Parameters
        ----------
        image: an instance of Image
        parameters: an instance of TilerPameters
        """
        if microscopy_metadata is not None:
            meta = microscopy_metadata["minimal"] | image.metadata
        else:
            meta = image.metadata
        return cls(
            image.data,
            image_metadata=meta,
            parameters=parameters,
            microscopy_metadata=microscopy_metadata,
        )

    @classmethod
    def from_h5(
        cls,
        image: "ImageWrapper",
        filepath: t.Union[str, Path],
        parameters: t.Optional[TilerParameters] = None,
    ):
        """
        Instantiate from an h5 file.

        Parameters
        ----------
        image: an instance of Image
        filepath: Path instance
            Path to an h5 file.
        parameters: an instance of TileParameters (optional)
        """
        tile_locs = TileLocations.from_h5(filepath)
        image_metadata = BridgeH5(filepath).meta_h5
        if "channels" in image.metadata:
            # update h5 metadata using the image
            image_metadata["channels"] = image.metadata["channels"]
        if parameters is None:
            parameters = TilerParameters.default()
        tiler = cls(
            image.data,
            image_metadata,
            parameters,
            tile_locations=tile_locs,
        )
        if hasattr(tile_locs, "drifts"):
            tiler.no_processed = len(tile_locs.drifts)
        return tiler

    @property
    def shape(self):
        """
        Return the shape of the image array.

        The image array is arranged as number of images, number of channels,
        number of z sections, and size of the image in y and x.
        """
        return self.image.shape

    @property
    def first_processed_tp(self) -> int:
        """Return the first time point to process."""
        return int(getattr(self, "initial_processing_tp", 0))

    @property
    def no_processed(self):
        """Return the number of processed images."""
        if not hasattr(self, "_no_processed"):
            self._no_processed = 0
        return self._no_processed

    @no_processed.setter
    def no_processed(self, value):
        self._no_processed = value

    @property
    def no_tiles(self):
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
        initial_image = self.image[
            self.first_processed_tp, self.ref_channel_index, self.ref_z
        ]
        if tile_size:
            # find the tiles, as (row, column), keeping those clear of the
            # edges with a margin for drift
            tile_locs = keep_away_from_edges(
                segment_traps(initial_image, tile_size),
                self.image.shape[-2:],
                tile_size,
            )
            # store tiles in an instance of TileLocations
            self.tile_locs = TileLocations.from_tiler(
                tile_locs, tile_size, image_size_yx=self.image.shape[-2:]
            )
        else:
            # one tile with its centre at the image's centre
            tile_locs, max_size = whole_image(self.image.shape[-2:])
            self.tile_locs = TileLocations.from_tiler(
                tile_locs, max_size=max_size,
                image_size_yx=self.image.shape[-2:],
            )

    def find_drift(self, tp: int):
        """
        Find the translational drift of an image from the one before.

        Use tiler.drift.drift_at: by default, register the image to the
        first image processed, where the tiles were found, and store the
        change in that displacement since the previous time point.
        Registering each image to the one before instead
        (drift_reference="previous"), as aliby did until 2026, sums the
        error of every registration.

        As a check, also register the image to the one before and log a
        warning if the two drifts disagree by more than drift_check_px,
        recording the time point in drift_disagreements.

        Arguments
        ---------
        tp: integer
            Index for a time point.
        """
        _, disagrees = drift_at(
            lambda frame: self.image[
                frame, self.ref_channel_index, self.ref_z
            ],
            tp,
            self.tile_locs.drifts,
            first_tp=self.first_processed_tp,
            reference=getattr(self, "drift_reference", "first"),
            check_px=getattr(self, "drift_check_px", 3),
            log=logging.getLogger("aliby"),
        )
        if disagrees:
            self.drift_disagreements.append(tp)

    @property
    def drift_disagreements(self) -> list[int]:
        """Return time points whose drift registrations disagreed."""
        if not hasattr(self, "_drift_disagreements"):
            self._drift_disagreements = []
        return self._drift_disagreements

    def load_image(
        self, tp: int, c: int, lazy: bool = True
    ) -> t.Union[np.ndarray, da.Array]:
        """
        Load image for one time point and channel using dask.

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
        lazy: bool, optional
            If True, return dask array without computing. Default is False.

        Returns
        -------
        image_all_z: an array of z slices for the entire image
            Returns np.ndarray if lazy=False, da.Array if lazy=True
        """
        image_all_z = self.image[tp, c]
        if not lazy and hasattr(image_all_z, "compute"):
            # if using dask fetch images
            image_all_z = image_all_z.compute(scheduler="synchronous")
        return image_all_z

    def get_lazy_tile_view(self, tile_id: int, tp: int, c: int) -> da.Array:
        """
        Return a lazy dask array view of a single tile.

        Do not load full image. Ensures consistent tile size by padding
        if necessary.

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
        tile_view: dask array
            A lazy view of the tile region with shape (z, y, x)
            Guaranteed to have consistent tile_size dimensions.
        """
        tile = self.tile_locs.tiles[tile_id]
        tile_range = tile.as_range(tp)
        image_all_z = self.image[tp, c]
        return self.get_tile_and_pad(image_all_z, tile_range, self.tile_size)

    def get_tile_data(
        self, tile_id: int, tp: int, c: int, lazy: bool = True
    ) -> da.Array:
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
        lazy: bool
            If True, return a dask array (default True for memory
            efficiency).

        Returns
        -------
        ndtile: dask array
            An array of (z, y, x) arrays, one for each z stack
        """
        # use lazy loading by default to avoid memory issues
        image_all_z = self.image[tp, c]
        tile = self.tile_locs.tiles[tile_id]
        ndtile = self.get_tile_and_pad(
            image_all_z, tile.as_range(tp), self.tile_size
        )
        # only compute if explicitly requested
        if not lazy:
            ndtile = ndtile.compute(scheduler="synchronous")
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
        if self.no_processed == 0 or not hasattr(self.tile_locs, "drifts"):
            self.initialise_tiles(self.tile_size)
        if hasattr(self.tile_locs, "drifts"):
            drift_len = len(self.tile_locs.drifts)
            if self.no_processed != drift_len:
                warnings.warn(
                    "Tiler: the number of processed tiles and the number of "
                    "drifts calculated do not match."
                )
                self.no_processed = drift_len
        # determine drift for this time point and update tile_locs.drifts
        self.find_drift(tp)
        # update no_processed
        self.no_processed = tp + 1
        # return result for writer
        return self.tile_locs.to_dict(tp, first_tp=self.first_processed_tp)

    def run(self, time_dim=None):
        """Tile all time points in an experiment at once."""
        if time_dim is None:
            time_dim = 0
        for tp in range(self.first_processed_tp, self.image.shape[time_dim]):
            self.run_tp(tp)
        return None

    def get_tp_data_for_one_channel(
        self, tp, c, lazy: bool = True
    ) -> t.Union[np.ndarray, da.Array]:
        """
        Return all tiles corrected for drift.

        Use memory-efficient lazy loading.

        Parameters
        ----------
        tp: integer
            An index for a time point
        c: integer
            An index for a channel
        lazy: bool, optional
            If True, return dask array without computing.
            Default is True for memory efficiency.

        Returns
        ----------
        Array of tiles with shape (no tiles, z-sections, y, x)
        Returns np.ndarray if lazy=False, da.Array if lazy=True
        """
        tiles = []
        # use lazy tile views instead of loading full image
        image_all_z = self.image[tp, c]
        # decompose into tiles using lazy views
        for tile in self.tile_locs:
            # pad tile if necessary - this remains lazy until computed
            ndtile = Tiler.get_tile_and_pad(
                image_all_z, tile.as_range(tp), tile.size
            )
            tiles.append(ndtile)
        result = da.stack(tiles)
        if not lazy:
            result = result.compute(scheduler="synchronous")
        return result

    def get_tiles_lazy(
        self, tp: int, c: int, tile_ids: t.List[int] = None
    ) -> da.Array:
        """
        Get multiple tiles with optimised chunking for memory efficiency.

        Create lazy arrays for each tile and stack with appropriate
        chunking to minimise memory usage.

        Parameters
        ----------
        tp: integer
            Time point index
        c: integer
            Channel index
        tile_ids: list of integers, optional
            List of tile indices to retrieve.
            If None, all tiles are returned.

        Returns
        -------
        tiles_array: dask array
            Lazy array with shape (n_tiles, z, y, x) and optimised chunking
        """
        if tile_ids is None:
            tile_ids = list(range(len(self.tile_locs)))
        # create lazy arrays for each tile using efficient tile views
        tiles = [
            self.get_lazy_tile_view(tile_id, tp, c) for tile_id in tile_ids
        ]
        # stack into a single dask array with appropriate chunking
        # one tile per chunk in the first dimension for efficient processing
        result = da.stack(tiles)
        # rechunk to optimize for tile-wise access
        return result.rechunk((1, -1, -1, -1))

    def get_tiles_timepoint(
        self,
        tp: int,
        channels: str or list[str] = None,
        z: int | list[int] = 0,
        lazy: bool = False,
    ) -> t.Union[np.ndarray, da.Array]:
        """
        Get all tiles as an array for a set of channels and a z-stack.

        Used by extractor.

        Parameters
        ---------
        tp: int
            Index of time point.
        channels: string or list of strings
            Names of channels of interest.
        z: int or list of int
            Indices of z-channel of interest.
        lazy: bool, optional
            If True, return dask array without computing. Default is False.

        Returns
        -------
        final: array
            Data arranged as (tiles, channels, 1, Z, Y, X), rows before
            columns. The axis of length one is a relic: callers index it
            away with [:, 0, 0], and tests/test_tile_golden.py pins the
            shape. tiler's own read_trap gives (time points, channels, Z,
            Y, X).
            Returns np.ndarray if lazy=False, da.Array if lazy=True
        """
        if channels is None:
            channels = ["Brightfield"]
        elif isinstance(channels, str):
            channels = [channels]
        # convert to indices
        channels = [
            (
                self.channels.index(channel)
                if isinstance(channel, str)
                else channel
            )
            for channel in channels
        ]
        # get the data as a list of length of the number of channels
        res = []
        for c in channels:
            # first dimension is number of traps
            tiles = self.get_tp_data_for_one_channel(tp, c, lazy=True)[:, z]
            # add back channel axis
            tiles = da.expand_dims(tiles, axis=1)
            res.append(tiles)
        # stack over channels
        final = da.stack(res, axis=1)
        if not lazy:
            final = final.compute(scheduler="synchronous")
        return final

    def get_tiles_timepoint_lazy(
        self,
        tp: int,
        channels: t.Union[str, t.List[str], t.List[int]] = None,
        z: t.Union[int, t.List[int]] = 0,
    ) -> da.Array:
        """
        Use lazy loading to implement get_tiles_timepoint.

        Use optimized dask arrays to avoid loading the entire
        dataset into memory.

        Parameters
        ----------
        tp: int
            Index of time point.
        channels: string, list of strings, or list of ints
            Names or indices of channels of interest.
        z: int or list of int
            Indices of z-stack of interest.

        Returns
        -------
        final: dask array
            Lazy array arranged as (tiles, channels, z, y, x)
        """
        if channels is None:
            channels = ["Brightfield"]
        elif isinstance(channels, str):
            channels = [channels]
        # convert to indices
        channel_indices = [
            (
                self.channels.index(channel)
                if isinstance(channel, str)
                else channel
            )
            for channel in channels
        ]
        # get the data as a list of length of the number of channels
        res = []
        for c in channel_indices:
            # Use lazy tile loading to avoid memory issues
            tiles = self.get_tiles_lazy(tp, c)[:, z]
            # add back channel axis
            tiles = da.expand_dims(tiles, axis=1)
            res.append(tiles)
        # stack over channels - remains lazy
        final = da.stack(res, axis=1)
        return final

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

    @staticmethod
    def get_tile_and_pad(image_array, slices, tile_size=None):
        """
        Pad slices if out of bounds.

        Parameters
        ----------
        image_array: array
            Slice of image (zstacks, y, x) - the entire position
            with zstacks as first axis
        slices: tuple of two slices
            Delineates indices for the y- and x- ranges of the tile,
            rows first, as everything here is (row, column).

        Returns
        -------
        tile: dask array
            A tile with all z stacks for the given slices.
            If some padding is needed, edge values are replicated.
            If much padding is needed, a tile of NaN is returned.
        """
        # tiler.crop.cut_tile cuts and pads; a dask image keeps tiles lazy
        return cut_tile(da.asarray(image_array), slices, tile_size)


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
