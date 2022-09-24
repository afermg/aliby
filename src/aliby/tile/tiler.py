"""
Tiler: Tiles and tracks traps.

The tasks of the Tiler are selecting regions of interest, or tiles, of an image - with one tile per trap, tracking and correcting for the drift of the microscope stage over time, and handling errors and bridging between the image data and ALIBY’s image-processing steps.

Tiler subclasses deal with either network connections or local files.

To find traps, we use a two-step process: we analyse the bright-field image to produce the template of a trap, and we fit this template to the image to find the traps' centres.

We use texture-based segmentation (entropy) to split the image into foreground -- cells and traps -- and background, which we then identify with an Otsu filter. Two methods are used to produce a template trap from these regions: pick the trap with the smallest minor axis length and average over all validated traps.

A peak-identifying algorithm recovers the x and y-axis location of traps in the original image, and we choose the templating approach that identifies the most traps

One key method is Tiler.run.

The image-processing is performed by traps/segment_traps.

The experiment is stored as an array with a standard indexing order of (Time, Channels, Z-stack, Y, X).
"""
import typing as t
import warnings
from functools import lru_cache
from pathlib import PosixPath

import dask.array as da
import h5py
import numpy as np
from agora.abc import ParametersABC, ProcessABC
from agora.io.writer import BridgeH5
from skimage.registration import phase_cross_correlation

from aliby.io.image import Image, ImageLocal
from aliby.tile.traps import segment_traps


class Trap:
    """
    Stores a trap's location and size.
    Allows checks to see if the trap should be padded.
    Can export the trap either in OMERO or numpy formats.
    """

    def __init__(self, centre, parent, size, max_size):
        self.centre = centre
        self.parent = parent  # used to access drifts
        self.size = size
        self.half_size = size // 2
        self.max_size = max_size

    def at_time(self, tp: int) -> t.List[int]:
        """
        Return trap centre at time tp by applying drifts

        Parameters
        ----------
        tp: integer
            Index for a time point

        Returns
        -------
        trap_centre:
        """
        drifts = self.parent.drifts
        trap_centre = self.centre - np.sum(drifts[: tp + 1], axis=0)
        return list(trap_centre.astype(int))

    def as_tile(self, tp):
        """
        Return trap in the OMERO tile format of x, y, w, h
        where x, y are at the bottom left corner of the tile
        and w and h are the tile width and height.

        Parameters
        ----------
        tp: integer
            Index for a time point

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

    def as_range(self, tp):
        """
        Return trap in a range format: two slice objects that can
        be used in arrays

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


class TrapLocations:
    """
    Stores each trap as an instance of Trap.
    Traps can be iterated.
    """

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
        self.traps = [
            Trap(centre, self, tile_size or max_size, max_size)
            for centre in initial_location
        ]
        self.drifts = drifts

    def __len__(self):
        return len(self.traps)

    def __iter__(self):
        yield from self.traps

    @property
    def shape(self):
        """
        Returns no of traps and no of drifts
        """
        return len(self.traps), len(self.drifts)

    def to_dict(self, tp):
        """
        Export inital locations, tile_size, max_size, and drifts
        as a dictionary

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
        # Returns ( ntraps, 2 ) ndarray with the trap centres as individual rows
        return np.array([trap.at_time(tp) for trap in self.traps])

    @classmethod
    def from_tiler_init(
        cls, initial_location, tile_size: int = None, max_size: int = 1200
    ):
        """
        Instantiate class from an instance of the Tiler class
        """
        return cls(initial_location, tile_size, max_size, drifts=[])

    @classmethod
    def read_hdf5(cls, file):
        """
        Instantiate class from a hdf5 file
        """
        with h5py.File(file, "r") as hfile:
            trap_info = hfile["trap_info"]
            initial_locations = trap_info["trap_locations"][()]
            drifts = trap_info["drifts"][()].tolist()
            max_size = trap_info.attrs["max_size"]
            tile_size = trap_info.attrs["tile_size"]
        trap_locs = cls(initial_locations, tile_size, max_size=max_size)
        trap_locs.drifts = drifts
        return trap_locs


class TilerParameters(ParametersABC):
    _defaults = {"tile_size": 117, "ref_channel": "Brightfield", "ref_z": 0}


class Tiler(ProcessABC):
    """
    Remote Timelapse Tiler.

    Finds traps and re-registers images if there is any drifting.
    Fetches images from a server.
    """

    def __init__(
        self,
        image: da.core.Array,
        metadata: dict,
        parameters: TilerParameters,
        trap_locs=None,
    ):
        """
        Initialise Tiler

        Parameters
        ----------
        image: an instance of Image
        metadata: dictionary
        parameters: an instance of TilerPameters
        trap_locs: (optional)
        """
        super().__init__(parameters)
        self.image = image
        self._metadata = metadata
        self.channels = metadata["channels"]
        assert (
            parameters.ref_channel in self.channels
        ), "Reference channel not in the available channels"
        self.ref_channel = self.get_channel_index(parameters.ref_channel)
        self.trap_locs = trap_locs
        try:
            self.z_perchannel = {
                ch: metadata["zsectioning/nsections"] if zsect else 1
                for zsect, ch in zip(
                    metadata["channels"], metadata["channels/zsect"]
                )
            }
        except Exception as e:
            print(f"Warning:Tiler: No z_perchannel data: {e}")

        self.tile_size = self.tile_size or min(self.image.shape[-2:])

    @classmethod
    def from_image(cls, image: Image, parameters: TilerParameters):
        """
        Instantiate Tiler from an Image instance

        Parameters
        ----------
        image: an instance of Image
        parameters: an instance of TilerPameters
        """
        return cls(image.data, image.metadata, parameters)

    @classmethod
    def from_h5(
        cls,
        image: t.Union[Image, ImageLocal],
        filepath: t.Union[str, PosixPath],
        parameters: TilerParameters = None,
    ):
        """
        Instantiate Tiler from hdf5 files

        Parameters
        ----------
        image: an instance of Image
        filepath: Path instance
            Path to a directory of h5 files
        parameters: an instance of TileParameters (optional)
        """
        trap_locs = TrapLocations.read_hdf5(filepath)
        metadata = BridgeH5(filepath).meta_h5
        metadata["channels"] = image.metadata["channels"]
        # metadata["zsectioning/nsections"] = image.metadata["zsectioning/nsections"]
        # metadata["channels/zsect"] = image.metadata["channels/zsect"]
        if parameters is None:
            parameters = TilerParameters.default()
        tiler = cls(
            image.data,
            metadata,
            parameters,
            trap_locs=trap_locs,
        )
        if hasattr(trap_locs, "drifts"):
            tiler.n_processed = len(trap_locs.drifts)
        return tiler

    @lru_cache(maxsize=2)
    def get_tc(self, t, c):
        """
        Load image using dask.
        Assumes the image is arranged as
            no of time points
            no of channels
            no of z stacks
            no of pixels in y direction
            no of pixels in z direction

        Parameters
        ----------
        t: integer
            An index for a time point
        c: integer
            An index for a channel

        Retruns
        -------
        full: an array of images
        """
        full = self.image[t, c].compute(scheduler="synchronous")
        return full

    @property
    def shape(self):
        """
        Returns properties of the time-lapse experiment
            no of channels
            no of time points
            no of z stacks
            no of pixels in y direction
            no of pixels in z direction
        """
        c, t, z, y, x = self.image.shape
        return (c, t, x, y, z)

    @property
    def n_processed(self):
        """
        Returns the number of images that have been processed
        """
        if not hasattr(self, "_n_processed"):
            self._n_processed = 0
        return self._n_processed

    @n_processed.setter
    def n_processed(self, value):
        self._n_processed = value

    @property
    def n_traps(self):
        """
        Returns number of traps
        """
        return len(self.trap_locs)

    def initialise_traps(self, tile_size: int = None):
        """
        Find initial trap positions if they have not been initialised.
        Removes all those that are too close to the edge so no padding
        is necessary.

        Parameters
        ----------
        tile_size: integer
            The size of a tile
        """
        initial_image = self.image[0, self.ref_channel, self.ref_z]
        if tile_size:
            half_tile = tile_size // 2
            # max_size is the minimal number of x or y pixels
            max_size = min(self.image.shape[-2:])
            # first time point, reference channel, reference z-position
            # find the traps
            trap_locs = segment_traps(initial_image, tile_size)
            # keep only traps that are not near an edge
            trap_locs = [
                [x, y]
                for x, y in trap_locs
                if half_tile < x < max_size - half_tile
                and half_tile < y < max_size - half_tile
            ]
            # store traps in an instance of TrapLocations
            self.trap_locs = TrapLocations.from_tiler_init(
                trap_locs, tile_size
            )
        else:
            yx_shape = self.image.shape[-2:]
            trap_locs = [[x // 2 for x in yx_shape]]
            self.trap_locs = TrapLocations.from_tiler_init(
                trap_locs, max_size=min(yx_shape)
            )

    def find_drift(self, tp):
        """
        Find any translational drift between two images at consecutive
        time points using cross correlation.

        Arguments
        ---------
        tp: integer
            Index for a time point
        """
        prev_tp = max(0, tp - 1)
        # cross-correlate
        drift, _, _ = phase_cross_correlation(
            self.image[prev_tp, self.ref_channel, self.ref_z],
            self.image[tp, self.ref_channel, self.ref_z],
        )
        # store drift
        if 0 < tp < len(self.trap_locs.drifts):
            self.trap_locs.drifts[tp] = drift.tolist()
        else:
            self.trap_locs.drifts.append(drift.tolist())

    def get_tp_data(self, tp, c):
        """
        Returns all traps corrected for drift.

        Parameters
        ----------
        tp: integer
            An index for a time point
        c: integer
            An index for a channel
        """
        traps = []
        # get OMERO image
        full = self.get_tc(tp, c)
        for trap in self.trap_locs:
            # pad trap if necessary
            ndtrap = self.ifoob_pad(full, trap.as_range(tp))
            traps.append(ndtrap)
        return np.stack(traps)

    def get_trap_data(self, trap_id, tp, c):
        """
        Returns a particular trap corrected for drift and padding

        Parameters
        ----------
        trap_id: integer
            Number of trap
        tp: integer
            Index of time points
        c: integer
            Index of channel

        Returns
        -------
        ndtrap: array
            An array of (x, y) arrays, one for each z stack
        """
        full = self.get_tc(tp, c)
        trap = self.trap_locs.traps[trap_id]
        ndtrap = self.ifoob_pad(full, trap.as_range(tp))
        return ndtrap

    def run_tp(self, tp):
        """
        Find traps if they have not yet been found.
        Determine any translational drift of the current image from the
        previous one.

        Arguments
        ---------
        tp: integer
            The time point to tile.
        """
        # assert tp >= self.n_processed, "Time point already processed"
        # TODO check contiguity?
        if self.n_processed == 0 or not hasattr(self.trap_locs, "drifts"):
            self.initialise_traps(self.tile_size)
        if hasattr(self.trap_locs, "drifts"):
            drift_len = len(self.trap_locs.drifts)
            if self.n_processed != drift_len:
                warnings.warn("Tiler:n_processed and ndrifts don't match")
                self.n_processed = drift_len
        # determine drift
        self.find_drift(tp)
        # update n_processed
        self.n_processed = tp + 1
        # return result for writer
        return self.trap_locs.to_dict(tp)

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
        #
        print(
            DeprecationWarning("Deprecated:Use get_tiles_timepoint instead.")
        )

        return self.get_tiles_timepoint(*args, **kwargs)

    # The next set of functions are necessary for the extraction object
    def get_tiles_timepoint(
        self, tp, tile_shape=None, channels=None, z=None
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
            Data arranged as (traps, channels, timepoints, X, Y, Z)
        """
        # FIXME add support for subtiling trap
        # FIXME can we ignore z(always  give)
        if channels is None:
            channels = [0]
        elif isinstance(channels, str):
            channels = [channels]
        # get the data
        res = []
        for c in channels:
            # only return requested z
            val = self.get_tp_data(tp, c)[:, z]
            # starts with the order: traps, z, y, x
            # returns the order: trap, C, T, X, Y, Z
            val = val.swapaxes(1, 3).swapaxes(1, 2)
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

    def get_channel_index(self, item):
        """
        Find index for channel

        Parameters
        ----------
        item: string
            The channel
        """
        for i, ch in enumerate(self.channels):
            if item in ch:
                return i

    @staticmethod
    def ifoob_pad(full, slices):
        """
        Returns the slices padded if it is out of bounds.

        Parameters
        ----------
        full: array
            Slice of OMERO image (zstacks, x, y) - the entire position
            with zstacks as first axis
        slices: tuple of two slices
            Delineates indiceds for the x- and y- ranges of the tile.

        Returns
        -------
        trap: array
            A tile with all z stacks for the given slices.
            If some padding is needed, the median of the image is used.
            If much padding is needed, a tile of NaN is returned.
        """
        # number of pixels in the y direction
        max_size = full.shape[-1]
        # ignore parts of the tile outside of the image
        y, x = [slice(max(0, s.start), min(max_size, s.stop)) for s in slices]
        # get the tile including all z stacks
        trap = full[:, y, x]
        # find extent of padding needed in x and y
        padding = np.array(
            [(-min(0, s.start), -min(0, max_size - s.stop)) for s in slices]
        )
        if padding.any():
            tile_size = slices[0].stop - slices[0].start
            if (padding > tile_size / 4).any():
                # too much of the tile is outside of the image
                # fill with NaN
                trap = np.full((full.shape[0], tile_size, tile_size), np.nan)
            else:
                # pad tile with median value of trap image
                trap = np.pad(trap, [[0, 0]] + padding.tolist(), "median")
        return trap
