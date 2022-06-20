"""Segment/segmented pipelines.
Includes splitting the image into traps/parts,
cell segmentation, nucleus segmentation."""
import warnings
from functools import lru_cache
import h5py
import numpy as np
from skimage.registration import phase_cross_correlation

from agora.abc import ParametersABC, ProcessABC
from agora.io.writer import load_attributes
from aliby.io.image import Image
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

    ###

    def padding_required(self, tp):
        """
        Check if we need to pad the trap image for this time point.
        """
        try:
            assert all(self.at_time(tp) - self.half_size >= 0)
            assert all(self.at_time(tp) + self.half_size <= self.max_size)
            # return False
        except AssertionError:
            return True
        return False

    def at_time(self, tp):
        """
        Return trap centre at time tp by applying drifts
        """
        drifts = self.parent.drifts
        return self.centre - np.sum(drifts[: tp + 1], axis=0)

    ###

    def as_tile(self, tp):
        """
        Return trap in the OMERO tile format of x, y, w, h

        Also returns the padding necessary for this tile.
        """
        x, y = self.at_time(tp)
        # tile bottom corner
        x = int(x - self.half_size)
        y = int(y - self.half_size)
        return x, y, self.size, self.size

    ###

    def as_range(self, tp):
        """
        Return trap in a range format: two slice objects that can
        be used in Arrays
        """
        x, y, w, h = self.as_tile(tp)
        return slice(x, x + w), slice(y, y + h)


###


class TrapLocations:
    """
    Stores each trap as an instance of Trap.
    Traps can be iterated.
    """

    def __init__(
        self, initial_location, tile_size, max_size=1200, drifts=None
    ):
        if drifts is None:
            drifts = []
        self.tile_size = tile_size
        self.max_size = max_size
        self.initial_location = initial_location
        self.traps = [
            Trap(centre, self, tile_size, max_size)
            for centre in initial_location
        ]
        self.drifts = drifts

    def __len__(self):
        return len(self.traps)

    def __iter__(self):
        yield from self.traps

    ###

    @property
    def shape(self):
        """
        Returns no of traps and no of drifts
        """
        return len(self.traps), len(self.drifts)

    def padding_required(self, tp):
        """
        Check if any traps need padding
        """
        return any([trap.padding_required(tp) for trap in self.traps])

    def to_dict(self, tp):
        """
        Export inital locations, tile_size, max_size, and drifts
        as a dictionary
        """
        res = dict()
        if tp == 0:
            res["trap_locations"] = self.initial_location
            res["attrs/tile_size"] = self.tile_size
            res["attrs/max_size"] = self.max_size
        res["drifts"] = np.expand_dims(self.drifts[tp], axis=0)
        return res

    ###

    @classmethod
    def from_tiler_init(cls, initial_location, tile_size, max_size=1200):
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


###


class TilerParameters(ParametersABC):
    _defaults = {"tile_size": 117, "ref_channel": "Brightfield", "ref_z": 0}


####


class Tiler(ProcessABC):
    """
    Remote Timelapse Tiler.

    Finds traps and re-registers images if there is any drifting.
    Fetches images from a server.
    """

    def __init__(
        self,
        image: Image,
        metadata,
        parameters: TilerParameters,
        trap_locs=None,
    ):
        super().__init__(parameters)
        self.image = image
        self.channels = metadata["channels"]
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

    ###

    @classmethod
    def from_image(cls, image: Image, parameters: TilerParameters):
        return cls(image.data, image.metadata, parameters)

    @classmethod
    def from_hdf5(cls, image: Image, filepath, parameters=None):
        trap_locs = TrapLocations.read_hdf5(filepath)
        metadata = load_attributes(filepath)
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

    ###

    @lru_cache(maxsize=2)
    def get_tc(self, t, c):
        full = self.image[t, c].compute(scheduler='synchronous')

        return full

    ###

    @property
    def shape(self):
        """
        Returns properties of the time-lapse experiment
            no of channels
            no of time points
            no of z stacks
            no of pixels in y direction
            no of pixles in z direction
        """
        c, t, z, y, x = self.image.shape
        return (c, t, x, y, z)

    @property
    def n_processed(self):
        if not hasattr(self, "_n_processed"):
            self._n_processed = 0
        return self._n_processed

    @n_processed.setter
    def n_processed(self, value):
        self._n_processed = value

    @property
    def n_traps(self):
        return len(self.trap_locs)

    @property
    def finished(self):
        """
        Returns True if all channels have been processed
        """
        return self.n_processed == self.image.shape[0]

    ###

    def _initialise_traps(self, tile_size):
        """
        Find initial trap positions.

        Removes all those that are too close to the edge so no padding
        is necessary.
        """
        half_tile = tile_size // 2
        # max_size is the minimal no of x or y pixels
        max_size = min(self.image.shape[-2:])
        # first time point, first channel, first z-position
        initial_image = self.image[0, self.ref_channel, self.ref_z]
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
        self.trap_locs = TrapLocations.from_tiler_init(trap_locs, tile_size)

    ###

    def find_drift(self, tp):
        """
        Find any translational drifts between two images at consecutive
        time points using cross correlation
        """
        # TODO check that the drift doesn't move any tiles out of
        # the image, remove them from list if so
        prev_tp = max(0, tp - 1)
        # cross-correlate
        drift, error, _ = phase_cross_correlation(
            self.image[prev_tp, self.ref_channel, self.ref_z],
            self.image[tp, self.ref_channel, self.ref_z],
        )
        # store drift
        if 0 < tp < len(self.trap_locs.drifts):
            self.trap_locs.drifts[tp] = drift.tolist()
        else:
            self.trap_locs.drifts.append(drift.tolist())

    ###

    def get_tp_data(self, tp, c):
        traps = []
        full = self.get_tc(tp, c)
        # if self.trap_locs.padding_required(tp):
        for trap in self.trap_locs:
            ndtrap = self.ifoob_pad(full, trap.as_range(tp))

            traps.append(ndtrap)
        return np.stack(traps)

    ###

    def get_trap_data(self, trap_id, tp, c):
        full = self.get_tc(tp, c)
        trap = self.trap_locs.traps[trap_id]
        ndtrap = self.ifoob_pad(full, trap.as_range(tp))
        return ndtrap

    ###

    def run_tp(self, tp):
        """
        Find traps if they have not yet been found.
        Determine any translational drift of the current image from the
        previous one.
        """
        # assert tp >= self.n_processed, "Time point already processed"
        # TODO check contiguity?
        if self.n_processed == 0 or not hasattr(self.trap_locs, "drifts"):
            self._initialise_traps(self.tile_size)
        if hasattr(self.trap_locs, "drifts"):
            drift_len = len(self.trap_locs.drifts)
            if self.n_processed != drift_len:
                raise Exception("Tiler:n_processed and ndrifts don't match")
                self.n_processed = drift_len
        # determine drift
        self.find_drift(tp)
        # update n_processed
        self.n_processed = tp + 1
        # return result for writer
        return self.trap_locs.to_dict(tp)

    def run(self):
        """
        Tile all time points in an experiment at once.
        """
        raise NotImplementedError()

    ###

    # The next set of functions are necessary for the extraction object
    def get_traps_timepoint(self, tp, tile_size=None, channels=None, z=None):
        # FIXME we currently ignore the tile size
        # FIXME can we ignore z(always  give)
        res = []
        for c in channels:
            val = self.get_tp_data(tp, c)[:, z]  # Only return requested z
            # positions
            # Starts at traps, z, y, x
            # Turn to Trap, C, T, X, Y, Z order
            val = val.swapaxes(1, 3).swapaxes(1, 2)
            val = np.expand_dims(val, axis=1)
            res.append(val)
        return np.stack(res, axis=1)

    ###

    def get_channel_index(self, item):
        for i, ch in enumerate(self.channels):
            if item in ch:
                return i

    ###

    def get_position_annotation(self):
        # TODO required for matlab support
        return None

    ###

    @staticmethod
    def ifoob_pad(full, slices):  # TODO Remove when inheriting TilerABC
        """
        Returns the slices padded if it is out of bounds

        Parameters:
        ----------
        full: (zstacks, max_size, max_size) ndarray
        Entire position with zstacks as first axis
        slices: tuple of two slices
        Each slice indicates an axis to index


        Returns
        Trap for given slices, padded with median if needed, or np.nan if the padding is too much
        """
        max_size = full.shape[-1]

        y, x = [slice(max(0, s.start), min(max_size, s.stop)) for s in slices]
        trap = full[:, y, x]

        padding = np.array(
            [(-min(0, s.start), -min(0, max_size - s.stop)) for s in slices]
        )
        if padding.any():
            tile_size = slices[0].stop - slices[0].start
            if (padding > tile_size / 4).any():
                trap = np.full((full.shape[0], tile_size, tile_size), np.nan)
            else:

                trap = np.pad(trap, [[0, 0]] + padding.tolist(), "median")

        return trap
