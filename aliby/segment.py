"""Segment/segmented pipelines.
Includes splitting the image into traps/parts,
cell segmentation, nucleus segmentation."""
import warnings
from functools import lru_cache

import h5py
import numpy as np

from pathlib import Path, PosixPath

from skimage.registration import phase_cross_correlation

from agora.base import ParametersABC, ProcessABC
from aliby.traps import segment_traps
from aliby.timelapse import TimelapseOMERO
from aliby.io.matlab import matObject
from aliby.traps import (
    identify_trap_locations,
    get_trap_timelapse,
    get_traps_timepoint,
    centre,
    get_trap_timelapse_omero,
)
from aliby.utils import accumulate, get_store_path

from aliby.io.writer import Writer, load_attributes
from aliby.io.metadata_parser import parse_logfiles

trap_template_directory = Path(__file__).parent / "trap_templates"
# TODO do we need multiple templates, one for each setup?
trap_template = np.array([])  # np.load(trap_template_directory / "trap_prime.npy")


def get_tile_shapes(x, tile_size, max_shape):
    half_size = tile_size // 2
    xmin = int(x[0] - half_size)
    ymin = max(0, int(x[1] - half_size))
    if xmin + tile_size > max_shape[0]:
        xmin = max_shape[0] - tile_size
    if ymin + tile_size > max_shape[1]:
        ymin = max_shape[1] - tile_size
    return xmin, xmin + tile_size, ymin, ymin + tile_size


###################### Dask versions ########################
class Trap:
    def __init__(self, centre, parent, size, max_size):
        self.centre = centre
        self.parent = parent  # Used to access drifts
        self.size = size
        self.half_size = size // 2
        self.max_size = max_size

    def padding_required(self, tp):
        """Check if we need to pad the trap image for this time point."""
        try:
            assert all(self.at_time(tp) - self.half_size >= 0)
            assert all(self.at_time(tp) + self.half_size <= self.max_size)
        except AssertionError:
            return True
        return False

    def at_time(self, tp):
        """Return trap centre at time tp"""
        drifts = self.parent.drifts
        return self.centre - np.sum(drifts[:tp], axis=0)

    def as_tile(self, tp):
        """Return trap in the OMERO tile format of x, y, w, h

        Also returns the padding necessary for this tile.
        """
        x, y = self.at_time(tp)
        # tile bottom corner
        x = int(x - self.half_size)
        y = int(y - self.half_size)
        return x, y, self.size, self.size

    def as_range(self, tp):
        """Return trap in a range format, two slice objects that can be used in Arrays"""
        x, y, w, h = self.as_tile(tp)
        return slice(x, x + w), slice(y, y + h)


class TrapLocations:
    def __init__(self, initial_location, tile_size, max_size=1200, drifts=[]):
        self.tile_size = tile_size
        self.max_size = max_size
        self.initial_location = initial_location
        self.traps = [
            Trap(centre, self, tile_size, max_size) for centre in initial_location
        ]
        self.drifts = drifts

    @classmethod
    def from_source(cls, fpath: str):
        with h5py.File(fpath, "r") as f:
            # TODO read tile size from file metadata
            drifts = f["trap_info/drifts"][()]
            tlocs = cls(f["trap_info/trap_locations"][()], tile_size=96, drifts=drifts)

        return tlocs

    @property
    def shape(self):
        return len(self.traps), len(self.drifts)

    def __len__(self):
        return len(self.traps)

    def __iter__(self):
        yield from self.traps

    def padding_required(self, tp):
        return any([trap.padding_required(tp) for trap in self.traps])

    def to_dict(self, tp):
        res = dict()
        if tp == 0:
            res["trap_locations"] = self.initial_location
            res["attrs/tile_size"] = self.tile_size
            res["attrs/max_size"] = self.max_size
        res["drifts"] = np.expand_dims(self.drifts[tp], axis=0)
        # res['processed_timepoints'] = tp
        return res

    @classmethod
    def read_hdf5(cls, file):
        with h5py.File(file, "r") as hfile:
            trap_info = hfile["trap_info"]
            initial_locations = trap_info["trap_locations"][()]
            drifts = trap_info["drifts"][()]
            max_size = trap_info.attrs["max_size"]
            tile_size = trap_info.attrs["tile_size"]
        trap_locs = cls(initial_locations, tile_size, max_size=max_size)
        trap_locs.drifts = drifts
        return trap_locs


class TilerParameters(ParametersABC):
    def __init__(
        self, tile_size: int, ref_channel: str, ref_z: int, template_name: str = None
    ):
        self.tile_size = tile_size
        self.ref_channel = ref_channel
        self.ref_z = ref_z
        self.template_name = template_name

    @classmethod
    def from_template(cls, template_name: str, ref_channel: str, ref_z: int):
        return cls(template.shape[0], ref_channel, ref_z, template_path=template_name)

    @classmethod
    def default(cls):
        return cls(96, "Brightfield", 0)


class Tiler(ProcessABC):
    """A dummy TimelapseTiler object fora Dask Demo.

    Does trap finding and image registration."""

    def __init__(
        self,
        image,
        metadata,
        parameters: TilerParameters,
    ):
        super().__init__(parameters)
        self.image = image
        self.channels = metadata["channels"]
        self.ref_channel = self.get_channel_index(parameters.ref_channel)

    @classmethod
    def from_image(cls, image, parameters: TilerParameters):
        return cls(image.data, image.metadata, parameters)

    @classmethod
    def from_hdf5(cls, image, filepath, tile_size=None):
        trap_locs = TrapLocations.read_hdf5(filepath)
        metadata = load_attributes(filepath)
        metadata["channels"] = metadata["channels/channel"].tolist()
        if tile_size is None:
            tile_size = trap_locs.tile_size
        return Tiler(
            image=image,
            metadata=metadata,
            template=None,
            tile_size=tile_size,
            trap_locs=trap_locs,
        )

    @lru_cache(maxsize=2)
    def get_tc(self, t, c):
        # Get image
        full = self.image[t, c].compute()  # FORCE THE CACHE
        return full

    @property
    def shape(self):
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
        return self.n_processed == self.image.shape[0]

    def _initialise_traps(self, tile_size):
        """Find initial trap positions.

        Removes all those that are too close to the edge so no padding is necessary.
        """
        half_tile = tile_size // 2
        max_size = min(self.image.shape[-2:])
        initial_image = self.image[
            0, self.ref_channel, self.ref_z
        ]  # First time point, first channel, first z-position
        trap_locs = segment_traps(initial_image, tile_size)
        trap_locs = [
            [x, y]
            for x, y in trap_locs
            if half_tile < x < max_size - half_tile
            and half_tile < y < max_size - half_tile
        ]
        self.trap_locs = TrapLocations(trap_locs, tile_size)

    def find_drift(self, tp):
        # TODO check that the drift doesn't move any tiles out of the image, remove them from list if so
        prev_tp = max(0, tp - 1)
        drift, error, _ = phase_cross_correlation(
            self.image[prev_tp, self.ref_channel, self.ref_z],
            self.image[tp, self.ref_channel, self.ref_z],
        )
        self.trap_locs.drifts.append(drift)

    def get_tp_data(self, tp, c):
        traps = []
        full = self.get_tc(tp, c)
        # if self.trap_locs.padding_required(tp):
        for trap in self.trap_locs:
            ndtrap = self.ifoob_pad(full, trap.as_range(tp))

            traps.append(ndtrap)
        return np.stack(traps)

    def get_trap_data(self, trap_id, tp, c):
        full = self.get_tc(tp, c)
        trap = self.trap_locs.traps[trap_id]
        ndtrap = self.ifoob_pad(full, trap.as_range(tp))

        return ndtrap

    @staticmethod
    def ifoob_pad(full, slices):
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

    def run_tp(self, tp):
        assert tp >= self.n_processed, "Time point already processed"
        # TODO check contiguity?
        if self.n_processed == 0:
            self._initialise_traps(self.tile_size)
        self.find_drift(tp)  # Get drift
        # update n_processed
        self.n_processed += 1
        # Return result for writer
        return self.trap_locs.to_dict(tp)

    def run(self, tp):
        if self.n_processed == 0:
            self._initialise_traps(self.tile_size)
        self.find_drift(tp)  # Get drift
        # update n_processed
        self.n_processed += 1
        # Return result for writer
        return self.trap_locs.to_dict(tp)

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

    def get_channel_index(self, item):
        for i, ch in enumerate(self.channels):
            if item in ch:
                return i

    def get_position_annotation(self):
        # TODO required for matlab support
        return None
