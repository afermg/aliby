"""Segment/segmented pipelines.
Includes splitting the image into traps/parts,
cell segmentation, nucleus segmentation."""
import warnings
from functools import lru_cache

import h5py
import numpy as np

from pathlib import Path, PosixPath

from skimage.registration import phase_cross_correlation

from core.traps import segment_traps
from core.timelapse import TimelapseOMERO
from core.io.matlab import matObject
from core.traps import (
    identify_trap_locations,
    get_trap_timelapse,
    get_traps_timepoint,
    centre,
    get_trap_timelapse_omero,
)
from core.utils import accumulate, get_store_path

from core.io.writer import Writer
from core.io.metadata_parser import parse_logfiles

trap_template_directory = Path(__file__).parent / "trap_templates"
# TODO do we need multiple templates, one for each setup?
trap_template = np.load(trap_template_directory / "trap_prime.npy")


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
    def __init__(self, initial_location, tile_size, max_size=1200):
        self.tile_size = tile_size
        self.max_size = max_size
        self.initial_location = initial_location
        self.traps = [Trap(centre, self, tile_size, max_size) for centre in
                      initial_location]
        self.drifts = []

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
            res['trap_locations'] = self.initial_location
            res['attrs/tile_size'] = self.tile_size
            res['attrs/max_size'] = self.max_size
        res['drifts'] = np.expand_dims(self.drifts[tp], axis=0)
        # res['processed_timepoints'] = tp
        return res

    @classmethod
    def read_hdf5(cls, file):
        with h5py.File(file, 'r') as hfile:
            trap_info = hfile['trap_info']
            initial_locations = trap_info['trap_locations'][()]
            drifts = trap_info['drifts'][()]
            max_size = trap_info.attrs['max_size']
            tile_size = trap_info.attrs['tile_size']
        trap_locs = cls(initial_locations, tile_size, max_size=max_size)
        trap_locs.drifts = drifts
        return trap_locs


class Tiler:
    """A dummy TimelapseTiler object fora Dask Demo.

    Does trap finding and image registration."""

    def __init__(self, image, metadata, template=None, tile_size=None,
                 ref_channel='Brightfield', ref_z=0):
        self.image = image
        self.name = metadata['name']
        self.channels = metadata['channels']
        self.trap_template = template if template is not None else \
            trap_template
        self.ref_channel = self.get_channel_index(ref_channel)
        self.ref_z = ref_z
        self.tile_size = tile_size or min(trap_template.shape)
        self.trap_locs = None
        self._tiles = None
        # FIXME REMOVE
        self.expt = None

    @classmethod
    def read_hdf5(image, filepath):
        # TODO
        # trap_locs = TrapLocations.read_hdf5(filename)
        # get metadata out of HDF5
        pass

    @lru_cache(maxsize=100)
    def get_tc(self, t, c):
        # Get image
        full = self.image[t, c].compute()  # FORCE THE CACHE
        if self.trap_locs.padding_required(t):
            # Add padding step so edge traps are fixed
            pad_width = (self.tile_size // 2, self.tile_size // 2)
            full = np.pad(full, ((0, 0), pad_width, pad_width), 'median')
        return full

    @property
    def shape(self):
        c, t, z, y, x = self.image.shape
        return (c,t, x, y, z)

    @property
    def n_processed(self):
        if self.trap_locs:
            return self.trap_locs.shape[1]
        return 0

    @property
    def n_traps(self):
        return len(self.trap_locs)

    @property
    def finished(self):
        return self.n_processed == self.image.shape[0]

    @property
    def tiles(self):
        return self._tiles

    def _initialise_traps(self, trap_template, tile_size):
        """Find initial trap positions.

        Removes all those that are too close to the edge so no padding is necessary.
        """
        half_tile = tile_size // 2
        max_size = min(self.image.shape[-2:])
        initial_image = self.image[
            0, self.ref_channel, self.ref_z]  # First time point, first channel, first z-position
        trap_locs = segment_traps(initial_image, tile_size)
        trap_locs = [[x, y] for x, y in trap_locs if
                     half_tile < x < max_size - half_tile and half_tile < y < max_size - half_tile]
        self.trap_locs = TrapLocations(trap_locs, tile_size)

    def find_drift(self, tp):
        # TODO check that the drift doesn't move any tiles out of the image, remove them from list if so
        prev_tp = max(0, tp - 1)
        drift, error, _ = phase_cross_correlation(
            self.image[prev_tp, self.ref_channel, self.ref_z],
            self.image[tp, self.ref_channel, self.ref_z])
        self.trap_locs.drifts.append(drift)

    def get_tp_data(self, tp, c):
        traps = []
        full = self.get_tc(tp, c)
        for trap in self.trap_locs:
            y, x = trap.as_range(tp)
            trap = full[:, y, x]
            # Check the shape of the trap
            if trap.shape != (5, self.tile_size,
                              self.tile_size):  # TODO find a better fix for this
                trap = np.zeros((5, self.tile_size, self.tile_size))
            traps.append(trap)
        return np.stack(traps)

    def run_tp(self, tp):
        assert tp >= self.n_processed, "Time point already processed"
        # TODO check contiguity?
        if self.n_processed == 0:
            self._initialise_traps(self.trap_template, self.tile_size)
        self.find_drift(tp)  # Get drift
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
        return self.channels.index(item)

    def get_position_annotation(self):
        # TODO required for matlab support
        return None

# ############################################################################
#
# class Tiler:
#     def __init__(self, raw_expt, finished=True, template=None, local=None):
#         self.expt = raw_expt
#         self.finished = finished
#         if template is None:
#             template = trap_template
#         self.trap_template = template
#         self.pos_mapper = dict()
#         self._current_position = self.expt.positions[0]
#         # Local version of annotation to use, by default None in which case
#         # it will use whatever is available on OMERO
#         self.local = local
#
#     def __getitem__(self, pos):
#         # Can ask for a position
#         if pos not in self.pos_mapper.keys():
#             if self.local:  # Use local version
#                 annotation = self.local
#             else:
#                 annotation = self.expt.get_position(pos).annotation  # Returns
#             # none if non-existent
#             self.pos_mapper[pos] = TimelapseTiler(
#                 self.expt.get_position(pos),
#                 self.trap_template,
#                 finished=self.finished,
#                 annotation=annotation,
#             )
#         return self.pos_mapper[pos]
#
#     @property
#     def trap_locations(self):
#         return self.current_tiler.trap_locations
#
#     @property
#     def n_timepoints(self):
#         return self.current_tiler.n_timepoints
#
#     @property
#     def n_traps(self):
#         return self.current_tiler.n_traps
#
#     @property
#     def positions(self):
#         return self.expt.positions
#
#     @property
#     def current_position(self):
#         return self.expt.current_position
#
#     @current_position.setter
#     def current_position(self, pos):
#         self.expt.current_position = pos
#
#     @property
#     def current_tiler(self):
#         """The current TimelapseTiler object
#
#         Corresponds to the TimelapseTiler that runs on the currently active
#         position on the experiment.
#         For most methods, the Tiler object is only a proxy object for what
#         ever TimelapseTiler is currently active.
#
#         For example:
#         $ tiler.get_trap_timelapse(0, 96) == tiler.current_tiler.get_trap_timelapse(0, 96)
#
#         This object cannot be set, directly. If you want to change the
#         position you are working on, you need to modify self.current_position.
#         If you want to directly access another position's tiler without
#         changing the current active position, use:
#         $ alternate_tiler = tiler[position_name]
#         """
#         pass
#
#     def get_trap_timelapse(self, trap_id, tile_size=96, channels=None, z=None):
#         return self.current_tiler.get_trap_timelapse(
#             trap_id, tile_size=tile_size, channels=channels, z=z
#         )
#
#     def get_traps_timepoint(self, tp, tile_size=96, channels=None, z=None):
#         return self.current_tiler.get_traps_timepoint(
#             tp, tile_size=tile_size, channels=channels, z=z
#         )
#
#     def run(self, keys, store, **kwargs):
#         save_dir = self.expt.root_dir
#         for pos, tps in accumulate(keys):
#             self[pos].run(tps, store, save_dir, **kwargs)
#         return keys
#
#
# class TimelapseTiler:
#     def __init__(self, timelapse, template, finished=True, annotation=None):
#         self.timelapse = timelapse
#         self.trap_template = template
#         self.trap_locations = []  # Todo: make a dummy TrapLocations with len(0)
#         self._reference = None
#         if finished and not annotation:
#             self.tile_timelapse()
#         elif annotation:
#             self.trap_locations = from_annotation(annotation)
#         else:
#             # TODO ?
#             pass
#             self.trap_locations = from_hdf(annotation)
#
#     def tile_timelapse(self, channel: int = 0):
#         """
#         Finds the tile positions in a time lapse (including drifts).
#         :param timelapse: The Timelapse object holding the raw data
#         :param channel: Which channel to use for tiling, default=0
#         :return: A dictionary containing trap centers as data.
#         trap_locations[timepoint][trap_id]
#         """
#         self._initialise_locations(0, channel=0)
#         for i in range(1, self.timelapse.size_t):
#             self.trap_locations[i] = self._get_drift(
#                 self.trap_locations._drifts[-1], i, channel=channel
#             )
#         return
#
#     def _initialise_locations(self, timepoint, channel=0):
#         img = np.squeeze(self.timelapse[channel, timepoint, :, :, 0])
#
#         self.trap_locations = TrapLocations(
#             identify_trap_locations(img, self.trap_template), timepoint
#         )
#         self._reference = centre(img)
#
#     def _get_transform(self, timepoint, channel=0):
#         # Todo: switch to this using OpenCV once it has been tested.
#         raise NotImplementedError(
#             "This function uses OpenCV and " "has not yet been implemented."
#         )
#
#     #     image = centre(self.timelapse[channel, timepoint, :, :, 0])
#     #     transform, _ = cv2.estimateAffinePartial2D(self._reference, image)
#     #     if transform is None:
#     #         return np.eye(2), np.zeros(2)
#     #     self._reference = image
#     #     return np.array_split(transform, 2, axis=1)
#
#     def _get_drift(self, prev_drift, timepoint, channel=0,
#                    reference_reset_drift=25):
#         """
#         Get drift between this timepoint and the reference image.
#         Note that this function changes the reference image, so
#         it should be used with caution.
#         :param timepoint: Time point to run analysis on
#         :param channel: Channel to run analysis on
#         :param reference_reset_drift: Maximum drift allowed, else assume none.
#         :return: drift
#         """
#         image = centre(np.squeeze(self.timelapse[channel, timepoint, :, :, 0]))
#         (
#             drift,
#             _,
#             _,
#         ) = feature.register_translation(self._reference, image)
#         if any(
#                 [
#                     np.abs(x - y).max() > reference_reset_drift
#                     for x, y in zip(drift, prev_drift)
#                 ]
#         ):
#             return np.zeros(2)
#         self._reference = image
#         return drift
#
#     def get_trap_timelapse(self, trap_id, tile_size=96, channels=None, z=None,
#                            t=None):
#         """
#         Get a timelapse for a given trap by specifying the trap_id
#         :param trap_id: An integer defining which trap to choose. Counted
#         between 0 and Tiler.n_traps - 1
#         :param tile_size: The size of the trap tile (centered around the
#         trap as much as possible, edge cases exist)
#         :param channels: Which channels to fetch, indexed from 0.
#         If None, defaults to [0]
#         :param z: Which z_stacks to fetch, indexed from 0.
#         If None, defaults to [0].
#         :return: A numpy array with the timelapse in (C,T,X,Y,Z) order
#         """
#         # TODO is there a better way of separating the two?
#         if isinstance(self.timelapse, TimelapseOMERO):
#             return get_trap_timelapse_omero(
#                 self.timelapse,
#                 self.trap_locations,
#                 trap_id,
#                 tile_size=tile_size,
#                 channels=channels,
#                 z=z,
#                 t=t,
#             )
#         return get_trap_timelapse(
#             self.timelapse,
#             self.trap_locations,
#             trap_id,
#             tile_size=tile_size,
#             channels=channels,
#             z=z,
#         )
#
#     def get_traps_timepoint(self, tp, tile_size=96, channels=None, z=None):
#         """
#         Get all the traps from a given timepoint
#         :param tp:
#         :param tile_size:
#         :param channels:
#         :param z:
#         :return: A numpy array with the traps in the (trap, C, T, X, Y,
#         Z) order
#         """
#         return get_traps_timepoint(
#             self.timelapse,
#             self.trap_locations,
#             tp,
#             tile_size=tile_size,
#             channels=channels,
#             z=z,
#         )
#
#     def _check_contiguous_time(self, timepoints):
#         # Fixme check fails
#         if max(timepoints) < self.n_timepoints:
#             warnings.warn(
#                 "Requested timepoints {} but timepoints already "
#                 "processed until time {}"
#                 ".".format(timepoints, self.n_timepoints)
#             )
#         contiguous = np.arange(self.n_timepoints, max(timepoints) + 1)
#         if not all([x == y for x, y in zip(contiguous, timepoints)]):
#             raise ValueError(
#                 "Timepoints not contiguous: expected {}, "
#                 "got {}".format(list(contiguous), timepoints)
#             )
#
#     def clear_cache(self):
#         self.timelapse.clear_cache()
#
#     def run(self, keys, store, save_dir):
#         """
#         :param keys: a list of timepoints to run tiling on.
#         :return:
#         """
#         position = self.timelapse.name
#         timepoints = sorted(keys)
#         # Initialise the store
#         store_file = get_store_path(save_dir, store, position)
#         # TODO remove
#
#         print(f'Tiler: Running {position} to {store_file}')
#         # ADD METADATA
#         metadata_writer = Writer(store_file)
#         metadata_dict = parse_logfiles(save_dir)
#
#         # intend to point to root rather than creating a new group called
#         # metadata, but somehow doesn't work -- may need to modify Writer.
#         metadata_writer.write(path='metadata',
#                               meta=metadata_dict,
#                               overwrite=True, )
#         with h5py.File(store_file, 'a') as h5:
#             store = h5.require_group('/trap_info/')
#         print(f"Tiler: Running {position} to {store_file}")
#         with h5py.File(store_file, "a") as h5:
#             store = h5.require_group("/trap_info/")
#             # RUN TRAP INFO
#             if "processed_timepoints" in store:
#                 processed = store["processed_timepoints"]
#             else:
#                 processed = store.create_dataset(
#                     "processed_timepoints",
#                     shape=(len(timepoints),),
#                     maxshape=(None,),
#                     dtype=np.uint16,
#                 )
#             # modify the time points based on the processed values
#             timepoints = [t for t in timepoints if t not in processed]
#             try:
#                 max_tp = max(timepoints)
#             except ValueError:  # There are no timepoints left
#                 return timepoints
#             if "trap_locations" in store:
#                 trap_locs = store["trap_locations"][()]
#                 self.trap_locations._initial_location = trap_locs
#             else:
#                 self._initialise_locations(0)
#                 store.create_dataset("trap_locations",
#                                      data=self.trap_locations[0])
#             # DRIFTS
#             if "drifts" in store:
#                 drifts = store["drifts"]
#                 # Expand the dataset to reach max_tp spots
#                 if drifts.shape[0] <= max_tp:
#                     drifts.resize(max_tp + 1, axis=0)
#             else:  # No drifts yet
#                 drifts = store.create_dataset(
#                     "drifts", shape=(max_tp + 1, 2), maxshape=(None, 2)
#                 )
#             for tp in timepoints:
#                 drift = self._get_drift(self.trap_locations._drifts[-1], tp)
#                 self.trap_locations[tp] = drift
#                 # Note this overwrites the drift for any given time point if
#                 # it has already been run
#                 drifts[tp] = drift
#             # Keep track of which time points have been processed
#             if processed.shape[0] < max_tp:
#                 processed.resize(max_tp, axis=0)
#             processed[timepoints] = timepoints
#         return timepoints
#
#
# def from_matlab(mat_timelapse):
#     """Create an initialised Timelapse Tiler from a Matlab Object"""
#     if isinstance(mat_timelapse, (str, Path)):
#         mat_timelapse = matObject(mat_timelapse)
#     timelapse_traps = mat_timelapse.get(
#         "timelapseTrapsOmero", mat_timelapse.get("timelapseTraps", None)
#     )
#     if timelapse_traps is None:
#         warnings.warn("Could not initialise from matlab")
#         return None
#     # The image rotation term takes into account the fact that in
#     # some experiments the images are flipped wrt to the cTimepoint data for some reason??
#     image_rotation = timelapse_traps["image_rotation"]
#     if image_rotation == -90:
#         order = ["ycenter", "xcenter"]
#     else:
#         order = ["xcenter", "ycenter"]
#
#     mat_trap_locs = timelapse_traps["cTimepoint"]["trapLocations"]
#     # Rewrite into 3D array of shape (time, trap, x/y) from dictionary
#     try:
#         mat_trap_locs = np.dstack(
#             [mat_trap_locs[order[0]], mat_trap_locs[order[1]]])
#     except (TypeError, IndexError):
#         mat_trap_locs = np.dstack(
#             [
#                 [loc[order[0]] for loc in mat_trap_locs if
#                  isinstance(loc, dict)],
#                 [loc[order[1]] for loc in mat_trap_locs if
#                  isinstance(loc, dict)],
#             ]
#         ).astype(int)
#     trap_locations = TrapLocations(initial_location=mat_trap_locs[0])
#     # Get drifts TODO check order is it loc_(x+1) - loc_(x) or vice versa?
#     drifts = mat_trap_locs[1:] - mat_trap_locs[:-1]
#     drifts = -drifts
#     for i, drift in enumerate(drifts):
#         tp = i + 1
#         # TODO check that all drifts are identical
#         trap_locations[tp] = drifts[i][0]
#     return trap_locations
#
#
# def from_hdf(store_name):
#     with h5py.File(store_name, "r") as store:
#         traps = store.require_group("trap_info")
#         trap_locations = TrapLocations(
#             initial_location=traps["trap_locations"][()])
#         # Drifts
#         for i, drift in enumerate(traps["drifts"][()]):
#             trap_locations[i] = drift
#         return trap_locations
#
#
# def from_annotation(annotation):
#     if isinstance(annotation, matObject):
#         return from_matlab(annotation)
#     if isinstance(annotation, str) or isinstance(annotation, PosixPath):
#         return from_hdf(annotation)
#     return None
