"""Segment/segmented pipelines.
Includes splitting the image into traps/parts,
cell segmentation, nucleus segmentation."""
import warnings

import h5py
from skimage import feature
import numpy as np
import pandas as pd
from pathlib import Path, PosixPath

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


class Tiler:
    def __init__(self, raw_expt, finished=True, template=None, local=None):
        self.expt = raw_expt
        self.finished = finished
        if template is None:
            template = trap_template
        self.trap_template = template
        self.pos_mapper = dict()
        self._current_position = self.expt.positions[0]
        # Local version of annotation to use, by default None in which case
        # it will use whatever is available on OMERO
        self.local = local

    def __getitem__(self, pos):
        # Can ask for a position
        if pos not in self.pos_mapper.keys():
            if self.local:  # Use local version
                annotation = self.local
            else:
                annotation = self.expt.get_position(pos).annotation  # Returns
            # none if non-existent
            self.pos_mapper[pos] = TimelapseTiler(
                self.expt.get_position(pos),
                self.trap_template,
                finished=self.finished,
                annotation=annotation,
            )
        return self.pos_mapper[pos]

    @property
    def trap_locations(self):
        return self.current_tiler.trap_locations

    @property
    def n_timepoints(self):
        return self.current_tiler.n_timepoints

    @property
    def n_traps(self):
        return self.current_tiler.n_traps

    @property
    def positions(self):
        return self.expt.positions

    @property
    def current_position(self):
        return self.expt.current_position

    @current_position.setter
    def current_position(self, pos):
        self.expt.current_position = pos

    @property
    def current_tiler(self):
        """The current TimelapseTiler object

        Corresponds to the TimelapseTiler that runs on the currently active
        position on the experiment.
        For most methods, the Tiler object is only a proxy object for what
        ever TimelapseTiler is currently active.

        For example:
        $ tiler.get_trap_timelapse(0, 96) == tiler.current_tiler.get_trap_timelapse(0, 96)

        This object cannot be set, directly. If you want to change the
        position you are working on, you need to modify self.current_position.
        If you want to directly access another position's tiler without
        changing the current active position, use:
        $ alternate_tiler = tiler[position_name]
        """
        pos_name = self.current_position.name
        return self[pos_name]

    @property
    def channels(self):
        return self.expt.channels

    def get_channel_index(self, channel):
        return self.current_position.get_channel_index(channel)

    def get_trap_timelapse(self, trap_id, tile_size=96, channels=None, z=None):
        return self.current_tiler.get_trap_timelapse(
            trap_id, tile_size=tile_size, channels=channels, z=z
        )

    def get_traps_timepoint(self, tp, tile_size=96, channels=None, z=None):
        return self.current_tiler.get_traps_timepoint(
            tp, tile_size=tile_size, channels=channels, z=z
        )

    def run(self, keys, store, **kwargs):
        save_dir = self.expt.root_dir
        for pos, tps in accumulate(keys):
            self[pos].run(tps, store, save_dir, **kwargs)
        return keys


class TrapLocations:
    def __init__(self, initial_location, initial_time=0):
        self._initial_location = initial_location
        self._drifts = [np.array([0, 0])]
        self._timepoints = [initial_time]

    @property
    def n_timepoints(self):
        return len(self._timepoints)

    @property
    def n_traps(self):
        return len(self._initial_location)

    @property
    def drifts(self):
        return np.stack(self._drifts)

    def __getitem__(self, item):
        return self._initial_location - np.sum(self.drifts[:item], axis=0)

    def __setitem__(self, key, value):
        if key in self._timepoints:
            self._drifts[key] = value
        else:
            self._drifts.append(value)
            self._timepoints.append(key)

    def __len__(self):
        return self.n_timepoints

    def __repr__(self):
        return "{} Traps, {} Timepoints, Drifts shape: {}".format(
            self.n_traps, self.n_timepoints, self.drifts.shape
        )

    def __iter__(self):
        i = 0
        while i < self.n_timepoints:
            yield self.__getitem__(i)
            i += 1


class TimelapseTiler:
    def __init__(self, timelapse, template, finished=True, annotation=None):
        self.timelapse = timelapse
        self.trap_template = template
        self.trap_locations = []  # Todo: make a dummy TrapLocations with len(0)
        self._reference = None
        if finished and not annotation:
            self.tile_timelapse()
        elif annotation:
            self.trap_locations = from_annotation(annotation)
        else:
            # TODO ?
            pass
            self.trap_locations = from_hdf(annotation)

    def tile_timelapse(self, channel: int = 0):
        """
        Finds the tile positions in a time lapse (including drifts).
        :param timelapse: The Timelapse object holding the raw data
        :param channel: Which channel to use for tiling, default=0
        :return: A dictionary containing trap centers as data.
        trap_locations[timepoint][trap_id]
        """
        self._initialise_locations(0, channel=0)
        for i in range(1, self.timelapse.size_t):
            self.trap_locations[i] = self._get_drift(
                self.trap_locations._drifts[-1], i, channel=channel
            )
        return

    def _initialise_locations(self, timepoint, channel=0):
        img = np.squeeze(self.timelapse[channel, timepoint, :, :, 0])

        self.trap_locations = TrapLocations(
            identify_trap_locations(img, self.trap_template), timepoint
        )
        self._reference = centre(img)

    def _get_transform(self, timepoint, channel=0):
        # Todo: switch to this using OpenCV once it has been tested.
        raise NotImplementedError(
            "This function uses OpenCV and " "has not yet been implemented."
        )

    #     image = centre(self.timelapse[channel, timepoint, :, :, 0])
    #     transform, _ = cv2.estimateAffinePartial2D(self._reference, image)
    #     if transform is None:
    #         return np.eye(2), np.zeros(2)
    #     self._reference = image
    #     return np.array_split(transform, 2, axis=1)

    @property
    def n_timepoints(self):
        return self.trap_locations.n_timepoints

    @property
    def n_traps(self):
        return self.trap_locations.n_traps

    def _get_drift(self, prev_drift, timepoint, channel=0, reference_reset_drift=25):
        """
        Get drift between this timepoint and the reference image.
        Note that this function changes the reference image, so
        it should be used with caution.
        :param timepoint: Time point to run analysis on
        :param channel: Channel to run analysis on
        :param reference_reset_drift: Maximum drift allowed, else assume none.
        :return: drift
        """
        image = centre(np.squeeze(self.timelapse[channel, timepoint, :, :, 0]))
        (
            drift,
            _,
            _,
        ) = feature.register_translation(self._reference, image)
        if any(
            [
                np.abs(x - y).max() > reference_reset_drift
                for x, y in zip(drift, prev_drift)
            ]
        ):
            return np.zeros(2)
        self._reference = image
        return drift

    def get_trap_timelapse(self, trap_id, tile_size=96, channels=None, z=None, t=None):
        """
        Get a timelapse for a given trap by specifying the trap_id
        :param trap_id: An integer defining which trap to choose. Counted
        between 0 and Tiler.n_traps - 1
        :param tile_size: The size of the trap tile (centered around the
        trap as much as possible, edge cases exist)
        :param channels: Which channels to fetch, indexed from 0.
        If None, defaults to [0]
        :param z: Which z_stacks to fetch, indexed from 0.
        If None, defaults to [0].
        :return: A numpy array with the timelapse in (C,T,X,Y,Z) order
        """
        # TODO is there a better way of separating the two?
        if isinstance(self.timelapse, TimelapseOMERO):
            return get_trap_timelapse_omero(
                self.timelapse,
                self.trap_locations,
                trap_id,
                tile_size=tile_size,
                channels=channels,
                z=z,
                t=t,
            )
        return get_trap_timelapse(
            self.timelapse,
            self.trap_locations,
            trap_id,
            tile_size=tile_size,
            channels=channels,
            z=z,
        )

    def get_traps_timepoint(self, tp, tile_size=96, channels=None, z=None):
        """
        Get all the traps from a given timepoint
        :param tp:
        :param tile_size:
        :param channels:
        :param z:
        :return: A numpy array with the traps in the (trap, C, T, X, Y,
        Z) order
        """
        return get_traps_timepoint(
            self.timelapse,
            self.trap_locations,
            tp,
            tile_size=tile_size,
            channels=channels,
            z=z,
        )

    def _check_contiguous_time(self, timepoints):
        # Fixme check fails
        if max(timepoints) < self.n_timepoints:
            warnings.warn(
                "Requested timepoints {} but timepoints already "
                "processed until time {}"
                ".".format(timepoints, self.n_timepoints)
            )
        contiguous = np.arange(self.n_timepoints, max(timepoints) + 1)
        if not all([x == y for x, y in zip(contiguous, timepoints)]):
            raise ValueError(
                "Timepoints not contiguous: expected {}, "
                "got {}".format(list(contiguous), timepoints)
            )

    def clear_cache(self):
        self.timelapse.clear_cache()

    def run(self, keys, store, save_dir):
        """
        :param keys: a list of timepoints to run tiling on.
        :return:
        """
        position = self.timelapse.name
        timepoints = sorted(keys)
        # Initialise the store
        store_file = get_store_path(save_dir, store, position)
        # TODO remove
        print(f"Tiler: Running {position} to {store_file}")
        with h5py.File(store_file, "a") as h5:
            store = h5.require_group("/trap_info/")
            # RUN TRAP INFO
            if "processed_timepoints" in store:
                processed = store["processed_timepoints"]
            else:
                processed = store.create_dataset(
                    "processed_timepoints",
                    shape=(len(timepoints),),
                    maxshape=(None,),
                    dtype=np.uint16,
                )
            # modify the time points based on the processed values
            timepoints = [t for t in timepoints if t not in processed]
            try:
                max_tp = max(timepoints)
            except ValueError:  # There are no timepoints left
                return timepoints
            if "trap_locations" in store:
                trap_locs = store["trap_locations"][()]
                self.trap_locations._initial_location = trap_locs
            else:
                self._initialise_locations(0)
                store.create_dataset("trap_locations", data=self.trap_locations[0])
            # DRIFTS
            if "drifts" in store:
                drifts = store["drifts"]
                # Expand the dataset to reach max_tp spots
                if drifts.shape[0] <= max_tp:
                    drifts.resize(max_tp + 1, axis=0)
            else:  # No drifts yet
                drifts = store.create_dataset(
                    "drifts", shape=(max_tp + 1, 2), maxshape=(None, 2)
                )
            for tp in timepoints:
                drift = self._get_drift(self.trap_locations._drifts[-1], tp)
                self.trap_locations[tp] = drift
                # Note this overwrites the drift for any given time point if
                # it has already been run
                drifts[tp] = drift
            # Keep track of which time points have been processed
            if processed.shape[0] < max_tp:
                processed.resize(max_tp, axis=0)
            processed[timepoints] = timepoints
        return timepoints


def from_matlab(mat_timelapse):
    """Create an initialised Timelapse Tiler from a Matlab Object"""
    if isinstance(mat_timelapse, (str, Path)):
        mat_timelapse = matObject(mat_timelapse)
    timelapse_traps = mat_timelapse.get(
        "timelapseTrapsOmero", mat_timelapse.get("timelapseTraps", None)
    )
    if timelapse_traps is None:
        warnings.warn("Could not initialise from matlab")
        return None
    # The image rotation term takes into account the fact that in
    # some experiments the images are flipped wrt to the cTimepoint data for some reason??
    image_rotation = timelapse_traps["image_rotation"]
    if image_rotation == -90:
        order = ["ycenter", "xcenter"]
    else:
        order = ["xcenter", "ycenter"]

    mat_trap_locs = timelapse_traps["cTimepoint"]["trapLocations"]
    # Rewrite into 3D array of shape (time, trap, x/y) from dictionary
    try:
        mat_trap_locs = np.dstack([mat_trap_locs[order[0]], mat_trap_locs[order[1]]])
    except (TypeError, IndexError):
        mat_trap_locs = np.dstack(
            [
                [loc[order[0]] for loc in mat_trap_locs if isinstance(loc, dict)],
                [loc[order[1]] for loc in mat_trap_locs if isinstance(loc, dict)],
            ]
        ).astype(int)
    trap_locations = TrapLocations(initial_location=mat_trap_locs[0])
    # Get drifts TODO check order is it loc_(x+1) - loc_(x) or vice versa?
    drifts = mat_trap_locs[1:] - mat_trap_locs[:-1]
    drifts = -drifts
    for i, drift in enumerate(drifts):
        tp = i + 1
        # TODO check that all drifts are identical
        trap_locations[tp] = drifts[i][0]
    return trap_locations


def from_hdf(store_name):
    with h5py.File(store_name, "r") as store:
        traps = store.require_group("trap_info")
        trap_locations = TrapLocations(initial_location=traps["trap_locations"][()])
        # Drifts
        for i, drift in enumerate(traps["drifts"][()]):
            trap_locations[i] = drift
        return trap_locations


def from_annotation(annotation):
    if isinstance(annotation, matObject):
        return from_matlab(annotation)
    if isinstance(annotation, str) or isinstance(annotation, PosixPath):
        return from_hdf(annotation)
    return None
