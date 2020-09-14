"""Segment/segmented pipelines.
Includes splitting the image into traps/parts,
cell segmentation, nucleus segmentation."""
import cv2
from skimage import feature
import numpy as np
import pandas as pd
from pathlib import Path

import core
from core.traps import identify_trap_locations, get_trap_timelapse, \
    get_traps_timepoint, align_timelapse_images, centre

trap_template_directory = Path(__file__).parent / 'trap_templates'
trap_template = np.load(trap_template_directory / 'trap_bg_1.npy')


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
    def __init__(self, raw_expt, finished=False):
        self.expt = raw_expt
        self.finished = finished
        self.pos_mapper = dict()
        self._current_position = self.expt.positions[0]

    @property
    def n_timepoints(self):
        return self.pos_mapper[self.current_position].n_timepoints

    @property
    def n_traps(self):
        return self.pos_mapper[self.current_position].n_traps

    @property
    def positions(self):
        return self.expt.positions

    @property
    def current_position(self):
        return self._current_position

    @current_position.setter
    def current_position(self, pos):
        if pos not in self.pos_mapper:
            self.pos_mapper[pos] = TimelapseTiler(self.expt.get_position(pos),
                                                  self.finished)
        self._current_position = pos
    @property
    def channels(self):
        return self.channels

    def get_channel_index(self, channel):
        return self.raw_expt.current_position.get_channel_index(channel)

    def get_trap_timelapse(self, trap_id, tile_size=96, channels=None, z=None):
        return self.pos_mapper[self.current_position].get_trap_timelapse(
            trap_id, tile_size=tile_size, channels=channels, z=z
        )

    def get_traps_timepoint(self, tp, tile_size=96, channels=None, z=None):
        return self.pos_mapper[self.current_position].get_traps_timepoint(
            tp, tile_size=tile_size, channels=channels, z=z
        )

    def run(self, keys):
        pass

class TrapLocations:
    def __init__(self, initial_location, initial_time=0):
        self._initial_location = initial_location
        self._drifts = [np.zeros((1, 2))]
        self._timepoints = [initial_time]

    @property
    def n_timepoints(self):
        return len(self._timepoints)

    @property
    def n_traps(self):
        return len(self._initial_location)

    def __getitem__(self, item):
        return self._initial_location + np.sum(self._drifts[:item + 1])

    def __setitem__(self, key, value):
        if key in self._timepoints:
            self._drifts[key] = value
        else:
            self._drifts.append(value)
            self._timepoints.append(key)

    def __repr__(self):
        pass

class TimelapseTiler:
    def __init__(self, timelapse, finished=False):
        self.timelapse = timelapse
        self.trap_locations = None
        self._reference = None
        if finished:
            self.tile_timelapse()

    def tile_timelapse(self, channel: int = 0):
        """
        Finds the tile positions in a time lapse (including drifts).
        :param timelapse: The Timelapse object holding the raw data
        :param channel: Which channel to use for tiling, default=0
        :return: A dictionary containing trap centers as data.
        trap_locations[timepoint][trap_id]
        """
        self._initialise_locations(0, channel=0)
        for i in range(self.timelapse.size_t):
            self.trap_locations[i] = self._get_drift(i, channel=channel)
        return

    def _initialise_locations(self, timepoint, channel=0):
        img = np.squeeze(self.timelapse[channel, timepoint, :, :, 0])

        self.trap_locations = TrapLocations(identify_trap_locations(
            img, trap_template
        ), timepoint)
        self._reference = centre(img)

    def _get_transform(self, timepoint, channel=0):
        # Todo: switch to this using OpenCV once it has been tested.
        raise NotImplementedError("This function uses OpenCV and "
                                  "has not yet been implemented.")
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

    def _get_drift(self, timepoint, channel=0, reference_reset_drift=25):
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
        drift, _, _, = feature.register_translation(self._reference, image)
        if any([abs(x) > reference_reset_drift
                for x in drift]):
            return np.zeros(2)
        self._reference = image
        return drift

    def get_trap_timelapse(self, trap_id, tile_size=96, channels=None, z=None):
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
        return get_trap_timelapse(self.timelapse,
                                  self.trap_locations,
                                  trap_id, tile_size=tile_size,
                                  channels=channels, z=z)

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
        return get_traps_timepoint(self.timelapse,
                                   self.trap_locations,
                                   tp, tile_size=tile_size, channels=channels,
                                   z=z)

    def _check_contiguous_time(self, timepoints):
        if max(timepoints) < self.n_timepoints:
            raise ValueError("Requested timepoints {} but timepoints already "
                             "processed until time {}"
                             ".".format(timepoints, self.n_timepoints))
        contiguous = np.arange(self.n_timepoints + 1, max(timepoints) + 1)
        if not all(contiguous == timepoints):
            raise ValueError("Timepoints not contiguous: expected {}, "
                             "got {}".format(list(contiguous), timepoints))

    def run(self, keys):
        """
        :param keys: a list of timepoints to run tiling on.
        :return:
        """
        timepoints = sorted(keys)
        if len(self.trap_locations) == 0:
            self._initialise_locations(timepoints.pop(0))
        self._check_contiguous_time(timepoints)
        for tp in timepoints:
            self.trap_locations[tp] = self._get_drift(tp)
        return keys
