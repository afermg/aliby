"""Segment/segmented pipelines.
Includes splitting the image into traps/parts,
cell segmentation, nucleus segmentation."""
import warnings

from skimage import feature
import numpy as np
import pandas as pd
from pathlib import Path

from core.io.matlab import matObject
from core.traps import identify_trap_locations, get_trap_timelapse, \
    get_traps_timepoint, centre
from core.utils import accumulate

trap_template_directory = Path(__file__).parent / 'trap_templates'
# TODO do we need multiple templates, one for each setup?
trap_template = np.load(trap_template_directory / 'trap_prime.npy')


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
    def __init__(self, raw_expt, finished=True, matlab=None, template=None):
        self.expt = raw_expt
        self.finished = finished
        self.matlab = matlab
        if template is None: 
            template = trap_template
        self.trap_template = template
        self.pos_mapper = dict()
        self._current_position = self.expt.positions[0]

    def __getitem__(self, pos):
        # Can ask for a position
        if pos not in self.pos_mapper.keys():
            pos_matlab = self._load_matlab(pos)
            self.pos_mapper[pos] = TimelapseTiler(self.expt.get_position(pos),
                                                  self.trap_template,
                                                  finished=self.finished,
                                                  matlab=pos_matlab)
        return self.pos_mapper[pos]

    def _load_matlab(self, pos):
        if self.matlab:
            pos_matlab = pos + self.matlab
            mat_timelapse = matObject(self.expt.root_dir / pos_matlab)
            return mat_timelapse
        else:
            return None

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
        pos = self._current_position
        if pos not in self.pos_mapper: 
            self.pos_mapper[pos] = self[pos]
        return self._current_position

    @current_position.setter
    def current_position(self, pos):
        self._current_position = pos

    @property
    def channels(self):
        return self.expt.channels

    def get_channel_index(self, channel):
        return self.expt.current_position.get_channel_index(channel)

    def get_trap_timelapse(self, trap_id, tile_size=96, channels=None, z=None):
        return self.pos_mapper[self.current_position].get_trap_timelapse(
            trap_id, tile_size=tile_size, channels=channels, z=z
        )

    def get_traps_timepoint(self, tp, tile_size=96, channels=None, z=None):
        return self.pos_mapper[self.current_position].get_traps_timepoint(
            tp, tile_size=tile_size, channels=channels, z=z
        )

    def run(self, keys, **kwargs):
        for pos, tps in accumulate(keys):
            self[pos].run(tps, **kwargs)
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
        return self._initial_location - np.sum(self.drifts[:item],
                axis=0)

    def __setitem__(self, key, value):
        if key in self._timepoints:
            self._drifts[key] = value
        else:
            self._drifts.append(value)
            self._timepoints.append(key)

    def __len__(self):
        return self.n_timepoints

    def __repr__(self):
        pass

class TimelapseTiler:
    def __init__(self, timelapse, template, finished=True, matlab=None):
        self.timelapse = timelapse
        self.trap_template = template
        self.trap_locations = [] # Todo: make a dummy TrapLocations with len(0)
        self._reference = None
        if finished and not matlab:
            self.tile_timelapse()
        elif matlab:
            self.trap_locations = from_matlab(matlab)

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
            self.trap_locations[i] = self._get_drift(self.trap_locations._drifts[-1], i, channel=channel)
        return

    def _initialise_locations(self, timepoint, channel=0):
        img = np.squeeze(self.timelapse[channel, timepoint, :, :, 0])

        self.trap_locations = TrapLocations(identify_trap_locations(
            img, self.trap_template
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
        drift, _, _, = feature.register_translation(self._reference, image)
        if any([np.abs(x-y).max() > reference_reset_drift for x,y in zip(drift, prev_drift)]):
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
        # Fixme check fails
        if max(timepoints) < self.n_timepoints:
            warnings.warn("Requested timepoints {} but timepoints already "
                             "processed until time {}"
                             ".".format(timepoints, self.n_timepoints))
        contiguous = np.arange(self.n_timepoints, max(timepoints) + 1)
        if not all([x==y for x,y in zip(contiguous,timepoints)]):
            raise ValueError("Timepoints not contiguous: expected {}, "
                             "got {}".format(list(contiguous), timepoints))

    def clear_cache(self):
        self.timelapse.clear_cache()

    def run(self, keys, trap_store, drift_store):
        """
        :param keys: a list of timepoints to run tiling on.
        :return:
        """
        timepoints = sorted(keys)
        # Get the position in the database that corresponds to the timelapse
        # of this tiler
        position = self.timelapse.name
        if len(self.trap_locations) == 0:
            initial_tp = timepoints.pop(0)
            self._initialise_locations(initial_tp)
            # Create a dataframe of all found traps
            traps = []
            for i in range(self.trap_locations.n_traps):
                x, y = self.trap_locations[initial_tp][i]
                traps.append((position, i, int(x), int(y)))
        # Save initial location of the traps
        trap_df = pd.DataFrame(traps, columns=['position', 'trap', 'x', 'y'])
        with open(trap_store, 'a') as f:
            trap_df.to_csv(f, header=f.tell() == 0)
        #self._check_contiguous_time(timepoints)
        drifts = []
        for tp in timepoints:
            drift = self._get_drift(self.trap_locations._drifts[-1], tp)
            self.trap_locations[tp] = drift
            # Update the drifts
            y, x = drift
            drifts.append((position, tp, x, y))
        drift_df = pd.DataFrame(drifts, columns=['position', 'timepoint',
                                                 'x', 'y'])
        with open(drift_store, 'a') as f:
            drift_df.to_csv(f, header=f.tell() == 0)
        return keys

def from_matlab(mat_timelapse):
    """Create an initialised Timelapse Tiler from a Matlab Object"""
    if isinstance(mat_timelapse, (str, Path)):
        mat_timelapse = matObject(mat_timelapse)
    # TODO what if it isn't a timelapseTrapsOmero?
    mat_trap_locs = mat_timelapse['timelapseTrapsOmero']['cTimepoint'][
        'trapLocations']
    # Rewrite into 3D array of shape (time, trap, x/y) from dictionary
    try:
        mat_trap_locs = np.dstack([mat_trap_locs['ycenter'], mat_trap_locs[
            'xcenter']])
    except TypeError:
        mat_trap_locs = np.dstack([
                                    [loc['ycenter'] for loc in mat_trap_locs
                                        if isinstance(loc, dict)],
                                    [loc['xcenter'] for loc in mat_trap_locs
                                        if isinstance(loc, dict)]
                                   ]).astype(int)
    trap_locations = TrapLocations(initial_location=mat_trap_locs[0])
    # Get drifts TODO check order is it loc_(x+1) - loc_(x) or vice versa?
    drifts = mat_trap_locs[1:] - mat_trap_locs[:-1]
    drifts = -drifts
    for i, drift in enumerate(drifts):
        tp = i + 1
        # TODO check that all drifts are identical
        trap_locations[tp] = drifts[i][0]
    return trap_locations


