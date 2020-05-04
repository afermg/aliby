"""Segment/segmented pipelines.
Includes splitting the image into traps/parts,
cell segmentation, nucleus segmentation."""
from skimage import feature
import numpy as np
from pathlib import Path

from core.traps import identify_trap_locations
from core.baby_client import BabyClient

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


def align_timelapse_images(raw_data, channel=0, reference_reset_time=80,
                           reference_reset_drift=25):
    """
    Uses image registration to align images in the timelapse.
    Uses the channel with id `channel` to perform the registration.

    Starts with the first timepoint as a reference and changes the
    reference to the current timepoint if either the images have moved
    by half of a trap width or `reference_reset_time` has been reached.

    Sets `self.drift`, a 3D numpy array with shape (t, drift_x, drift_y).
    We assume no drift occurs in the z-direction.

    :param reference_reset_drift: Upper bound on the allowed drift before
    resetting the reference image.
    :param reference_reset_time: Upper bound on number of time points to
    register before resetting the reference image.
    :param channel: index of the channel to use for image registration.
    """

    def centre(img, percentage=0.3):
        y, x = img.shape
        cropx = int(np.ceil(x * percentage))
        cropy = int(np.ceil(y * percentage))
        startx = int(x // 2 - (cropx // 2))
        starty = int(y // 2 - (cropy // 2))
        return img[starty:starty + cropy, startx:startx + cropx]

    ref = centre(np.squeeze(raw_data[channel, 0, :, :, 0]))
    size_t = raw_data.shape[1]

    drift = [np.array([0, 0])]
    for i in range(1, size_t):
        img = centre(np.squeeze(raw_data[channel, i, :, :, 0]))

        shifts, _, _ = feature.register_translation(ref, img)
        # If a huge move is detected at a single time point it is taken
        # to be inaccurate and the correction from the previous time point
        # is used.
        # This might be common if there is a focus loss for example.
        if any([abs(x - y) > reference_reset_drift
                for x, y in zip(shifts, drift[-1])]):
            shifts = drift[-1]

        drift.append(shifts)
        ref = img

        # TODO test necessity for references, description below
        #   If the images have drifted too far from the reference or too
        #   much time has passed we change the reference and keep track of
        #   which images are kept as references
    return np.stack(drift)


class SegmentedExperiment(object):
    def __init__(self, raw_expt, baby_url='http://localhost:5101',
                 baby_config=dict()):
        """
        A base class for a segmented experiment.

        :param raw_expt:
        """
        self.raw_expt = raw_expt
        self.rotation = None
        self.trap_locations = dict()
        self.cell_outlines = None
        self.compartment_outlines = None

        # Set up the baby client
        self.baby_client = BabyClient(url=baby_url, **baby_config)

        # Tile the current position
        self.trap_locations[self.current_position] = self.tile_timelapse()

    @property
    def n_traps(self):
        return len(self.trap_locations[self.current_position])

    @property
    def positions(self):
        return self.raw_expt.positions

    @property
    def current_position(self):
        return str(self.raw_expt.current_position)

    @current_position.setter
    def current_position(self, position):
        self.raw_expt.current_position = position
        # Tile that position
        if self.current_position not in self.trap_locations.keys():
            self.trap_locations[self.current_position] = self.tile_timelapse()

    @property
    def channels(self):
        return self.raw_expt.channels

    def get_channel_index(self, channel):
        return self.raw_expt.current_position.get_channel_index(channel)

    def tile_timelapse(self, channel=0):
        # Get the drift and references
        drifts = align_timelapse_images(self.raw_expt, channel=channel)
        # Find traps in the references
        trap_locations = {0: identify_trap_locations(
            np.squeeze(self.raw_expt[channel, 0, :, :, 0]), trap_template)}
        for i in range(len(drifts)):
            trap_locations[i] = trap_locations[0] \
                                - np.sum(drifts[:i, [1, 0]], axis=0)

        return trap_locations

    def get_trap_timelapse(self, trap_id, tile_size=96, channels=None, z=None):
        """
        Get a timelapse for a given trap by specifying the trap_id
        :param trap_id: An integer defining which trap to choose. Counted
        between 0 and SegmentedExperiment.n_traps - 1
        :param tile_size: The size of the trap tile (centered around the
        trap as much as possible, edge cases exist)
        :param channels: Which channels to fetch, indexed from 0.
        If None, defaults to [0]
        :param z: Which z_stacks to fetch, indexed from 0.
        If None, defaults to [0].
        :return: A numpy array with the timelapse in (C,T,X,Y,Z) order
        """
        # Set the defaults (list is mutable)
        channels = channels if channels is not None else [0]
        z = z if z is not None else [0]
        # Get trap location for that id:
        trap_centers = [self.trap_locations[self.current_position][i][trap_id]
                        for i in
                        range(len(self.trap_locations[self.current_position]))]

        max_shape = (self.raw_expt.shape[2], self.raw_expt.shape[3])
        tiles_shapes = [get_tile_shapes(x, tile_size, max_shape)
                        for x in trap_centers]

        timelapse = [self.raw_expt[channels, i, xmin:xmax, ymin:ymax, z] for
                     i, (xmin, xmax, ymin, ymax) in enumerate(tiles_shapes)]
        return np.hstack(timelapse)

    def get_traps_timepoint(self, tp, tile_size=96, channels=[0], z=[0]):
        """
        Get all the traps from a given timepoint
        :param tp:
        :param tile_size:
        :param channels:
        :param z:
        :return: A numpy array with the traps in the (trap, C, T, X, Y,
        Z) order
        """
        traps = []
        max_shape = (self.raw_expt.shape[2], self.raw_expt.shape[3])
        for trap_center in self.trap_locations[self.current_position]:
            xmin, xmax, ymin, ymax = get_tile_shapes(trap_center, tile_size,
                                                     max_shape)
            traps.append(self.raw_expt[channels, tp, xmin:xmax, ymin:ymax, z])
        return np.stack(traps)

    def cell_segmentation_per_trap(self, trap_id):
        pass

    def cell_segmentation_per_position(self, pos):
        pass

    def segment_full_experiment(self):
        pass
