"""Segment/segmented pipelines.
Includes splitting the image into traps/parts,
cell segmentation, nucleus segmentation."""
from skimage import feature
import numpy as np
from scipy.spatial import distance

from core.traps import identify_trap_locations

trap_template = np.load('/Users/s1893247/PhD/pipeline-core/core/trap_templates'
                        '/trap_bg_1.npy')


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
    ref = np.squeeze(raw_data[channel, 0, :, :, 0])
    size_t = raw_data.shape[1]
    # TODO take only a central region of the image

    drift = [np.array([0, 0])]
    references = [0]
    processed = 0
    for i in range(1, size_t):
        img = np.squeeze(raw_data[channel, i, :, :, 0])

        shifts, _, _ = feature.register_translation(ref, img)
        # If a huge move is detected at a single time point it is taken
        # to be inaccurate and the correction from the previous time point
        # is used.
        # This might be common if there is a focus loss for example.
        if any([abs(x - y) > reference_reset_drift
                for x, y in zip(shifts, drift[-1])]):
            shifts = drift[-1]

        drift.append(shifts)
        processed += 1

        # If the images have drifted too far from the reference or too
        # much time has passed we change the reference and keep track of
        # which images are kept as references
        if (any([x > reference_reset_drift for x in drift[-1]])
                or processed > reference_reset_time):
            ref = img
            processed = 0
            references.append(i)
    return np.stack(drift), references


class SegmentedExperiment(object):
    def __init__(self, raw_expt):
        """
        A base class for a segmented experiment.

        :param raw_expt:
        """
        self.raw_expt = raw_expt
        self.rotation = None
        self.trap_locations = self.tile_timelapse()
        self.cell_outlines = None
        self.compartment_outlines = None

    def tile_timelapse(self, channel=0):
        # Get the drift and references
        drifts, refs = align_timelapse_images(self.raw_expt, channel=channel)
        # Find traps in the references
        trap_locations = {ref: identify_trap_locations(
            np.squeeze(self.raw_expt[channel, ref, :, :, 0]),
            trap_template) for ref in refs}
        # Compare reference direct traps-location to trap-location from drift
        # Add drift to obtain the trap locations for each image
        # TODO make single for loop using conditional clause for references
        for j in range(len(refs) - 1):
            for i in range(refs[j], refs[j + 1]):
                trap_locations[i] = (
                        trap_locations[refs[j]] - drifts[i, [1, 0]])

            indices = np.argmin(distance.cdist(trap_locations[refs[j + 1]],
                                               trap_locations[refs[j]]
                                               - drifts[refs[j + 1], [1, 0]]),
                                axis=0)

            trap_locations[refs[j + 1]] = trap_locations[refs[j + 1]][indices]
        # Add the timepoints after the final reference
        for i in range(refs[-1], len(drifts)):
            trap_locations[i] = (
                    trap_locations[refs[-1]] - drifts[i, [1, 0]])
        return trap_locations

    def get_trap_timelapse(self, trap_id, tile_size=96, channels=[0],
                           z=[0]):
        # Get trap location for that id:
        trap_centers = [self.trap_locations[i][trap_id] for i in
                        range(len(self.trap_locations))]

        half_size = tile_size // 2

        def get_tile_shapes(x, max_shape=(self.raw_expt.shape[2],
                                          self.raw_expt.shape[3])):
            xmin = int(x[0] - half_size)
            ymin = max(0, int(x[1] - half_size))
            if xmin + tile_size > max_shape[0]:
                xmin = max_shape[0] - tile_size
            if ymin + tile_size > max_shape[1]:
                ymin = max_shape[1] - tile_size
            return xmin, xmin + tile_size, ymin, ymin + tile_size

        tiles_shapes = [get_tile_shapes(x) for x in trap_centers]

        timelapse = [self.raw_expt[channels, i, xmin:xmax, ymin:ymax, z] for
                     i, (xmin, xmax, ymin, ymax) in enumerate(tiles_shapes)]
        return np.hstack(timelapse)
