"""Segment/segmented pipelines.
Includes splitting the image into traps/parts,
cell segmentation, nucleus segmentation."""
from skimage import feature
import numpy as np

from core.traps import identify_trap_locations

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
    ref = raw_data[channel, 0, :, :, 0]
    size_t = raw_data.shape[1]
    # TODO take only a central region of the image

    drift = [np.array([0, 0])]
    references = [0]
    processed = 0
    for i in range(1, size_t):
        img = raw_data[channel, i, :, :, 0]

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
        self.trap_locations = None
        self.cell_outlines = None
        self.compartment_outlines = None

    def tile_timelapse(self):
        # Get the drift and references
        # Find traps in the references
        # Add drift to obtain the trap locations for each image
        # Compare reference direct traps-location to trap-location from drift
        pass
