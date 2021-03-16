"""
A set of utilities for dealing with ALCATRAS traps
"""

import numpy as np
from skimage import transform, feature


def identify_trap_locations(image, trap_template, optimize_scale=True,
                            downscale=0.35, trap_size=None):
    """
    Identify the traps in a single image based on a trap template.
    This assumes a trap template that is similar to the image in question
    (same camera, same magification; ideally same experiment).

    This method speeds up the search by downscaling both the image and
    the trap template before running the template match.
    It also optimizes the scale and the rotation of the trap template.

    :param image:
    :param trap_template:
    :param optimize_scale:
    :param downscale:
    :param trap_rotation:
    :return:
    """
    trap_size = trap_size if trap_size is not None else trap_template.shape[0]
    img = transform.rescale(image, downscale)
    temp = transform.rescale(trap_template, downscale)

    # TODO random search hyperparameter optimization
    # optimize rotation
    matches = {rotation: feature.match_template(
        img, transform.rotate(temp,
                              rotation,
                              cval=np.median(img)),
        pad_input=True,
        mode='median'
    ) ** 2 for rotation in [0, 90, 180, 270]}
    best_rotation = max(matches, key=lambda x: np.percentile(matches[x], 99.9))
    temp = transform.rotate(temp, best_rotation, cval=np.median(img))

    if optimize_scale:
        scales = np.linspace(0.5, 2, 10)
        matches = {scale: feature.match_template(
            img, transform.rescale(temp, scale),
            mode='median',
            pad_input=True) ** 2
                   for scale in scales}
        best_scale = max(matches, key=lambda x: np.percentile(matches[x],
                                                              99.9))
        matched = matches[best_scale]
    else:
        matched = feature.match_template(img, temp, pad_input=True,
                                         mode='median')

    coordinates = feature.peak_local_max(
        transform.rescale(matched, 1 / downscale),
        min_distance=trap_template.shape[0] * 0.70,
        exclude_border=trap_size // 3)
    return coordinates


def get_tile_shapes(x, tile_size, max_shape):
    half_size = tile_size // 2
    xmin = int(x[0] - half_size)
    ymin = max(0, int(x[1] - half_size))
    # if xmin + tile_size > max_shape[0]:
    #     xmin = max_shape[0] - tile_size
    # if ymin + tile_size > max_shape[1]:
    # #     ymin = max_shape[1] - tile_size
    # return max(xmin, 0), xmin + tile_size, max(ymin, 0), ymin + tile_size
    return xmin, xmin + tile_size, ymin, ymin + tile_size

def in_image(img, xmin, xmax, ymin, ymax, xidx=2, yidx=3):
    if xmin >= 0 and ymin >= 0:
        if xmax < img.shape[xidx] and ymax < img.shape[yidx]:
            return True
    else:
        return False

def get_xy_tile(img, xmin, xmax, ymin, ymax, xidx=2, yidx=3, pad_val=None):
    if pad_val is None:
        pad_val = np.median(img)
    # Get the tile from the image
    idx = [slice(None)] * len(img.shape)
    idx[xidx] = slice(max(0, xmin), min(xmax, img.shape[xidx]))
    idx[yidx] = slice(max(0, ymin), min(ymax, img.shape[yidx]))
    tile = img[tuple(idx)]
    # Check if the tile is in the image
    if in_image(img, xmin, xmax, ymin, ymax, xidx, yidx):
        return tile
    else:
        # Add padding
        pad_shape = [(0, 0)] * len(img.shape)
        pad_shape[xidx] = (max(-xmin, 0), max(xmax - img.shape[xidx], 0))
        pad_shape[yidx] = (max(-ymin, 0), max(ymax - img.shape[yidx], 0))
        tile = np.pad(tile, pad_shape, constant_values=pad_val)
    return tile

def get_trap_timelapse(raw_expt, trap_locations, trap_id, tile_size=96,
                       channels=None,
                       z=None):
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
    # Set the defaults (list is mutable)
    channels = channels if channels is not None else [0]
    z = z if z is not None else [0]
    # Get trap location for that id:
    trap_centers = [trap_locations[i][trap_id]
                    for i in
                    range(len(trap_locations))]

    max_shape = (raw_expt.shape[2], raw_expt.shape[3])
    tiles_shapes = [get_tile_shapes((x[0], x[1]), tile_size, max_shape)
                    for x in trap_centers]

    # Fixme: is this less efficient on OMERO?
    timelapse = [get_xy_tile(raw_expt[channels, i, :, :, z],
                             xmin, xmax, ymin, ymax, pad_val=None)
                 for i, (xmin, xmax, ymin, ymax) in enumerate(tiles_shapes)]
    return np.hstack(timelapse)


def get_traps_timepoint(raw_expt, trap_locations, tp, tile_size=96,
                        channels=None, z=None):
    """
    Get all the traps from a given time point
    :param raw_expt:
    :param trap_locations:
    :param tp:
    :param tile_size:
    :param channels:
    :param z:
    :return: A numpy array with the traps in the (trap, C, T, X, Y,
    Z) order
    """

    # Set the defaults (list is mutable)
    channels = channels if channels is not None else [0]
    z = z if z is not None else [0]

    # Get full img
    img = raw_expt[channels, tp, :, :, z]
    pad_val = np.median(img)

    traps = []
    max_shape = (raw_expt.shape[2], raw_expt.shape[3])
    for trap_center in trap_locations[tp]:
        xmin, xmax, ymin, ymax= get_tile_shapes(trap_center, tile_size,
                                                 max_shape)
        traps.append(get_xy_tile(img, xmin, xmax, ymin, ymax, pad_val=pad_val))
    return np.stack(traps, axis=0)


def centre(img, percentage=0.3):
    y, x = img.shape
    cropx = int(np.ceil(x * percentage))
    cropy = int(np.ceil(y * percentage))
    startx = int(x // 2 - (cropx // 2))
    starty = int(y // 2 - (cropy // 2))
    return img[starty:starty + cropy, startx:startx + cropx]


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
