"""
A set of utilities for dealing with ALCATRAS traps
"""

from copy import copy

import numpy as np
import matplotlib.colors as colors
from tqdm import tqdm

from skimage import transform, feature
from skimage.filters.rank import entropy
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, closing, square
from skimage.registration import phase_cross_correlation
from skimage.util import img_as_ubyte


def half_floor(x, tile_size):
    return x - tile_size // 2


def half_ceil(x, tile_size):
    return x + -(tile_size // -2)


def segment_traps(
    image,
    tile_size,
    downscale=0.4,
    disk_radius_frac=0.01,
    square_size=3,
    min_frac_tilesize=0.2,
    max_frac_tilesize=0.8,
    **identify_traps_kwargs,
):
    """
    Uses an entropy filter and Otsu thresholding to find a trap template,
    which is then passed to identify_trap_locations.

    The hyperparameters have not been optimised.

    Parameters
    ----------
    image: 2D array
    tile_size: integer
        Size of the tile
    downscale: float (optional)
        Fraction by which to shrink image
    disk_radius_frac: float (optional)
        Radius of disk using in the entropy filter
    square_size: integer (optional)
        Parameter for a morphological closing applied to thresholded
        image
    min_frac_tilesize: float (optional)
    max_frac_tilesize: float (optional)
        Used to determine bounds on the major axis length of regions
        suspected of containing traps.
    identify_traps_kwargs:
        Passed to identify_trap_locations

    Returns
    -------
    traps: an array of pairs of integers
        The coordinates of the centroids of the traps
    """
    # keep a memory of image in case need to re-run
    img = image
    # bounds on major axis length of traps
    min_mal = min_frac_tilesize * np.sqrt(2) * tile_size
    max_mal = max_frac_tilesize * np.sqrt(2) * tile_size

    # shrink image
    if downscale != 1:
        img = transform.rescale(image, downscale)
    # generate an entropy image using a disk footprint
    disk_radius = int(min([disk_radius_frac * x for x in img.shape]))
    entropy_image = entropy(img_as_ubyte(img), disk(disk_radius))
    if downscale != 1:
        entropy_image = transform.rescale(entropy_image, 1 / downscale)
    # find Otsu threshold for entropy image
    thresh = threshold_otsu(entropy_image)
    # apply morphological closing to thresholded, and so binary, image
    bw = closing(entropy_image > thresh, square(square_size))
    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label distinct regions of the image
    label_image = label(cleared)
    # find regions likely to contain traps:
    # with a major axis length within a certain range
    # and a centroid at least tile_size // 2 away from the image edge
    idx_valid_region = [
        (i, region)
        for i, region in enumerate(regionprops(label_image))
        if min_mal < region.major_axis_length < max_mal
        and tile_size // 2
        < region.centroid[0]
        < half_floor(image.shape[0], tile_size) - 1
        and tile_size // 2
        < region.centroid[1]
        < half_floor(image.shape[1], tile_size) - 1
    ]
    idx, valid_region = zip(*idx_valid_region)

    # find centroids and minor axes lengths of valid regions
    centroids = (
        np.array([x.centroid for x in valid_region]).round().astype(int)
    )
    minals = [region.minor_axis_length for region in valid_region]
    # coords for best trap
    x, y = np.round(centroids[np.argmin(minals)]).astype(int)

    # make a template using the best trap in the image
    template = image[
        half_floor(x, tile_size):half_ceil(x, tile_size),
        half_floor(y, tile_size):half_ceil(y, tile_size),
    ]
    # make candidate templates from the other traps found
    candidate_templates = [
        image[
            half_floor(x, tile_size):half_ceil(x, tile_size),
            half_floor(y, tile_size):half_ceil(y, tile_size),
        ]
        for x, y in centroids
    ]
    # make a mean template from all the found traps
    mean_template = np.dstack(candidate_templates).astype(int).mean(axis=-1)

    # find traps using the best found trap
    traps = identify_trap_locations(image, template, **identify_traps_kwargs)
    # find traps using the mean trap template
    mean_traps = identify_trap_locations(
        image, mean_template, **identify_traps_kwargs
    )
    # choose the approach that identifies the most traps
    if len(traps) < len(mean_traps):
        traps = mean_traps

    # if there are too few traps, try again
    traps_retry = []
    if len(traps) < 30 and downscale != 1:
        print("Tiler:TrapIdentification: Trying again.")
        traps_retry = segment_traps(image, tile_size, downscale=1)

    # return results with the most number of traps
    if len(traps_retry) < len(traps):
        return traps
    else:
        return traps_retry


###


def identify_trap_locations(
    image, trap_template, optimize_scale=True, downscale=0.35, trap_size=None
):
    """
    Identify the traps in a single image based on a trap template,
    which requires the trap template to be similar to the image
    (same camera, same magification - ideally the same experiment).

    Uses normalised correlation in scikit-image's match_template.

    The search is speeded up by downscaling both the image and
    the trap template before running the template matching.

    The trap template is rotated and re-scaled to improve matching.
    The parameters of the rotation and rescaling are optimised, although
    over restricted ranges.

    Parameters
    ----------
    image: 2D array
    trap_template: 2D array
    optimize_scale : boolean (optional)
    downscale: float (optional)
        Fraction by which to downscale to increase speed
    trap_size: integer (optional)
        If unspecified, the size is determined from the trap_template

    Returns
    -------
    coordinates: an array of pairs of integers
    """
    if trap_size is None:
        trap_size = trap_template.shape[0]
    # careful: the image is float16!
    img = transform.rescale(image.astype(float), downscale)
    template = transform.rescale(trap_template, downscale)

    # try multiple rotations of template to determine
    # which best matches the image
    # result is squared because the sign of the correlation is unimportant
    matches = {
        rotation: feature.match_template(
            img,
            transform.rotate(template, rotation, cval=np.median(img)),
            pad_input=True,
            mode="median",
        )
        ** 2
        for rotation in [0, 90, 180, 270]
    }
    # find best rotation
    best_rotation = max(matches, key=lambda x: np.percentile(matches[x], 99.9))
    # rotate template by best rotation
    template = transform.rotate(template, best_rotation, cval=np.median(img))

    if optimize_scale:
        # try multiple scales appled to template to determine which
        # best matches the image
        scales = np.linspace(0.5, 2, 10)
        matches = {
            scale: feature.match_template(
                img,
                transform.rescale(template, scale),
                mode="median",
                pad_input=True,
            )
            ** 2
            for scale in scales
        }
        # find best scale
        best_scale = max(
            matches, key=lambda x: np.percentile(matches[x], 99.9)
        )
        # choose the best result - an image of normalised correlations
        # with the template
        matched = matches[best_scale]
    else:
        # find the image of normalised correlations with the template
        matched = feature.match_template(
            img, template, pad_input=True, mode="median"
        )
    # re-scale back the image of normalised correlations
    # find the coordinates of local maxima
    coordinates = feature.peak_local_max(
        transform.rescale(matched, 1 / downscale),
        min_distance=int(trap_size * 0.70),
        exclude_border=(trap_size // 3),
    )
    return coordinates


###############################################################
# functions below here do not appear to be used any more
###############################################################


def stretch_image(image):
    image = ((image - image.min()) / (image.max() - image.min())) * 255
    minval = np.percentile(image, 2)
    maxval = np.percentile(image, 98)
    image = np.clip(image, minval, maxval)
    image = (image - minval) / (maxval - minval)
    return image


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


def tile_where(centre, x, y, MAX_X, MAX_Y):
    # Find the position of the tile
    xmin = int(centre[1] - x // 2)
    ymin = int(centre[0] - y // 2)
    xmax = xmin + x
    ymax = ymin + y
    # What do we actually have available?
    r_xmin = max(0, xmin)
    r_xmax = min(MAX_X, xmax)
    r_ymin = max(0, ymin)
    r_ymax = min(MAX_Y, ymax)
    return xmin, ymin, xmax, ymax, r_xmin, r_ymin, r_xmax, r_ymax


def get_tile(shape, center, raw_expt, ch, t, z):
    """Returns a tile from the raw experiment with a given shape.

    :param shape: The shape of the tile in (C, T, Z, Y, X) order.
    :param center: The x,y position of the centre of the tile
    :param
    """
    _, _, x, y, _ = shape
    _, _, MAX_X, MAX_Y, _ = raw_expt.shape
    tile = np.full(shape, np.nan)

    # Find the position of the tile
    xmin = int(center[1] - x // 2)
    ymin = int(center[0] - y // 2)
    xmax = xmin + x
    ymax = ymin + y
    # What do we actually have available?
    r_xmin = max(0, xmin)
    r_xmax = min(MAX_X, xmax)
    r_ymin = max(0, ymin)
    r_ymax = min(MAX_Y, ymax)

    # Fill values
    tile[
        :,
        :,
        (r_xmin - xmin): (r_xmax - xmin),
        (r_ymin - ymin): (r_ymax - ymin),
        :,
    ] = raw_expt[ch, t, r_xmin:r_xmax, r_ymin:r_ymax, z]
    # fill_val = np.nanmedian(tile)
    # np.nan_to_num(tile, nan=fill_val, copy=False)
    return tile


def get_traps_timepoint(
    raw_expt, trap_locations, tp, tile_size=96, channels=None, z=None
):
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
    z_positions = z if z is not None else [0]
    if isinstance(z_positions, slice):
        n_z = z_positions.stop
        z_positions = list(range(n_z))  # slice is not iterable error
    elif isinstance(z_positions, list):
        n_z = len(z_positions)
    else:
        n_z = 1

    n_traps = len(trap_locations[tp])
    trap_ids = list(range(n_traps))
    shape = (len(channels), 1, tile_size, tile_size, n_z)
    # all tiles
    zct_tiles, slices, trap_ids = all_tiles(
        trap_locations, shape, raw_expt, z_positions, channels, [tp], trap_ids
    )
    # TODO Make this an explicit function in TimelapseOMERO
    images = raw_expt.pixels.getTiles(zct_tiles)
    # Initialise empty traps
    traps = np.full((n_traps,) + shape, np.nan)
    for trap_id, (z, c, _, _), (y, x), image in zip(
        trap_ids, zct_tiles, slices, images
    ):
        ch = channels.index(c)
        z_pos = z_positions.index(z)
        traps[trap_id, ch, 0, x[0]: x[1], y[0]: y[1], z_pos] = image
    for x in traps:  # By channel
        np.nan_to_num(x, nan=np.nanmedian(x), copy=False)
    return traps


def centre(img, percentage=0.3):
    y, x = img.shape
    cropx = int(np.ceil(x * percentage))
    cropy = int(np.ceil(y * percentage))
    startx = int(x // 2 - (cropx // 2))
    starty = int(y // 2 - (cropy // 2))
    return img[starty: starty + cropy, startx: startx + cropx]


def align_timelapse_images(
    raw_data, channel=0, reference_reset_time=80, reference_reset_drift=25
):
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

        shifts, _, _ = phase_cross_correlation(ref, img)
        # If a huge move is detected at a single time point it is taken
        # to be inaccurate and the correction from the previous time point
        # is used.
        # This might be common if there is a focus loss for example.
        if any(
            [
                abs(x - y) > reference_reset_drift
                for x, y in zip(shifts, drift[-1])
            ]
        ):
            shifts = drift[-1]

        drift.append(shifts)
        ref = img

        # TODO test necessity for references, description below
        #   If the images have drifted too far from the reference or too
        #   much time has passed we change the reference and keep track of
        #   which images are kept as references
    return np.stack(drift), ref
