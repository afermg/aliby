"""
A set of utilities for dealing with ALCATRAS traps
"""

import numpy as np
from tqdm import tqdm

from skimage import transform, feature
from skimage.filters.rank import entropy
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, closing, square


def stretch_image(image):
    image = ((image - image.min()) / (image.max() - image.min())) * 255
    minval = np.percentile(image, 2)
    maxval = np.percentile(image, 98)
    image = np.clip(image, minval, maxval)
    image = (image - minval) / (maxval - minval)
    return image


def segment_traps(image, tile_size, downscale=0.4):
    # Make image go between 0 and 255
    img = image  # Keep a memory of image in case need to re-run
    # stretched = stretch_image(image)
    # img = stretch_image(image)
    # TODO Optimise the hyperparameters
    disk_radius = int(min([0.01 * x for x in img.shape]))
    min_area = 0.2 * (tile_size ** 2)
    if downscale != 1:
        img = transform.rescale(image, downscale)
    entropy_image = entropy(img, disk(disk_radius))
    if downscale != 1:
        entropy_image = transform.rescale(entropy_image, 1 / downscale)

    # apply threshold
    thresh = threshold_otsu(entropy_image)
    bw = closing(entropy_image > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    areas = [
        region.area
        for region in regionprops(label_image)
        if region.area > min_area and region.area < tile_size ** 2 * 0.8
    ]
    traps = (
        np.array(
            [
                region.centroid
                for region in regionprops(label_image)
                if region.area > min_area and region.area < tile_size ** 2 * 0.8
            ]
        )
        .round()
        .astype(int)
    )
    ma = (
        np.array(
            [
                region.minor_axis_length
                for region in regionprops(label_image)
                if region.area > min_area and region.area < tile_size ** 2 * 0.8
            ]
        )
        .round()
        .astype(int)
    )
    maskx = (tile_size // 2 < traps[:, 0]) & (
        traps[:, 0] < image.shape[0] - tile_size // 2
    )
    masky = (tile_size // 2 < traps[:, 1]) & (
        traps[:, 1] < image.shape[1] - tile_size // 2
    )

    traps = traps[maskx & masky, :]
    ma = ma[maskx & masky]

    chosen_trap_coords = np.round(traps[ma.argmin()]).astype(int)
    x, y = chosen_trap_coords
    template = image[
        x - tile_size // 2 : x + tile_size // 2, y - tile_size // 2 : y + tile_size // 2
    ]

    traps = identify_trap_locations(image, template)

    if len(traps) < 10 and downscale != 1:
        print("Trying again.")
        return segment_traps(image, tile_size, downscale=1)

    return traps


# def segment_traps(image, tile_size, downscale=0.4):
#     # Make image go between 0 and 255
#     img = image  # Keep a memory of image in case need to re-run
#     image = stretch_image(image)
#     # TODO Optimise the hyperparameters
#     disk_radius = int(min([0.01 * x for x in img.shape]))
#     min_area = 0.1 * (tile_size ** 2)
#     if downscale != 1:
#         img = transform.rescale(image, downscale)
#     entropy_image = entropy(img, disk(disk_radius))
#     if downscale != 1:
#         entropy_image = transform.rescale(entropy_image, 1 / downscale)

#     # apply threshold
#     thresh = threshold_otsu(entropy_image)
#     bw = closing(entropy_image > thresh, square(3))

#     # remove artifacts connected to image border
#     cleared = clear_border(bw)

#     # label image regions
#     label_image = label(cleared)
#     traps = [
#         region.centroid for region in regionprops(label_image) if region.area > min_area
#     ]
#     if len(traps) < 10 and downscale != 1:
#         print("Trying again.")
#         return segment_traps(image, tile_size, downscale=1)
#     return traps


def identify_trap_locations(
    image, trap_template, optimize_scale=True, downscale=0.35, trap_size=None
):
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
    # Careful, the image is float16!
    img = transform.rescale(image.astype(float), downscale)
    temp = transform.rescale(trap_template, downscale)

    # TODO random search hyperparameter optimization
    # optimize rotation
    matches = {
        rotation: feature.match_template(
            img,
            transform.rotate(temp, rotation, cval=np.median(img)),
            pad_input=True,
            mode="median",
        )
        ** 2
        for rotation in [0, 90, 180, 270]
    }
    best_rotation = max(matches, key=lambda x: np.percentile(matches[x], 99.9))
    temp = transform.rotate(temp, best_rotation, cval=np.median(img))

    if optimize_scale:
        scales = np.linspace(0.5, 2, 10)
        matches = {
            scale: feature.match_template(
                img, transform.rescale(temp, scale), mode="median", pad_input=True
            )
            ** 2
            for scale in scales
        }
        best_scale = max(matches, key=lambda x: np.percentile(matches[x], 99.9))
        matched = matches[best_scale]
    else:
        matched = feature.match_template(img, temp, pad_input=True, mode="median")

    coordinates = feature.peak_local_max(
        transform.rescale(matched, 1 / downscale),
        min_distance=int(trap_template.shape[0] * 0.70),
        exclude_border=(trap_size // 3),
    )
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


def get_trap_timelapse(
    raw_expt, trap_locations, trap_id, tile_size=117, channels=None, z=None
):
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
    trap_centers = [trap_locations[i][trap_id] for i in range(len(trap_locations))]

    max_shape = (raw_expt.shape[2], raw_expt.shape[3])
    tiles_shapes = [
        get_tile_shapes((x[0], x[1]), tile_size, max_shape) for x in trap_centers
    ]

    timelapse = [
        get_xy_tile(
            raw_expt[channels, i, :, :, z], xmin, xmax, ymin, ymax, pad_val=None
        )
        for i, (xmin, xmax, ymin, ymax) in enumerate(tiles_shapes)
    ]
    return np.hstack(timelapse)


def get_trap_timelapse_omero(
    raw_expt, trap_locations, trap_id, tile_size=117, channels=None, z=None, t=None
):
    """
    Get a timelapse for a given trap by specifying the trap_id
    :param raw_expt: A Timelapse object from which data is obtained
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
    z_positions = z if z is not None else [0]
    times = (
        t if t is not None else np.arange(raw_expt.shape[1])
    )  # TODO choose sub-set of time points
    shape = (len(channels), len(times), tile_size, tile_size, len(z_positions))
    # Get trap location for that id:
    zct_tiles, slices, trap_ids = all_tiles(
        trap_locations, shape, raw_expt, z_positions, channels, times, [trap_id]
    )

    # TODO Make this an explicit function in TimelapseOMERO
    images = raw_expt.pixels.getTiles(zct_tiles)
    timelapse = np.full(shape, np.nan)
    total = len(zct_tiles)
    for (z, c, t, _), (y, x), image in tqdm(
        zip(zct_tiles, slices, images), total=total
    ):
        ch = channels.index(c)
        tp = times.tolist().index(t)
        z_pos = z_positions.index(z)
        timelapse[ch, tp, x[0] : x[1], y[0] : y[1], z_pos] = image

    # for x in timelapse:  # By channel
    #    np.nan_to_num(x, nan=np.nanmedian(x), copy=False)
    return timelapse


def all_tiles(trap_locations, shape, raw_expt, z_positions, channels, times, traps):
    _, _, x, y, _ = shape
    _, _, MAX_X, MAX_Y, _ = raw_expt.shape

    trap_ids = []
    zct_tiles = []
    slices = []
    for z in z_positions:
        for ch in channels:
            for t in times:
                for trap_id in traps:
                    centre = trap_locations[t][trap_id]
                    xmin, ymin, xmax, ymax, r_xmin, r_ymin, r_xmax, r_ymax = tile_where(
                        centre, x, y, MAX_X, MAX_Y
                    )
                    slices.append(
                        ((r_ymin - ymin, r_ymax - ymin), (r_xmin - xmin, r_xmax - xmin))
                    )
                    tile = (r_ymin, r_xmin, r_ymax - r_ymin, r_xmax - r_xmin)
                    zct_tiles.append((z, ch, t, tile))
                    trap_ids.append(trap_id)  # So we remember the order!
    return zct_tiles, slices, trap_ids


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
        :, :, (r_xmin - xmin) : (r_xmax - xmin), (r_ymin - ymin) : (r_ymax - ymin), :
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
        traps[trap_id, ch, 0, x[0] : x[1], y[0] : y[1], z_pos] = image
    for x in traps:  # By channel
        np.nan_to_num(x, nan=np.nanmedian(x), copy=False)
    return traps


def centre(img, percentage=0.3):
    y, x = img.shape
    cropx = int(np.ceil(x * percentage))
    cropy = int(np.ceil(y * percentage))
    startx = int(x // 2 - (cropx // 2))
    starty = int(y // 2 - (cropy // 2))
    return img[starty : starty + cropy, startx : startx + cropx]


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

        shifts, _, _ = feature.register_translation(ref, img)
        # If a huge move is detected at a single time point it is taken
        # to be inaccurate and the correction from the previous time point
        # is used.
        # This might be common if there is a focus loss for example.
        if any([abs(x - y) > reference_reset_drift for x, y in zip(shifts, drift[-1])]):
            shifts = drift[-1]

        drift.append(shifts)
        ref = img

        # TODO test necessity for references, description below
        #   If the images have drifted too far from the reference or too
        #   much time has passed we change the reference and keep track of
        #   which images are kept as references
    return np.stack(drift)
