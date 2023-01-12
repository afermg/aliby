"""
A set of utilities for dealing with ALCATRAS traps
"""


import numpy as np
from skimage import feature, transform
from skimage.filters import threshold_otsu
from skimage.filters.rank import entropy
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, square
from skimage.segmentation import clear_border
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
    min_frac_tilesize=0.3,
    **identify_traps_kwargs,
):
    """
    Uses an entropy filter and Otsu thresholding to find a trap template,
    which is then passed to identify_trap_locations.

    To obtain candidate traps it the major axis length of a tile must be smaller than tilesize.

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
    min_mal = min_frac_tilesize * tile_size

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
        if min_mal < region.major_axis_length < tile_size
        and tile_size // 2
        < region.centroid[0]
        < half_floor(image.shape[0], tile_size) - 1
        and tile_size // 2
        < region.centroid[1]
        < half_floor(image.shape[1], tile_size) - 1
    ]

    assert idx_valid_region, "No valid tiling regions found"

    _, valid_region = zip(*idx_valid_region)

    # find centroids and minor axes lengths of valid regions
    centroids = (
        np.array([x.centroid for x in valid_region]).round().astype(int)
    )
    minals = [region.minor_axis_length for region in valid_region]
    # coords for best trap
    x, y = np.round(centroids[np.argmin(minals)]).astype(int)

    # make candidate templates from the other traps found
    candidate_templates = [
        image[
            half_floor(x, tile_size) : half_ceil(x, tile_size),
            half_floor(y, tile_size) : half_ceil(y, tile_size),
        ]
        for x, y in centroids
    ]
    # make a mean template from all the found traps
    mean_template = np.dstack(candidate_templates).astype(int).mean(axis=-1)

    # find traps using the mean trap template
    traps = identify_trap_locations(
        image, mean_template, **identify_traps_kwargs
    )

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
    # FIXME Used in aliby.utils.imageViewer
    image = ((image - image.min()) / (image.max() - image.min())) * 255
    minval = np.percentile(image, 2)
    maxval = np.percentile(image, 98)
    image = np.clip(image, minval, maxval)
    image = (image - minval) / (maxval - minval)
    return image