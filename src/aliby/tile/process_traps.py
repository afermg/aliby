"""Functions for identifying and dealing with ALCATRAS traps."""

from typing import Any, Optional, Union
import numpy as np
import numpy.typing as npt
from aliby.global_settings import global_settings
from skimage import feature, transform
from skimage.filters import threshold_otsu, gaussian
from skimage.filters.rank import entropy
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening
from skimage.segmentation import clear_border
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.figure import Figure
from matplotlib.axes import Axes

try:
    from skimage.morphology import footprint_rectangle

    HAS_FOOTPRINT_RECTANGLE = True
except ImportError:
    from skimage.morphology import square

    HAS_FOOTPRINT_RECTANGLE = False


def half_floor(x: int, tile_size: int) -> int:
    """Round down tile_size."""
    return x - tile_size // 2


def half_ceil(x: int, tile_size: int) -> int:
    """Round up tile_size."""
    return x + -(tile_size // -2)


def segment_traps(
    image: Union[npt.NDArray[np.floating[Any]], Any],
    tile_size: int,
    downscale: float = 0.4,
    disk_radius_frac: float = 0.01,
    square_size: int = 3,
    min_frac_tilesize: float = 0.3,
    **identify_traps_kwargs: Any,
) -> npt.NDArray[np.int_]:
    """
    Find a trap template using an entropy filter and Otsu thresholding.

    Pass template to identify_trap_locations. To obtain candidate traps
    the major axis length of a tile must be smaller than tilesize.

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
    identify_traps_kwargs:
        Passed to identify_trap_locations

    Returns
    -------
    traps: an array of pairs of integers
        The coordinates of the centroids of the traps.
    """
    if hasattr(image, "compute"):
        # convert to numpy if dask
        image = image.compute()
    # TODO
    image = correct_illumination(image, sigma=200)
    # adjust parameters for particular tile size
    scale_factor = (
        tile_size / global_settings.imaging_specifications["tile_size"]
    )
    disk_radius_frac *= scale_factor
    min_frac_tilesize *= scale_factor
    square_size = int(square_size * scale_factor)
    # bounds on major axis length of traps
    min_trap_size = min_frac_tilesize * tile_size
    # shrink image
    if downscale != 1:
        img = transform.rescale(image, downscale)
    else:
        img = image
    # generate an entropy image using a disk footprint
    disk_radius = int(min(img.shape) * disk_radius_frac)
    entropy_image = entropy(img_as_ubyte(img), disk(disk_radius))
    if downscale != 1:
        # upscale
        entropy_image = transform.rescale(entropy_image, 1 / downscale)
    # find Otsu threshold for entropy image
    thresh = threshold_otsu(entropy_image)
    # apply morphological closing to thresholded image
    if HAS_FOOTPRINT_RECTANGLE:
        bw = closing(
            entropy_image > thresh,
            footprint_rectangle((square_size, square_size)),
        )
    else:
        bw = closing(entropy_image > thresh, square(square_size))
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    # label distinct regions of the image
    label_image = label(cleared)
    # find regions likely to contain traps:
    # with a major axis length within a certain range
    # and a centroid at least half_tile_size away from the image edge
    half_tile_size = tile_size // 2
    idx_valid_region = [
        (i, region)
        for i, region in enumerate(regionprops(label_image))
        if (min_trap_size < region.major_axis_length < tile_size)
        and (
            half_tile_size
            < region.centroid[0]
            < image.shape[0] - half_tile_size - 1
        )
        and (
            half_tile_size
            < region.centroid[1]
            < image.shape[1] - half_tile_size - 1
        )
    ]
    if idx_valid_region:
        _, valid_region = zip(*idx_valid_region)
    else:
        raise ValueError("No valid tiles found.")
    # find centroids of valid regions
    centroids = (
        np.array([region.centroid for region in valid_region])
        .round()
        .astype(int)
    )
    # make candidate templates as tile_size x tile_size image slices
    candidate_templates = [
        image[
            half_floor(x, tile_size) : half_ceil(x, tile_size),
            half_floor(y, tile_size) : half_ceil(y, tile_size),
        ]
        for x, y in centroids
    ]
    # make a mean template by averaging all the candidate templates
    mean_template = np.stack(candidate_templates).mean(axis=0)
    # find traps using the mean trap template
    traps = identify_trap_locations(
        image, mean_template, **identify_traps_kwargs
    )
    # try again if there are too few traps
    traps_retry = []
    if len(traps) < 30 and downscale != 1:
        print("Tiler:TrapIdentification: Trying again.")
        traps_retry = segment_traps(image, tile_size, downscale=1)
    # TODO: uncomment to help debug
    # plot_trap_locations(image, traps, tile_size)
    # return results with the most number of traps
    if len(traps_retry) < len(traps):
        return traps
    else:
        return traps_retry


def identify_trap_locations(
    image: npt.NDArray[np.floating[Any]],
    trap_template: npt.NDArray[np.floating[Any]],
    optimize_scale: bool = True,
    downscale: float = 0.35,
    trap_size: Optional[int] = None,
) -> npt.NDArray[np.int_]:
    """
    Identify the traps in a single image based on a trap template.

    Requires the trap template to be similar to the image
    (same camera, same magnification - ideally the same experiment).

    Use normalised correlation in scikit-image's to match_template.

    The search is sped up by down-scaling both the image and
    the trap template before running the template matching.

    The trap template is rotated and re-scaled to improve matching.
    The parameters of the rotation and re-scaling are optimised, although
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
    # careful: the image is float16
    img = transform.rescale(image.astype(np.float16), downscale)
    template = transform.rescale(trap_template, downscale)
    # try multiple rotations of template to determine
    # which best matches the image
    # result uses absolute value because the sign is unimportant
    matches = {
        rotation: np.abs(
            feature.match_template(
                img,
                transform.rotate(template, rotation, cval=np.median(img)),
                pad_input=True,
                mode="median",
            )
        )
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
            scale: np.abs(
                feature.match_template(
                    img,
                    transform.rescale(template, scale),
                    mode="median",
                    pad_input=True,
                )
            )
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


def correct_illumination(
    image: npt.NDArray[np.floating[Any]], sigma: float
) -> npt.NDArray[np.floating[Any]]:
    """
    Correct uneven illumination using gaussian background estimation.

    Return image normalised to [0, 1] range for compatibility with
    skimage functions that expect float images in this range.
    """
    image_float = image.astype(np.float32)
    background = gaussian(image_float, sigma=sigma, preserve_range=True)
    corrected = image_float - background + np.median(background)
    corrected = np.clip(corrected, 0, None)
    # normalise to [0, 1] range for compatibility with skimage
    if corrected.max() > 0:
        corrected = corrected / corrected.max()
    return corrected


def plot_trap_locations(
    image: Union[npt.NDArray[np.floating[Any]], Any],
    trap_coordinates: npt.NDArray[np.int_],
    tile_size: int,
    figsize: tuple[float, float] = (9, 9),
    marker_size: Optional[float] = None,
) -> tuple[Figure, Axes]:
    """
    Visualise the image with predicted trap locations overlaid.

    Parameters
    ----------
    image : 2D array
        The microscopy image
    trap_coordinates : array of pairs of integers
        The (row, col) coordinates of trap centroids
    tile_size : integer
        Size of the tile (used to scale marker size)
    figsize : tuple of floats (optional)
        Figure size in inches
    marker_size : float (optional)
        Radius of circles marking trap locations. If None, set to
        tile_size / 2

    Returns
    -------
    fig : matplotlib figure
        The figure object
    ax : matplotlib axes
        The axes object
    """
    if hasattr(image, "compute"):
        # convert to numpy if dask
        image = image.compute()
    if marker_size is None:
        marker_size = tile_size / 7
    # create figure
    fig, ax = plt.subplots(figsize=figsize)
    # display image with greyscale colourmap
    ax.imshow(image, cmap="gray", interpolation="nearest")
    # overlay trap locations as circles
    for row, col in trap_coordinates:
        circle = Circle(
            (col, row),
            radius=marker_size,
            fill=False,
            edgecolor="red",
            linewidth=2,
            alpha=0.7,
        )
        ax.add_patch(circle)
    # add title with trap count
    ax.set_title(f"Trap locations (n={len(trap_coordinates)})", fontsize=14)
    ax.set_xlabel("Column (pixels)", fontsize=12)
    ax.set_ylabel("Row (pixels)", fontsize=12)
    # remove tick labels for cleaner visualisation
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show(block=False)
    return fig, ax
