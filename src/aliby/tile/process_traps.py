"""Functions for identifying and dealing with ALCATRAS traps."""

from typing import Any, Union
import numpy as np
import numpy.typing as npt
from aliby.global_settings import global_settings
from skimage import feature, transform
from skimage.filters import threshold_otsu, gaussian
from skimage.filters.rank import entropy
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
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
    """Calculate lower bound for a tile of tile_size centred at x."""
    return x - tile_size // 2


def half_ceil(x: int, tile_size: int) -> int:
    """Calculate upper bound for a tile of tile_size centred at x."""
    return x + -(tile_size // -2)


def segment_traps(
    image: Union[npt.NDArray[np.floating[Any]], Any],
    tile_size: int,
    downscale: float = 0.4,
    disk_radius_frac: float = 0.01,
    square_size: int = 3,
    min_frac_tilesize: float = 0.3,
    debug: bool = False,
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
        image.
    min_frac_tilesize: float (optional)
        Minimum fraction of tile_size for valid trap regions.
    debug: bool (optional)
        If True, plot trap locations for visual inspection.
    identify_traps_kwargs:
        Passed to identify_trap_locations.

    Returns
    -------
    traps: an array of pairs of integers
        The coordinates of the centroids of the traps.
    """
    if hasattr(image, "compute"):
        image = image.compute()
    image = correct_illumination(image, sigma=1.7 * tile_size)
    # scale parameters for tile size if tile size is not the default
    scale_factor = (
        tile_size / global_settings.imaging_specifications["tile_size"]
    )
    disk_radius_frac *= scale_factor
    min_frac_tilesize *= scale_factor
    square_size = int(square_size * scale_factor)
    min_trap_size = min_frac_tilesize * tile_size
    # find trap template
    entropy_image = _compute_entropy_image(image, downscale, disk_radius_frac)
    label_image = _segment_entropy_image(entropy_image, square_size)
    centroids = _find_valid_trap_regions(
        label_image, image.shape, tile_size, min_trap_size
    )
    mean_template = _create_mean_template(image, centroids, tile_size)
    # find traps using template matching
    traps = identify_trap_locations(
        image, mean_template, **identify_traps_kwargs
    )
    # retry with no downscaling if too few traps found
    traps_retry = []
    if len(traps) < 30 and downscale != 1:
        print(f"Tiler - TrapIdentification: Found {len(traps)}; trying again.")
        traps_retry = segment_traps(image, tile_size, downscale=1)
    # find result with most traps
    if len(traps_retry) < len(traps):
        final_traps = traps
    else:
        final_traps = traps_retry
    if debug:
        plot_trap_locations(image, final_traps, tile_size)
    return final_traps


def identify_trap_locations(
    image: npt.NDArray[np.floating[Any]],
    trap_template: npt.NDArray[np.floating[Any]],
    optimize_scale: bool = True,
    downscale: float = 0.35,
    trap_size: int | None = None,
) -> npt.NDArray[np.int_]:
    """
    Identify the traps in a single image based on a trap template.

    Requires the trap template to be similar to the image
    (same camera, same magnification - ideally the same experiment).

    Use normalised correlation via scikit-image's match_template.

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
    # downscale image and template for faster processing
    img = transform.rescale(image.astype(np.float32), downscale)
    template = transform.rescale(trap_template, downscale)
    # coarse rotation search at 90 degree intervals
    best_rotation, _ = _find_best_rotation(img, template, [0, 90, 180, 270])
    # fine rotation search around the best coarse rotation
    fine_rotations = list(
        np.linspace(best_rotation - 10, best_rotation + 10, 5)
    )
    best_rotation, _ = _find_best_rotation(img, template, fine_rotations)
    # apply best rotation to template
    template = transform.rotate(template, best_rotation, cval=np.median(img))
    if optimize_scale:
        # coarse scale search
        coarse_scales = np.linspace(0.6, 1.8, 7)
        best_scale, _ = _find_best_scale(img, template, coarse_scales)
        # fine scale search around best coarse scale
        fine_scales = np.linspace(best_scale - 0.1, best_scale + 0.1, 5)
        best_scale, matched = _find_best_scale(img, template, fine_scales)
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
    # filter by grid regularity to remove false positives
    coordinates = _filter_by_grid_regularity(coordinates, trap_size)
    return coordinates


def correct_illumination(
    image: npt.NDArray[np.floating[Any]], sigma: float
) -> npt.NDArray[np.floating[Any]]:
    """
    Correct uneven illumination using Gaussian background estimation.

    Return image normalised to [0, 1] range for compatibility with
    skimage functions that expect float images in this range.

    Parameters
    ----------
    image : 2D array
        The input image.
    sigma : float
        Standard deviation for Gaussian blur used to estimate background.

    Returns
    -------
    corrected : 2D array
        Background-corrected image normalised to [0, 1].
    """
    image_float = image.astype(np.float32)
    background = gaussian(image_float, sigma=sigma, preserve_range=True)
    corrected = image_float - background + np.median(background)
    corrected = np.clip(corrected, 0, None)
    # normalise to [0, 1] range for compatibility with skimage
    if corrected.max() > 0:
        corrected = corrected / corrected.max()
    return corrected


def _compute_entropy_image(
    image: npt.NDArray[np.floating[Any]],
    downscale: float,
    disk_radius_frac: float,
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute entropy image with optional downscaling for speed.

    Parameters
    ----------
    image : 2D array
        The input image.
    downscale : float
        Fraction by which to shrink image before computing entropy.
    disk_radius_frac : float
        Radius of disk footprint as fraction of image size.

    Returns
    -------
    entropy_image : 2D array
        The entropy-filtered image at original resolution.
    """
    if downscale != 1:
        img = transform.rescale(image, downscale)
    else:
        img = image
    disk_radius = int(min(img.shape) * disk_radius_frac)
    entropy_image = entropy(img_as_ubyte(img), disk(disk_radius))
    if downscale != 1:
        entropy_image = transform.rescale(entropy_image, 1 / downscale)
    return entropy_image


def _segment_entropy_image(
    entropy_image: npt.NDArray[np.floating[Any]],
    square_size: int,
) -> npt.NDArray[np.int_]:
    """
    Threshold entropy image and apply morphological cleaning.

    Parameters
    ----------
    entropy_image : 2D array
        The entropy-filtered image.
    square_size : int
        Size of square footprint for morphological closing.

    Returns
    -------
    label_image : 2D array
        Labelled regions after thresholding and cleaning.
    """
    thresh = threshold_otsu(entropy_image)
    if HAS_FOOTPRINT_RECTANGLE:
        bw = closing(
            entropy_image > thresh,
            footprint_rectangle((square_size, square_size)),
        )
    else:
        bw = closing(entropy_image > thresh, square(square_size))
    cleared = clear_border(bw)
    return label(cleared)


def _find_valid_trap_regions(
    label_image: npt.NDArray[np.int_],
    image_shape: tuple[int, int],
    tile_size: int,
    min_trap_size: float,
) -> npt.NDArray[np.int_]:
    """
    Find regions likely to be traps and return their centroids.

    Parameters
    ----------
    label_image : 2D array
        Labelled regions from segmentation.
    image_shape : tuple of (height, width)
        Shape of the original image.
    tile_size : int
        Size of tile in pixels.
    min_trap_size : float
        Minimum major axis length for valid traps.

    Returns
    -------
    centroids : array of shape (n, 2)
        Integer coordinates of valid trap centroids.

    Raises
    ------
    ValueError
        If no valid trap regions are found.
    """
    half_tile_size = tile_size // 2
    valid_regions = [
        region
        for region in regionprops(label_image)
        if (min_trap_size < region.major_axis_length < tile_size)
        and (
            half_tile_size
            < region.centroid[0]
            < image_shape[0] - half_tile_size - 1
        )
        and (
            half_tile_size
            < region.centroid[1]
            < image_shape[1] - half_tile_size - 1
        )
    ]
    if not valid_regions:
        raise ValueError("No valid tiles found.")
    centroids = (
        np.array([region.centroid for region in valid_regions])
        .round()
        .astype(int)
    )
    return centroids


def _create_mean_template(
    image: npt.NDArray[np.floating[Any]],
    centroids: npt.NDArray[np.int_],
    tile_size: int,
) -> npt.NDArray[np.floating[Any]]:
    """
    Create mean template from candidate trap regions.

    Parameters
    ----------
    image : 2D array
        The input image.
    centroids : array of shape (n, 2)
        Coordinates of trap centroids.
    tile_size : int
        Size of tile in pixels.

    Returns
    -------
    mean_template : 2D array
        Average of all candidate trap templates.
    """
    candidate_templates = [
        image[
            half_floor(x, tile_size) : half_ceil(x, tile_size),
            half_floor(y, tile_size) : half_ceil(y, tile_size),
        ]
        for x, y in centroids
    ]
    return np.stack(candidate_templates).mean(axis=0)


def _score_correlation(
    matched: npt.NDArray[np.floating[Any]],
    k: int = 30,
) -> float:
    """
    Score a correlation image by the mean of the top k values.

    Parameters
    ----------
    matched : 2D array
        The correlation image from template matching.
    k : int
        Number of top values to average (should approximate expected
        trap count).

    Returns
    -------
    score : float
        Mean of the top k correlation values.
    """
    flat = matched.ravel()
    # pick top correlations - faster than sorting
    top_k = np.partition(flat, -k)[-k:]
    return float(np.mean(top_k))


def _rotate_coordinates(
    coordinates: npt.NDArray[np.int_],
    angle: float,
) -> npt.NDArray[np.floating[Any]]:
    """
    Rotate coordinates by angle (radians) around centroid.

    Parameters
    ----------
    coordinates : array of shape (n, 2)
        The (row, col) coordinates.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    rotated : array of shape (n, 2)
        The rotated coordinates.
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    centroid = coordinates.mean(axis=0)
    centred = coordinates - centroid
    return centred @ rotation_matrix.T + centroid


def _score_horizontal_alignment(
    y_coords: npt.NDArray[np.floating[Any]],
    bin_width: float,
) -> float:
    """
    Score how well y-coordinates cluster into horizontal bands.

    Higher score means better alignment into horizontal lines.

    Parameters
    ----------
    y_coords : 1D array
        The y-coordinates after rotation.
    bin_width : float
        Width of histogram bins.

    Returns
    -------
    score : float
        Sum of squared bin counts (rewards concentration).
    """
    bins = np.arange(y_coords.min(), y_coords.max() + bin_width, bin_width)
    counts, _ = np.histogram(y_coords, bins=bins)
    return float(np.sum(counts**2))


def _find_optimal_rotation(
    coordinates: npt.NDArray[np.int_],
    tile_size: int,
    bin_width_frac: float = 0.15,
    angle_step: float = 2.0,
) -> float:
    """
    Find rotation angle that maximises horizontal line alignment.

    Parameters
    ----------
    coordinates : array of shape (n, 2)
        The (row, col) coordinates.
    tile_size : int
        Size of tile in pixels, used to compute bin width.
    bin_width_frac : float
        Bin width as a fraction of tile_size.
    angle_step : float
        Step size in degrees for angle search.

    Returns
    -------
    best_angle : float
        Optimal rotation angle in radians.
    """
    bin_width = tile_size * bin_width_frac
    angles = np.arange(0, 180, angle_step)
    best_score = -1.0
    best_angle = 0.0
    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)
        rotated = _rotate_coordinates(coordinates, angle_rad)
        score = _score_horizontal_alignment(rotated[:, 0], bin_width)
        if score > best_score:
            best_score = score
            best_angle = angle_rad
    return best_angle


def _filter_by_grid_regularity(
    coordinates: npt.NDArray[np.int_],
    tile_size: int,
    bin_width_frac: float = 0.15,
    min_points_per_line: int = 4,
) -> npt.NDArray[np.int_]:
    """
    Filter coordinates by finding optimal rotation.

    Keep points on well-populated horizontal lines.

    Parameters
    ----------
    coordinates : array of shape (n, 2)
        The (row, col) coordinates of detected traps.
    tile_size : int
        Size of tile in pixels, used to compute bin width.
    bin_width_frac : float
        Bin width as a fraction of tile_size.
    min_points_per_line : int
        Minimum points for a line to be considered valid.

    Returns
    -------
    filtered : array of shape (m, 2)
        The filtered coordinates (m <= n).
    """
    if len(coordinates) < min_points_per_line:
        return coordinates
    bin_width = tile_size * bin_width_frac
    # find optimal rotation
    angle = _find_optimal_rotation(coordinates, tile_size, bin_width_frac)
    rotated = _rotate_coordinates(coordinates, angle)
    # bin y-coordinates
    y_coords = rotated[:, 0]
    bins = np.arange(y_coords.min(), y_coords.max() + bin_width, bin_width)
    bin_indices = np.digitize(y_coords, bins)
    # count points per bin
    unique_bins, counts = np.unique(bin_indices, return_counts=True)
    valid_bins = unique_bins[counts >= min_points_per_line]
    # keep points in valid bins
    valid_mask = np.isin(bin_indices, valid_bins)
    return coordinates[valid_mask]


def _find_best_rotation(
    img: npt.NDArray[np.floating[Any]],
    template: npt.NDArray[np.floating[Any]],
    rotations: list[float],
) -> tuple[float, npt.NDArray[np.floating[Any]]]:
    """
    Find the rotation angle that best matches the template to the image.

    Parameters
    ----------
    img : 2D array
        The image to match against.
    template : 2D array
        The template to rotate and match.
    rotations : list of floats
        Rotation angles in degrees to try.

    Returns
    -------
    best_rotation : float
        The rotation angle with the best match.
    best_matched : 2D array
        The correlation image for the best rotation.
    """
    fill_value = np.median(img)
    rotation_matches = {
        rotation: np.abs(
            feature.match_template(
                img,
                transform.rotate(template, rotation, cval=fill_value),
                pad_input=True,
                mode="median",
            )
        )
        for rotation in rotations
    }
    best_rotation = max(
        rotation_matches,
        key=lambda x: _score_correlation(rotation_matches[x]),
    )
    return best_rotation, rotation_matches[best_rotation]


def _find_best_scale(
    img: npt.NDArray[np.floating[Any]],
    template: npt.NDArray[np.floating[Any]],
    scales: npt.NDArray[np.floating[Any]],
) -> tuple[float, npt.NDArray[np.floating[Any]]]:
    """
    Find the scale factor that best matches the template to the image.

    Parameters
    ----------
    img : 2D array
        The image to match against.
    template : 2D array
        The template to scale and match.
    scales : 1D array
        Scale factors to try.

    Returns
    -------
    best_scale : float
        The scale factor with the best match.
    best_matched : 2D array
        The correlation image for the best scale.
    """
    scale_matches = {
        scale: np.abs(
            feature.match_template(
                img,
                transform.rescale(template, scale),
                pad_input=True,
                mode="median",
            )
        )
        for scale in scales
    }
    best_scale = max(
        scale_matches,
        key=lambda x: _score_correlation(scale_matches[x]),
    )
    return best_scale, scale_matches[best_scale]


def plot_trap_locations(
    image: Union[npt.NDArray[np.floating[Any]], Any],
    trap_coordinates: npt.NDArray[np.int_],
    tile_size: int,
    figsize: tuple[float, float] = (9, 9),
    marker_size: float | None = None,
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
        tile_size / 7.

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
    plt.waitforbuttonpress()
    return fig, ax
