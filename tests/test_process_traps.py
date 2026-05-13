"""
Unit and integration tests for aliby.tile.process_traps.

Covers all helper functions in isolation with small synthetic arrays,
tests the retry orchestration of segment_traps via mocks, and runs one
slow integration test of identify_trap_locations on a synthetic ring-grid image.
"""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from skimage import draw, measure

from aliby.tile.process_traps import (
    _compute_entropy_image,
    _create_mean_template,
    _filter_by_grid_regularity,
    _find_optimal_rotation,
    _find_valid_trap_regions,
    _rotate_coordinates,
    _score_correlation,
    _score_horizontal_alignment,
    _segment_entropy_image,
    correct_illumination,
    half_ceil,
    half_floor,
    identify_trap_locations,
    segment_traps,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_label_image(shape, circles):
    """Create a labelled image containing circular regions.

    Parameters
    ----------
    shape: tuple
        Image dimensions (rows, cols).
    circles: list of (row, col, radius)
        Each tuple defines a filled disk.
    """
    binary = np.zeros(shape, dtype=bool)
    for r, c, rad in circles:
        rr, cc = draw.disk((r, c), rad, shape=shape)
        binary[rr, cc] = True
    return measure.label(binary)


def _make_ring_image(n_rows, n_cols, tile_size, ring_radius):
    """Create a synthetic image with bright rings on a black background.

    Parameters
    ----------
    n_rows: int
    n_cols: int
    tile_size: int
    ring_radius: float
        Radius of each ring in pixels.

    Returns
    -------
    image: 2D float32 array
    centres: array of shape (n_rows*n_cols, 2)
        Row, col coordinates of ring centres.
    """
    H = (n_rows + 1) * tile_size
    W = (n_cols + 1) * tile_size
    image = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.ogrid[:H, :W]
    centres = []
    for i in range(n_rows):
        for j in range(n_cols):
            cy = (i + 1) * tile_size
            cx = (j + 1) * tile_size
            centres.append((cy, cx))
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            image[np.abs(dist - ring_radius) < 2] = 1.0
    return image, np.array(centres)


# ---------------------------------------------------------------------------
# half_floor / half_ceil
# ---------------------------------------------------------------------------


def test_half_floor_even_tile():
    assert half_floor(64, 32) == 48


def test_half_floor_odd_tile():
    # 33 // 2 = 16, so 64 - 16 = 48
    assert half_floor(64, 33) == 48


def test_half_ceil_even_tile():
    # -(32 // -2) = 16, so 64 + 16 = 80
    assert half_ceil(64, 32) == 80


def test_half_ceil_odd_tile():
    # -(33 // -2) = 17, so 64 + 17 = 81
    assert half_ceil(64, 33) == 81


@pytest.mark.parametrize("tile_size", [32, 33, 64, 117])
def test_tile_extent_equals_tile_size(tile_size):
    """half_ceil - half_floor is always exactly tile_size."""
    x = 200
    assert half_ceil(x, tile_size) - half_floor(x, tile_size) == tile_size


# ---------------------------------------------------------------------------
# _score_correlation
# ---------------------------------------------------------------------------


def test_score_correlation_top_k_mean():
    # top 2 of [1,2,3,4,5] are 4 and 5 → mean 4.5
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _score_correlation(arr, k=2) == pytest.approx(4.5)


def test_score_correlation_all_zeros():
    assert _score_correlation(np.zeros(50), k=5) == 0.0


def test_score_correlation_2d_array():
    # ravel: values 0-11; top 2 are 10 and 11 → mean 10.5
    arr = np.arange(12, dtype=float).reshape(3, 4)
    assert _score_correlation(arr, k=2) == pytest.approx(10.5)


# ---------------------------------------------------------------------------
# _rotate_coordinates
# ---------------------------------------------------------------------------


def test_rotate_coordinates_zero_angle():
    coords = np.array([[10.0, 20.0], [30.0, 40.0]])
    result = _rotate_coordinates(coords, 0.0)
    assert_array_almost_equal(result, coords)


def test_rotate_coordinates_pi_negates_relative_positions():
    # symmetric pair about origin; π rotation swaps them
    coords = np.array([[1.0, 0.0], [-1.0, 0.0]])
    result = _rotate_coordinates(coords, np.pi)
    assert_array_almost_equal(result, [[-1.0, 0.0], [1.0, 0.0]], decimal=10)


def test_rotate_coordinates_preserves_distance_from_centroid():
    coords = np.array([[3.0, 4.0], [-3.0, -4.0]])
    rotated = _rotate_coordinates(coords, 1.234)
    centroid = coords.mean(axis=0)
    d_orig = np.linalg.norm(coords - centroid, axis=1)
    d_rot = np.linalg.norm(rotated - rotated.mean(axis=0), axis=1)
    assert_array_almost_equal(d_orig, d_rot)


# ---------------------------------------------------------------------------
# _score_horizontal_alignment
# ---------------------------------------------------------------------------


def test_score_horizontal_alignment_all_same_y():
    # all 4 points within one bin_width=2 → one bin → 4² = 16
    y = np.array([0.0, 0.5, 0.25, 0.75])
    assert _score_horizontal_alignment(y, bin_width=2.0) == 16.0


def test_score_horizontal_alignment_spread():
    # each point in its own bin → 4 × 1² = 4
    y = np.array([0.0, 5.0, 10.0, 15.0])
    assert _score_horizontal_alignment(y, bin_width=1.0) == 4.0


def test_score_horizontal_alignment_clustered_beats_spread():
    y_clust = np.array([0.0, 0.1, 0.0, 0.1])
    y_spread = np.array([0.0, 2.0, 4.0, 6.0])
    assert (
        _score_horizontal_alignment(y_clust, bin_width=1.0)
        > _score_horizontal_alignment(y_spread, bin_width=1.0)
    )


# ---------------------------------------------------------------------------
# _find_optimal_rotation
# ---------------------------------------------------------------------------


def test_find_optimal_rotation_horizontal_grid():
    """Perfectly horizontal grid → angle within 10° of 0 or π."""
    coords = np.array(
        [[r * 60.0, c * 60.0] for r in range(4) for c in range(6)]
    )
    angle = _find_optimal_rotation(coords, tile_size=60)
    assert min(abs(angle), abs(angle - np.pi)) < np.radians(10)


# ---------------------------------------------------------------------------
# _filter_by_grid_regularity
# ---------------------------------------------------------------------------


def test_filter_by_grid_too_few_points_unchanged():
    """Fewer than min_points_per_line → all returned as-is."""
    coords = np.array([[0, 0], [1, 0], [2, 0]])
    result = _filter_by_grid_regularity(coords, tile_size=60)
    assert_array_equal(result, coords)


def test_filter_by_grid_keeps_regular_grid():
    """4×5 grid (5 pts/row ≥ 4) → all 20 points kept."""
    coords = np.array(
        [[r * 60.0, c * 60.0] for r in range(4) for c in range(5)]
    )
    result = _filter_by_grid_regularity(
        coords, tile_size=60, min_points_per_line=4
    )
    assert len(result) == len(coords)


def test_filter_by_grid_drops_outlier():
    """Point far from any populated horizontal line should be filtered."""
    grid = np.array(
        [[r * 60.0, c * 60.0] for r in range(4) for c in range(4)]
    )
    outlier = np.array([[300.0, 300.0]])
    coords = np.vstack([grid, outlier])
    result = _filter_by_grid_regularity(coords, tile_size=60)
    assert len(result) < len(coords)


# ---------------------------------------------------------------------------
# _find_valid_trap_regions
# ---------------------------------------------------------------------------

# tile_size=60 → half_tile=30, threshold for valid axis: (18, 60)


def test_find_valid_trap_regions_valid_circle():
    label_img = _make_label_image((200, 200), [(100, 100, 15)])
    # major_axis ≈ 30: 18 < 30 < 60 ✓; centroid (100,100): 30 < 100 < 169 ✓
    centroids = _find_valid_trap_regions(label_img, (200, 200), 60, 18)
    assert len(centroids) == 1
    assert_array_almost_equal(centroids[0], [100, 100], decimal=0)


def test_find_valid_trap_regions_too_small():
    label_img = _make_label_image((200, 200), [(100, 100, 3)])
    # major_axis ≈ 6 < min_trap_size 18 → filtered
    with pytest.raises(ValueError, match="No valid tiles"):
        _find_valid_trap_regions(label_img, (200, 200), 60, 18)


def test_find_valid_trap_regions_too_large():
    label_img = _make_label_image((200, 200), [(100, 100, 35)])
    # major_axis ≈ 70 > tile_size 60 → filtered
    with pytest.raises(ValueError, match="No valid tiles"):
        _find_valid_trap_regions(label_img, (200, 200), 60, 18)


def test_find_valid_trap_regions_border():
    # centroid at row 20 < half_tile 30 → filtered by border check
    label_img = _make_label_image((200, 200), [(20, 100, 10)])
    with pytest.raises(ValueError, match="No valid tiles"):
        _find_valid_trap_regions(label_img, (200, 200), 60, 5)


def test_find_valid_trap_regions_empty():
    label_img = np.zeros((200, 200), dtype=int)
    with pytest.raises(ValueError, match="No valid tiles"):
        _find_valid_trap_regions(label_img, (200, 200), 60, 18)


# ---------------------------------------------------------------------------
# _create_mean_template
# ---------------------------------------------------------------------------


def test_create_mean_template_shape():
    image = np.zeros((200, 200), dtype=np.float32)
    centroids = np.array([[100, 100], [100, 150]])
    template = _create_mean_template(image, centroids, tile_size=32)
    assert template.shape == (32, 32)


def test_create_mean_template_uniform_image():
    image = np.ones((200, 200), dtype=np.float32)
    centroids = np.array([[100, 100]])
    template = _create_mean_template(image, centroids, tile_size=32)
    assert_array_equal(template, np.ones((32, 32)))


def test_create_mean_template_averages_two_regions():
    image = np.zeros((200, 200), dtype=np.float32)
    # second centroid's region is all ones
    # half_floor(100,32)=84, half_ceil(100,32)=116
    # half_floor(150,32)=134, half_ceil(150,32)=166
    image[84:116, 134:166] = 1.0
    centroids = np.array([[100, 100], [100, 150]])
    template = _create_mean_template(image, centroids, tile_size=32)
    np.testing.assert_allclose(template, np.full((32, 32), 0.5))


# ---------------------------------------------------------------------------
# correct_illumination
# ---------------------------------------------------------------------------


def test_correct_illumination_shape():
    image = np.random.default_rng(0).random((100, 100)).astype(np.float32)
    result = correct_illumination(image, sigma=10.0)
    assert result.shape == image.shape


def test_correct_illumination_output_range():
    image = np.random.default_rng(0).random((100, 100)).astype(np.float32)
    result = correct_illumination(image, sigma=10.0)
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-5


def test_correct_illumination_uniform_input():
    # uniform image: background = image → corrected = constant → normalised to 1
    image = np.full((50, 50), 0.5, dtype=np.float32)
    result = correct_illumination(image, sigma=5.0)
    np.testing.assert_allclose(result, np.ones((50, 50)), atol=1e-4)


# ---------------------------------------------------------------------------
# _compute_entropy_image
# ---------------------------------------------------------------------------


def test_compute_entropy_image_shape_no_downscale():
    image = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    result = _compute_entropy_image(image, downscale=1, disk_radius_frac=0.05)
    assert result.shape == image.shape


def test_compute_entropy_image_shape_with_downscale():
    image = np.random.default_rng(0).random((100, 100)).astype(np.float32)
    result = _compute_entropy_image(image, downscale=0.5, disk_radius_frac=0.1)
    # rounding in rescale → allow ±2 px tolerance
    assert abs(result.shape[0] - image.shape[0]) <= 2
    assert abs(result.shape[1] - image.shape[1]) <= 2


# ---------------------------------------------------------------------------
# _segment_entropy_image
# ---------------------------------------------------------------------------


def test_segment_entropy_image_bimodal():
    """Centre block of high entropy, surrounded by zero, not touching border."""
    entropy_img = np.zeros((100, 100), dtype=np.float32)
    entropy_img[30:70, 30:70] = 1.0
    label_img = _segment_entropy_image(entropy_img, square_size=3)
    assert label_img.max() >= 1


def test_segment_entropy_image_returns_integer_array():
    entropy_img = np.zeros((100, 100), dtype=np.float32)
    label_img = _segment_entropy_image(entropy_img, square_size=3)
    assert np.issubdtype(label_img.dtype, np.integer)


# ---------------------------------------------------------------------------
# segment_traps retry orchestration (mock-based)
# ---------------------------------------------------------------------------


def test_segment_traps_retries_when_few_traps():
    """When < 30 traps found, segment_traps retries with downscale=1."""
    image = np.zeros((200, 200), dtype=np.float32)
    few_coords = np.zeros((5, 2), dtype=int)
    many_coords = np.zeros((35, 2), dtype=int)
    mods = "aliby.tile.process_traps"
    patches_cfg = [
        (f"{mods}.correct_illumination", {"return_value": image}),
        (f"{mods}._compute_entropy_image", {"return_value": image}),
        (
            f"{mods}._segment_entropy_image",
            {"return_value": np.zeros((200, 200), dtype=int)},
        ),
        (
            f"{mods}._find_valid_trap_regions",
            {"return_value": np.array([[100, 100]])},
        ),
        (
            f"{mods}._create_mean_template",
            {"return_value": np.zeros((117, 117))},
        ),
    ]
    with ExitStack() as stack:
        for target, kwargs in patches_cfg:
            stack.enter_context(patch(target, **kwargs))
        mock_itl = stack.enter_context(
            patch(
                f"{mods}.identify_trap_locations",
                side_effect=[few_coords, many_coords],
            )
        )
        result = segment_traps(image, 117)
    assert mock_itl.call_count == 2
    assert len(result) == len(many_coords)


# ---------------------------------------------------------------------------
# integration: identify_trap_locations on a synthetic ring-grid image
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_identify_trap_locations_ring_grid():
    """Template matching locates rings at known grid positions."""
    tile_size = 60
    n_rows, n_cols = 5, 8  # 40 traps > retry threshold of 30
    ring_radius = 18
    image, expected_centres = _make_ring_image(
        n_rows, n_cols, tile_size, ring_radius
    )
    # use the first trap's crop as the template
    cy, cx = expected_centres[0]
    template = image[
        half_floor(cy, tile_size) : half_ceil(cy, tile_size),
        half_floor(cx, tile_size) : half_ceil(cx, tile_size),
    ]
    result = identify_trap_locations(
        image, template, optimize_scale=False
    )
    # at least 80 % of expected traps found
    assert len(result) >= int(0.8 * len(expected_centres))
    # every expected centre has a detected centre within half a tile
    half_tile = tile_size // 2
    for ey, ex in expected_centres:
        dists = np.sqrt(
            (result[:, 0] - ey) ** 2 + (result[:, 1] - ex) ** 2
        )
        assert dists.min() < half_tile, (
            f"no detection near expected centre ({ey}, {ex})"
        )
