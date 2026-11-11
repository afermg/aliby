"""
Check where extraction says a cell sits in the image.

A cell's centroid is measured in its tile, and the tile moves, so turning
one into an image coordinate means adding the tile's origin. That origin is
sooth's, the definition of record: it is where the tile was actually cut.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sooth

from extraction.core.extractor import (
    Extractor,
    ExtractorParameters,
    build_extraction_tree_from_meta,
)
from extraction.core.functions import cell_functions
from extraction.core.functions.loaders import (
    load_all_functions,
    load_reduction_functions,
)
from extraction.core.functions.math_utils import div0

CENTROIDS_YX = [(10.5, 3.25), (30.0, 12.0)]
CENTRES_YX = [[100, 60], [140, 200]]


class FakeTileLocs:
    """Stand in for TileLocations, holding centres that do not drift."""

    def __init__(self, centres_yx):
        self.centres = np.asarray(centres_yx, dtype=float)

    def centres_at_time(self, tp):
        """Return every trap's centre at a time point."""
        return self.centres


def extractor_with(tile_size):
    """Return something with just the attributes the method reads."""
    return SimpleNamespace(
        tiler=SimpleNamespace(
            tile_size=tile_size,
            tile_locs=FakeTileLocs(CENTRES_YX),
            spatial_location=None,
        )
    )


def extract_dict(tp=0):
    """Return the centroid frames extraction builds, one cell per trap."""
    index = pd.MultiIndex.from_arrays(
        [[0, 1], [1, 1]], names=["trap", "cell_label"]
    )
    return {
        "general/null/centroid_x": pd.DataFrame(
            {tp: [centroid[1] for centroid in CENTROIDS_YX]}, index=index
        ),
        "general/null/centroid_y": pd.DataFrame(
            {tp: [centroid[0] for centroid in CENTROIDS_YX]}, index=index
        ),
    }


@pytest.mark.parametrize("tile_size", [117, 96, (48, 16)])
def test_a_cell_is_placed_where_its_tile_was_cut(tile_size):
    """
    Check the image coordinate is the tile's origin plus the centroid.

    The origin is ``centre - size // 2`` on each axis. Extraction used the
    tile's *geometric* centre, ``(size - 1) / 2``, which is half a pixel
    away from it for an even size -- so every cell of an even-sized tile
    was placed half a pixel down and to the right of where its tile sat.
    Every aliby run so far has used 117, which is odd, and for an odd size
    the two agree exactly.
    """
    result = extract_dict()
    Extractor.add_spatial_locations_of_cells(
        extractor_with(tile_size), result
    )

    for row, (centroid, centre) in enumerate(zip(CENTROIDS_YX, CENTRES_YX)):
        wanted_y, wanted_x = sooth.tile_to_image_yx(
            np.asarray(centroid), centre, tile_size
        )
        assert result["general/null/image_x"][0].iloc[row] == wanted_x
        assert result["general/null/image_y"][0].iloc[row] == wanted_y


def test_a_position_whose_only_trap_is_trap_zero_is_placed_too():
    """
    Check a single trap named zero still gets image coordinates.

    A trap is a name, not a count. Asking ``np.any(traps)`` asks whether any
    trap is named something other than zero, so a position holding one trap
    -- or a per-trap crop, which holds exactly one -- had its cells left at
    their tile-local centroids, silently. A normal run has traps 0..N, so
    some name is non-zero and the question happened to answer correctly.
    """
    index = pd.MultiIndex.from_arrays(
        [[0], [1]], names=["trap", "cell_label"]
    )
    result = {
        "general/null/centroid_x": pd.DataFrame({0: [3.25]}, index=index),
        "general/null/centroid_y": pd.DataFrame({0: [10.5]}, index=index),
    }
    Extractor.add_spatial_locations_of_cells(extractor_with(117), result)

    wanted_y, wanted_x = sooth.tile_to_image_yx(
        np.asarray(CENTROIDS_YX[0]), CENTRES_YX[0], 117
    )
    assert result["general/null/image_x"][0].iloc[0] == wanted_x
    assert result["general/null/image_y"][0].iloc[0] == wanted_y


class FakeTile:
    """Stand in for a Tile, cut at a fixed place."""

    def __init__(self, slices):
        self.slices = slices

    def as_range(self, tp):
        """Return the tile's slices, which do not drift."""
        return self.slices


def square(top, left, side=6, shape=(20, 20)):
    """Return a square cell mask."""
    mask = np.zeros(shape, dtype=bool)
    mask[top : top + side, left : left + side] = True
    return mask


def extractor_for_obscured(slices, image_yx=(100, 100), exclude=True):
    """Return something with what find_obscured and the functions read."""
    return SimpleNamespace(
        params=SimpleNamespace(exclude_obscured=exclude),
        tiler=SimpleNamespace(
            shape=(1, 1, 1, *image_yx),
            tile_size=20,
            tile_locs=SimpleNamespace(tiles=[FakeTile(slices)]),
        ),
        obscured={},
        pdms_mask=None,
        cell_fun_names={"area"},
        all_funs={
            "area": lambda masks, trap, channels: masks.sum(axis=(1, 2)),
            "background_area": lambda masks, trap, channels, exclude_mask: (
                ~masks.any(axis=0)
            ).sum(),
        },
    )


def test_a_cell_cut_off_by_its_tile_is_obscured():
    """
    Check a cell touching the tile border is set aside.

    Its area is that of the part inside the tile, not of the cell.
    """
    masks = [np.stack([square(7, 7), square(0, 7), square(7, 14)])]
    labels = {0: [1, 2, 3]}
    extractor = extractor_for_obscured((slice(40, 60), slice(40, 60)))

    obscured = Extractor.find_obscured(extractor, 0, masks, labels)

    assert obscured == {0: {2, 3}}


def test_a_cell_over_a_tile_padded_beyond_the_image_is_obscured():
    """
    Check a cell drawn over padding is set aside.

    The tile hangs three columns off the image's right edge, where tiler
    repeats the last column, so a cell ending two columns from the tile's
    border is drawn over pixels that were never imaged.
    """
    masks = [np.stack([square(7, 7), square(7, 12)])]
    labels = {0: [1, 2]}
    extractor = extractor_for_obscured((slice(40, 60), slice(83, 103)))

    obscured = Extractor.find_obscured(extractor, 0, masks, labels)

    assert obscured == {0: {2}}


def test_obscured_cells_are_kept_when_asked():
    masks = [np.stack([square(0, 7)])]
    extractor = extractor_for_obscured(
        (slice(40, 60), slice(40, 60)), exclude=False
    )

    assert Extractor.find_obscured(extractor, 0, masks, {0: [1]}) == {}


def test_an_obscured_cell_is_not_measured_but_is_not_background():
    """
    Check an obscured cell is skipped by cell functions alone.

    Its pixels are still a cell's, so a background estimate that counted
    them would be contaminated by its fluorescence.
    """
    masks = [np.stack([square(7, 7), square(0, 7)])]
    extractor = extractor_for_obscured((slice(40, 60), slice(40, 60)))
    extractor.obscured = {0: {2}}
    tile = np.zeros((20, 20))

    areas, cells = Extractor.apply_extraction_function(
        extractor, [tile], masks, "area", {0: [1, 2]}, ["GFP"]
    )
    (background,), _ = Extractor.apply_extraction_function(
        extractor, [tile], masks, "background_area", {0: [1, 2]}, ["GFP"]
    )

    assert cells == ((0, 1),)
    assert areas == (36,)
    assert background == 400 - 2 * 36


def two_cells(size=40):
    """Return the masks of two cells in one tile, as (2, size, size)."""
    masks = np.zeros((2, size, size), dtype=bool)
    masks[0, 5:15, 5:15] = True
    masks[1, 22:34, 22:34] = True
    return masks


def test_div0_divides_the_first_by_the_second():
    """Divide one slice by the other, with fill where there is no answer."""
    # both slices were taken from one list, so the second was divided by
    # itself and every answer was one
    array = np.stack(
        (np.full((3, 3), 6.0), np.array([[2.0, 2.0, 0.0]] * 3)), axis=-1
    )
    assert div0(array).tolist() == [[3.0, 3.0, 0.0]] * 3


def test_moment_of_inertia_leaves_the_image_as_it_was():
    """Measure every cell of a tile, and leave the tile to be measured."""
    # the pixels outside the cell were set to zero in the tile itself: the
    # second cell of a tile got NaN, and every function after it zeros
    image = np.random.default_rng(0).random((40, 40)) + 1
    before = image.copy()
    _cell_funs, all_funs = load_all_functions()
    found = all_funs["moment_of_inertia"](two_cells(), image, ["GFP"])
    assert not np.isnan(found).any()
    assert (image == before).all()


def test_total_squared_is_right_for_an_image_of_integers():
    """Square pixels as floats."""
    # 1000 squared is more than a uint16 holds
    image = np.full((40, 40), 1000, dtype=np.uint16)
    mask = two_cells()[0]
    assert cell_functions.total_squared(mask, image) == mask.sum() * 1e6


def test_a_ratio_divides_one_channel_by_the_other():
    """Find the ratio of two channels, whatever they are called."""
    # the channels were looked for as the second and third of two, and
    # one had to be called mCherry
    image = np.stack((np.full((40, 40), 6.0), np.full((40, 40), 2.0)), -1)
    mask = two_cells()[0]
    channels = ["GFP", "Flavin"]
    assert cell_functions.ratio_1_over_2(mask, image, channels) == 3
    assert cell_functions.ratio_2_over_1(mask, image, channels) == 1 / 3
    image[7, 7, 1] = 0
    assert np.isnan(cell_functions.ratio_1_over_2(mask, image, channels))
    assert cell_functions.ratio_2_over_1(mask, image, channels) == 1 / 3


def test_no_membrane_is_looked_for_in_cy5():
    """Know a channel by its name when the channels come as a list."""
    # a list of channels is never in a list of names, so cy5 was fitted
    image = np.random.default_rng(0).random((40, 40))
    found = cell_functions.membrane_fluorescence(
        two_cells()[1], image, ["cy5"]
    )
    assert np.isnan(list(found.values())).all()


def test_outlines_read_the_masks_once():
    """Read a time point's edge masks from the h5 file one time."""
    # they were read once for every trap of the position

    class FakeCells:
        ntraps = 3
        reads = 0

        def at_time(self, tp, kind):
            self.reads += 1
            return {0: [two_cells()[0]], 1: [], 2: list(two_cells())}

    cells = FakeCells()
    labels = {0: [4], 1: [], 2: [1, 7]}
    outlines = Extractor.get_outlines(
        SimpleNamespace(get_cell_labels=lambda tp, cell_labels, cells: labels),
        0,
        labels,
        cells,
    )
    assert cells.reads == 1
    assert np.unique(outlines[0]).tolist() == [0, 4]
    assert outlines[1].size == 0
    assert np.unique(outlines[2]).tolist() == [0, 1, 7]


def test_a_channel_with_no_background_to_subtract_has_no_bgsub():
    """Extract _bgsub only for channels whose background is subtracted."""
    # it was extracted from an image that was None, which raised
    asked = []

    def reduce_extract(tiles, **kwargs):
        asked.append(tiles)
        return {}

    fake = SimpleNamespace(
        params=SimpleNamespace(subtract_background={"GFP"}),
        reduce_extract=reduce_extract,
    )
    img = {"GFP": "gfp", "mCherry": "mcherry"}
    img_bgsub = {"GFP_bgsub": "gfp_bgsub", "mCherry_bgsub": None}
    for channel in ("GFP", "mCherry"):
        found = Extractor._extract_channel(
            fake, channel, {}, img, img_bgsub, [], {}, set()
        )
        assert (channel + "_bgsub" in found) == (channel == "GFP")
    assert asked == ["gfp", "gfp_bgsub", "mcherry"]


def disc(size, centre_y, centre_x, radius):
    """Return the mask of a round cell in a square tile."""
    y, x = np.ogrid[:size, :size]
    return (y - centre_y) ** 2 + (x - centre_x) ** 2 <= radius**2


@pytest.mark.parametrize(
    "mask",
    [
        disc(117, 30, 30, 9),
        disc(117, 60, 70, 12.5),
        # against the tile's edge
        disc(117, 3, 110, 8),
        disc(117, 40, 40, 9) | disc(117, 40, 55, 6),
    ],
)
def test_a_cell_measures_the_same_wherever_its_tile_ends(mask):
    """Measure a cell in its own box as in its whole tile."""
    # the functions measure within the cell's box, which is quicker by
    # eight times for a tile of 117 pixels
    rows, columns = np.nonzero(mask)
    box = mask[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    in_larger_tile = np.pad(mask, 40)
    for name in ("volume", "ellipsoidal_volume", "eccentricity"):
        function = getattr(cell_functions, name)
        assert function(mask) == pytest.approx(function(box), rel=1e-12)
        assert function(mask) == pytest.approx(
            function(in_larger_tile), rel=1e-12
        )


@pytest.mark.parametrize(
    "shape_yx, wanted",
    [((1, 1), (1, 1)), ((1, 6), (1, 3)), ((2, 6), (1, 6)), ((2, 2), (1, 2))],
)
def test_a_cell_with_no_interior_has_a_major_axis_of_half_its_area(
    shape_yx, wanted
):
    """Give a cell that is all edge axes that are its own."""
    # its major axis came from a distance transform of an array with no
    # zeros, which is not defined: 8 for a cell of one pixel, and another
    # number for the same cell elsewhere in its tile
    for corner in (10, 60):
        mask = np.zeros((117, 117), dtype=bool)
        mask[corner : corner + shape_yx[0], corner : corner + shape_yx[1]] = 1
        assert cell_functions.min_maj_approximation(mask) == wanted
        assert not np.isnan(cell_functions.eccentricity(mask))


def test_an_empty_mask_has_no_volume():
    """Measure a mask that holds no cell without an error."""
    mask = np.zeros((117, 117), dtype=bool)
    assert cell_functions.volume(mask) == 0
    assert cell_functions.ellipsoidal_volume(mask) == 0


def test_a_function_of_two_channels_is_given_both():
    """Extract a ratio from the images, and from those less background."""
    _cell_funs, all_funs = load_all_functions()
    fake = SimpleNamespace(
        params=SimpleNamespace(
            multichannel_funs={
                "ratio": [["GFP", "mCherry"], "max", "ratio_1_over_2"]
            }
        ),
        cell_fun_names=_cell_funs,
        all_funs=all_funs,
        obscured={},
    )
    fake.apply_extraction_function = (
        lambda *args: Extractor.apply_extraction_function(fake, *args)
    )
    # one tile of three z-sections, as (tiles, z, y, x)
    img = {
        "GFP": np.full((1, 3, 40, 40), 6.0),
        "mCherry": np.full((1, 3, 40, 40), 2.0),
    }
    img_bgsub = {
        "GFP_bgsub": np.full((1, 3, 40, 40), 4.0),
        "mCherry_bgsub": np.full((1, 3, 40, 40), 1.0),
    }
    masks = [two_cells()]
    labels = {0: [1, 2]}
    found = Extractor.extract_multichannel_functions(
        fake, labels, img, img_bgsub, masks
    )["ratio"]["max"]
    assert found["ratio_1_over_2"] == ((3.0, 3.0), ((0, 1), (0, 2)))
    assert found["ratio_1_over_2_bgsub"] == ((4.0, 4.0), ((0, 1), (0, 2)))
    # a channel whose background is not subtracted has no such image
    img_bgsub["mCherry_bgsub"] = None
    found = Extractor.extract_multichannel_functions(
        fake, labels, img, img_bgsub, masks
    )["ratio"]["max"]
    assert list(found) == ["ratio_1_over_2"]


def test_a_function_of_brightfield_has_no_bgsub():
    """Extract nothing less background where brightfield is a channel."""
    # brightfield was left out of the images and kept among the names,
    # so the function had one image for two channels and gave only NaN
    cell_funs, all_funs = load_all_functions()
    fake = SimpleNamespace(
        params=SimpleNamespace(
            multichannel_funs={
                "ratio": [["GFP", "Brightfield"], "max", "ratio_1_over_2"]
            }
        ),
        cell_fun_names=cell_funs,
        all_funs=all_funs,
        obscured={},
    )
    fake.apply_extraction_function = (
        lambda *args: Extractor.apply_extraction_function(fake, *args)
    )
    img = {
        "GFP": np.full((1, 3, 40, 40), 6.0),
        "Brightfield": np.full((1, 3, 40, 40), 2.0),
    }
    img_bgsub = {"GFP_bgsub": np.full((1, 3, 40, 40), 4.0)}
    found = Extractor.extract_multichannel_functions(
        fake, {0: [1, 2]}, img, img_bgsub, [two_cells()]
    )["ratio"]["max"]
    assert found == {"ratio_1_over_2": ((3.0, 3.0), ((0, 1), (0, 2)))}


def test_a_function_of_a_channel_not_extracted_is_warned_of(tmp_path):
    """Say once that a multichannel function cannot be extracted."""
    # it was passed over at every time point and nothing said so
    warnings = []
    handler = logging.Handler()
    handler.emit = lambda record: warnings.append(record.getMessage())
    logger = logging.getLogger("aliby")
    logger.addHandler(handler)
    try:
        extractor = Extractor(
            ExtractorParameters(
                tree={"general": {"null": ["area"]}, "GFP": {"max": ["mean"]}},
                multichannel_funs={
                    "kept": [["GFP", "Brightfield"], "max", "ratio_1_over_2"],
                    "lost": [["GFP", "mCherry"], "max", "ratio_1_over_2"],
                },
                identify_vacuoles=False,
            ),
            store=tmp_path / "none.h5",
            tiler=SimpleNamespace(channels=["Brightfield", "GFP"]),
        )
    finally:
        logger.removeHandler(handler)
    assert list(extractor.params.multichannel_funs) == ["kept"]
    assert len(warnings) == 1
    assert "lost" in warnings[0] and "mCherry" in warnings[0]


def test_a_centroid_counts_pixels_from_zero():
    """Find a cell's centroid at the indices of its pixels."""
    # pixels were counted from one, so every cell was placed a pixel down
    # and to the right of where it is, in its tile and in the image
    corner = np.zeros((117, 117), dtype=bool)
    corner[0, 0] = True
    assert cell_functions.centroid(corner) == (0, 0)
    mask = np.zeros((117, 117), dtype=bool)
    mask[3:6, 7:10] = True
    assert cell_functions.centroid_x(mask) == 8
    assert cell_functions.centroid_y(mask) == 4
    # as sooth finds it
    round_cell = disc(117, 60, 70, 12.5)
    rows, columns = np.nonzero(round_cell)
    assert cell_functions.centroid(round_cell) == pytest.approx(
        (columns.mean(), rows.mean())
    )


def test_a_cell_is_placed_at_its_own_pixel_of_the_image():
    """Place a cell of one pixel at that pixel's indices in the image."""
    tile_size = 117
    centre_yx = CENTRES_YX[0]
    origin_y, origin_x = sooth.tile_origin_yx(centre_yx, tile_size)
    mask = np.zeros((tile_size, tile_size), dtype=bool)
    mask[20, 31] = True
    index = pd.MultiIndex.from_arrays(
        [[0], [1]], names=["trap", "cell_label"]
    )
    result = {
        "general/null/centroid_x": pd.DataFrame(
            {0: [cell_functions.centroid_x(mask)]}, index=index
        ),
        "general/null/centroid_y": pd.DataFrame(
            {0: [cell_functions.centroid_y(mask)]}, index=index
        ),
    }
    Extractor.add_spatial_locations_of_cells(
        extractor_with(tile_size), result
    )
    assert result["general/null/image_x"][0].iloc[0] == origin_x + 31
    assert result["general/null/image_y"][0].iloc[0] == origin_y + 20



###
# what is extracted, and what is asked of the images
###


def warnings_from(action):
    """Return the messages the aliby logger is given while action runs."""
    messages = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    logger = logging.getLogger("aliby")
    logger.addHandler(handler)
    try:
        action()
    finally:
        logger.removeHandler(handler)
    return messages


def test_a_channel_that_is_not_extracted_is_warned_of():
    """Say which channels are left out for want of a name in the list."""
    # YFP and CFP were passed over and nothing said so
    channels = ["Brightfield", "YFP", "GFP", "CFP"]
    messages = warnings_from(
        lambda: build_extraction_tree_from_meta({"channels": channels})
    )
    assert len(messages) == 1
    assert "YFP, CFP" in messages[0] and "GFP," not in messages[0]
    assert not warnings_from(
        lambda: build_extraction_tree_from_meta(
            {"channels": ["Brightfield", "GFP", "cy5"]}
        )
    )


def test_cells_are_not_placed_without_their_centroids():
    """Extract from a tree with no centroids, placing no cell."""
    # the centroids were looked up whether or not they were asked for
    result = {"general/null/area": pd.DataFrame({0: [50]})}
    Extractor.add_spatial_locations_of_cells(extractor_with(117), result)
    assert list(result) == ["general/null/area"]


def real_extractor(obscured=None):
    """Return something that applies the real functions to cells."""
    cell_funs, all_funs = load_all_functions()
    fake = SimpleNamespace(
        cell_fun_names=cell_funs,
        all_funs=all_funs,
        obscured=obscured or {},
        pdms_mask=None,
        dict_fun_keys={},
        reduced_tiles={},
    )
    for method in (
        "apply_extraction_function",
        "apply_extraction_functions",
        "apply_pixel_functions",
    ):
        setattr(
            fake,
            method,
            lambda *args, method=method, **kwargs: getattr(Extractor, method)(
                fake, *args, **kwargs
            ),
        )
    return fake


def test_a_function_of_several_values_is_named_alike_with_no_cells():
    """Name a dict's values the same at a time point with no cells."""
    # with no cells there was one empty entry under the function's name
    fake = real_extractor()
    tiles = np.random.default_rng(0).random((1, 40, 40)) * 100
    funs = ["membrane_fluorescence", "mean"]
    with_cells = fake.apply_extraction_functions(
        tiles, [two_cells()], funs, ["GFP"], {0: [1, 2]}
    )
    no_cells = fake.apply_extraction_functions(
        tiles, [np.array([])], funs, ["GFP"], {0: []}
    )
    assert "membrane_fluorescence_fl" in with_cells
    assert set(no_cells) == set(with_cells)
    assert all(found == ((), ()) for found in no_cells.values())


def test_only_cell_functions_can_be_asked_for():
    """Load the functions that measure a cell, and not their helpers."""
    cell_funs, _all_funs = load_all_functions()
    for helper in (
        "centroid",
        "crop_to_cell",
        "min_maj_approximation",
        "identify_vacuole",
        "is_model_available",
        "warn_about_missing_models",
        "_get_model",
    ):
        assert helper not in cell_funs
    assert {"area", "volume", "mean", "ratio_1_over_2"} <= set(cell_funs)


def test_div0_is_not_a_reduction():
    """Offer no reduction that cannot reduce z."""
    assert "div0" not in load_reduction_functions()


def test_functions_of_pixels_give_what_each_gives_alone():
    """Measure a cell's pixels once for every function that needs them."""
    fake = real_extractor(obscured={0: {2}})
    rng = np.random.default_rng(1)
    tiles = rng.integers(0, 4000, (2, 40, 40)).astype(np.uint16)
    masks = [two_cells(), two_cells()[:1]]
    labels = {0: [1, 2], 1: [5]}
    funs = list(cell_functions.PIXEL_FUNCTIONS) + ["moment_of_inertia"]
    found = fake.apply_extraction_functions(
        tiles, masks, funs, ["GFP"], labels
    )
    assert list(found) == funs
    for fun in funs:
        alone = fake.apply_extraction_function(
            tiles, masks, fun, labels, ["GFP"]
        )
        assert found[fun] == alone
        # the cell its tile cuts off is not measured
        assert found[fun][1] == ((0, 1), (1, 5))
    mask = two_cells()[0]
    for fun, of_pixels in cell_functions.PIXEL_FUNCTIONS.items():
        assert getattr(cell_functions, fun)(mask, tiles[0]) == of_pixels(
            tiles[0][mask]
        )


def test_images_are_reduced_once_for_a_cell_and_its_parts(monkeypatch):
    """Reduce a channel's z sections once for the cell and its parts."""
    import extraction.core.extractor as extractor_module

    fake = real_extractor()
    fake.reduce_extract = lambda **kwargs: Extractor.reduce_extract(
        fake, **kwargs
    )
    reduced = []
    reduce_z = extractor_module.reduce_z
    monkeypatch.setattr(
        extractor_module,
        "reduce_z",
        lambda *args, **kwargs: reduced.append(1) or reduce_z(*args, **kwargs),
    )
    tiles = np.random.default_rng(2).random((1, 3, 40, 40))
    asked = dict(
        tiles=tiles,
        reduction_funs={"max": ["mean"], "mean": ["mean"]},
        channels=["GFP"],
        cell_labels={0: [1, 2]},
    )
    whole = fake.reduce_extract(masks=[two_cells()], reduced_as="GFP", **asked)
    part = fake.reduce_extract(masks=[two_cells()], reduced_as="GFP", **asked)
    assert len(reduced) == 2
    assert whole == part
    assert whole["max"]["mean"][0] == tuple(
        tiles[0].max(axis=0)[mask].mean() for mask in two_cells()
    )
    # with no name, nothing is kept
    fake.reduce_extract(masks=[two_cells()], **asked)
    assert len(reduced) == 4


def test_a_background_channel_is_not_extracted_less_its_background():
    """Extract no _bgsub for a channel such as Cy5."""
    # the background channels were looked for where they never were, so
    # the background of an image less its background was extracted
    asked = []
    fake = SimpleNamespace(
        params=SimpleNamespace(
            subtract_background={"GFP", "cy5"},
            background_channels={"cy5"},
            intracellular_masks=set(),
        ),
        reduce_extract=lambda tiles, **kwargs: asked.append(tiles) or {},
    )
    fake._extract_channel = lambda *args: Extractor._extract_channel(
        fake, *args
    )
    tree = {"tree": {"GFP": {}, "cy5": {}}}
    img = {"GFP": "gfp", "cy5": "cy5"}
    img_bgsub = {"GFP_bgsub": "gfp_bgsub", "cy5_bgsub": None}
    found = Extractor.extract_single_channel_functions(
        fake, tree, {}, img, img_bgsub, []
    )
    assert list(found) == ["GFP", "GFP_bgsub", "cy5"]
    assert asked == ["gfp", "gfp_bgsub", "cy5"]


def background_subtracter(multichannel_funs):
    """Return something that subtracts backgrounds as an Extractor does."""
    return SimpleNamespace(
        params=SimpleNamespace(
            subtract_background={"GFP", "cy5"},
            background_channels={"cy5"},
            multichannel_funs=multichannel_funs,
        )
    )


def test_a_background_channel_has_no_image_less_background_unless_used():
    """Subtract a background channel's background only for a ratio."""
    tiles = np.random.default_rng(3).random((2, 2, 1, 3, 20, 20))
    bgs = np.ones((2, 20, 20), dtype=bool)
    tree_dict = {
        "channels_tree": {"GFP": {}, "cy5": {}},
        "channels": ("GFP", "cy5"),
    }
    _img, img_bgsub = Extractor.get_imgs_background_subtract(
        background_subtracter({}), tree_dict, tiles, bgs
    )
    assert img_bgsub["cy5_bgsub"] is None
    assert img_bgsub["GFP_bgsub"] is not None
    ratio = {"ratio": [["GFP", "cy5"], "max", "ratio_1_over_2"]}
    _img, img_bgsub = Extractor.get_imgs_background_subtract(
        background_subtracter(ratio), tree_dict, tiles, bgs
    )
    assert img_bgsub["cy5_bgsub"] is not None


def test_brightfield_is_the_reference_channel_whatever_its_name(
    monkeypatch,
):
    """Read brightfield by the Tiler's channel, and outlines if needed."""
    import extraction.core.extractor as extractor_module

    monkeypatch.setattr(extractor_module, "Cells", lambda store: None)
    asked = []

    def get_tiles_timepoint(tp, channels, z, lazy):
        asked.append((channels, z, lazy))
        return np.zeros((1, 1, 1, 3, 20, 20))

    fake = SimpleNamespace(
        store=None,
        params=SimpleNamespace(intracellular_masks=set()),
        tiler=SimpleNamespace(
            ref_channel_index=2,
            shape=(4, 3, 3, 100, 100),
            get_tiles_timepoint=get_tiles_timepoint,
        ),
        get_cell_labels=lambda tp, cell_labels, cells: {0: []},
        get_masks=lambda tp, masks, cells: [np.array([])],
        get_outlines=lambda *args: pytest.fail("the outlines were read"),
        get_tiles=lambda tp, channels, lazy: None,
        set_pdms_mask=lambda brightfield, masks: None,
        get_background_masks=lambda masks, tile_size: np.array([]),
        get_imgs_background_subtract=lambda tree_dict, tiles, bgs: ({}, {}),
    )
    _labels, _masks, outlines, img, _img_bgsub = Extractor.load_tp_data(
        fake, 1, 20, {"channels": ()}
    )
    assert asked == [([2], [0, 1, 2], False)]
    assert img["Brightfield"].shape == (1, 3, 20, 20)
    assert outlines is None


def test_an_extractor_has_its_parameters(tmp_path):
    """Give the parameters as every step of the pipeline does."""
    parameters = ExtractorParameters(
        tree={"general": {"null": ["area"]}}, identify_vacuoles=False
    )
    extractor = Extractor(parameters, store=tmp_path / "none.h5")
    assert extractor.parameters is parameters
