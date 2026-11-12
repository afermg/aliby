"""
Tests of what a run of the pipeline relies on between its steps.

Each pins a fault found in October 2026: the tiler's z section, the
channels extracted, how Signals are written and read back, time in
minutes, early stopping, a time point with no cells, and the images
read at each time point.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import h5py
import numpy as np
import pandas as pd
import pytest

from agora.io.cells import Cells
from agora.io.signal import Signal
from agora.io.writers import BabyWriter, ExtractorWriter, write_meta_to_h5
from aliby.global_settings import global_settings
from aliby.tile.tiler import Tiler, TilerParameters, find_channel_index
from aliby.tile.tiles import TileLocations
from extraction.tree import build_extraction_tree_from_meta

AREA = "general/null/area"
# as a Signal is asked for
SIGNAL = "/extraction/" + AREA


def areas(values: dict, tp: int) -> pd.DataFrame:
    """Make the extractor's data frame of areas for one time point."""
    index = pd.MultiIndex.from_tuples(values, names=["trap", "cell_label"])
    return pd.DataFrame({tp: list(values.values())}, index=index)


def write_position(path, interval=300, ntps=3, index=False):
    """Write an h5 with the area of two cells at each time point."""
    write_meta_to_h5(
        path,
        {
            "channels": ["Brightfield"],
            "time_settings/ntimepoints": ntps,
            "time_settings/timeinterval": interval,
        },
    )
    writer = ExtractorWriter(path)
    for tp in range(ntps):
        df = areas({(0, 1): 100.0 + tp, (1, 1): 200.0 + tp}, tp)
        if index:
            # as aliby wrote Signals until October 2026
            flat = df.reset_index().melt(
                id_vars=["trap", "cell_label"],
                var_name="time",
                value_name="value",
            )
            flat = flat.astype(
                {
                    "trap": np.int16,
                    "cell_label": np.int16,
                    "time": np.int32,
                    "value": np.float32,
                }
            )
            with pd.HDFStore(path, mode="a") as store:
                store.append("/extraction/" + AREA, flat, format="table")
        else:
            writer.write({AREA: df})
    return path


###
# the tiler's z section
###


def test_the_tiler_takes_its_z_section_from_the_metadata():
    pipeline = pytest.importorskip("aliby.pipeline")
    meta = {"metadata": {"full": {"number_z_sections": {"Brightfield": 5}}}}
    ref_z = pipeline.PipelineParameters.apply_ref_z(meta)
    assert ref_z == 2
    full = type("Meta", (), {"full": {}})
    tiler = pipeline.PipelineParameters.build_tiler_defaults(full, {}, ref_z)
    assert tiler["ref_z"] == 2


def test_a_z_section_that_is_asked_for_is_used():
    pipeline = pytest.importorskip("aliby.pipeline")
    full = type("Meta", (), {"full": {}})
    tiler = pipeline.PipelineParameters.build_tiler_defaults(
        full, {"ref_z": 0}, 2
    )
    assert tiler["ref_z"] == 0


def test_metadata_without_z_sections_leaves_the_default():
    pipeline = pytest.importorskip("aliby.pipeline")
    assert (
        pipeline.PipelineParameters.apply_ref_z({"metadata": {"full": {}}})
        is None
    )


###
# channels
###


@pytest.mark.parametrize(
    "channels",
    [
        ["Brightfield", "GFPFast", "GFP"],
        ["Brightfield", "GFP_Z_10ms", "GFP_Z_30ms"],
    ],
)
def test_channels_that_begin_alike_are_all_extracted(channels):
    tree = build_extraction_tree_from_meta({"channels": channels})["tree"]
    assert set(tree) == {"general", *channels[1:]}


def test_a_channel_matched_in_part_is_warned_of():
    messages = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    logger = logging.getLogger("aliby")
    logger.addHandler(handler)
    try:
        assert find_channel_index(["Brightfield", "GFP_Z"], "GFP") == 1
        assert len(messages) == 1
        assert find_channel_index(["Brightfield", "GFP_Z"], "GFP_Z") == 1
        assert len(messages) == 1
    finally:
        logger.removeHandler(handler)


@pytest.mark.parametrize(
    "channels, channel, index",
    [
        (["Brightfield", "GFPFast", "GFP"], "GFP", 2),
        (["Brightfield", "GFPFast", "GFP"], "GFPFast", 1),
        (["Brightfield", "GFPFast", "GFP"], "gfp", 2),
        (["Brightfield", "Cy5"], "cy5", 1),
    ],
)
def test_a_channel_named_in_full_is_found_before_one_that_begins_alike(
    channels, channel, index
):
    assert find_channel_index(channels, channel) == index


def test_choosing_when_to_stop_early_leaves_the_global_settings_alone():
    pipeline = pytest.importorskip("aliby.pipeline")
    before = dict(global_settings.earlystop)
    meta = SimpleNamespace(minimal={"time_settings/ntimepoints": 5}, full={})
    defaults = pipeline.PipelineParameters.build_general_defaults(
        "expt", Path("results/expt"), meta, {"earlystop": {"min_tp": 7}}
    )
    assert defaults["general"]["earlystop"]["min_tp"] == 7
    assert global_settings.earlystop == before


###
# Signals in the h5 file
###


def test_a_signal_is_written_with_no_index(tmp_path):
    path = write_position(tmp_path / "position.h5")
    with h5py.File(path, "r") as f:
        assert list(f["extraction/" + AREA]) == ["table"]


def test_a_signal_reads_the_same_with_and_without_an_index(tmp_path):
    without = Signal(write_position(tmp_path / "without.h5"))
    with_index = Signal(write_position(tmp_path / "with.h5", index=True))
    assert without.available == with_index.available == ["extraction/" + AREA]
    pd.testing.assert_frame_equal(
        without.get_raw(SIGNAL), with_index.get_raw(SIGNAL)
    )


def test_a_signal_knows_its_number_of_time_points(tmp_path):
    signal = Signal(write_position(tmp_path / "position.h5", ntps=4))
    assert signal.ntimepoints == 4


def test_available_lists_each_signal_once(tmp_path):
    signal = Signal(write_position(tmp_path / "position.h5"))
    assert signal.available == ["extraction/" + AREA]


def test_a_time_point_with_no_cells_is_still_a_column(tmp_path):
    path = tmp_path / "position.h5"
    write_meta_to_h5(path, {"time_settings/timeinterval": 300})
    writer = ExtractorWriter(path)
    for tp in (0, 1, 3):
        writer.write({AREA: areas({(0, 1): 100.0}, tp)})
    df = Signal(path).get_raw(SIGNAL, in_minutes=False)
    assert list(df.columns) == [0, 1, 2, 3]
    assert df[2].isna().all()


###
# time in minutes
###


@pytest.mark.parametrize(
    "interval, minutes",
    [(300, [0, 5, 10]), (150, [0, 2.5, 5]), (30, [0, 0.5, 1])],
)
def test_columns_in_minutes_are_exact(tmp_path, interval, minutes):
    signal = Signal(write_position(tmp_path / "position.h5", interval))
    assert list(signal.get_raw(SIGNAL).columns) == minutes


def test_whole_minutes_stay_integers(tmp_path):
    signal = Signal(write_position(tmp_path / "position.h5", 300))
    assert signal.get_raw(SIGNAL).columns.dtype.kind == "i"


def test_a_list_of_signals_keeps_time_points_when_asked(tmp_path):
    signal = Signal(write_position(tmp_path / "position.h5", 300))
    (df,) = signal.get([SIGNAL], in_minutes=False)
    assert list(df.columns) == [0, 1, 2]


###
# early stopping
###

EARLYSTOP = {
    "min_tp": 100,
    "thresh_pos_clogged": 0.4,
    "thresh_trap_ncells": 2,
    "thresh_trap_area": 0.5,
    "ntps_to_eval": 2,
}


def crowded(tp, crowded_trap=True):
    """Give trap 0 three large cells, or one, and trap 1 one small cell."""
    cells = {(1, 1): 5.0}
    for label in range(1, 4 if crowded_trap else 2):
        cells[(0, label)] = 30.0
    return areas(cells, tp)


def test_a_crowded_tile_is_clogged():
    pipeline = pytest.importorskip("aliby.pipeline")
    check = pipeline.CloggingCheck(EARLYSTOP, tile_size=10)
    assert check.update(crowded(0)) == 0.5


def test_the_latest_time_point_counts_towards_clogging():
    pipeline = pytest.importorskip("aliby.pipeline")
    check = pipeline.CloggingCheck(EARLYSTOP, tile_size=10)
    for tp in range(3):
        assert check.update(crowded(tp, crowded_trap=False)) == 0
    # one crowded time point of two averages two cells, which is not over
    assert check.update(crowded(3)) == 0
    assert check.update(crowded(4)) == 0.5


def test_time_points_with_no_cells_are_not_clogged():
    pipeline = pytest.importorskip("aliby.pipeline")
    check = pipeline.CloggingCheck(EARLYSTOP, tile_size=10)
    assert check.update(None) == 0
    assert check.update(pd.DataFrame()) == 0


def test_clogging_is_the_same_from_memory_and_from_the_file(tmp_path):
    pipeline = pytest.importorskip("aliby.pipeline")
    path = tmp_path / "position.h5"
    write_meta_to_h5(path, {"time_settings/timeinterval": 300})
    writer = ExtractorWriter(path)
    check = pipeline.CloggingCheck(EARLYSTOP, tile_size=10)
    for tp, is_crowded in enumerate([False, False, True, True]):
        fraction = check.update(crowded(tp, is_crowded))
        writer.write({AREA: crowded(tp, is_crowded)})
    assert fraction == 0.5
    assert pipeline.check_earlystop(path, EARLYSTOP, 10) == fraction


###
# a time point with no cells
###


def tile_result(labels):
    """Make BABY's result for one tile."""
    return {
        "cell_label": list(labels),
        "centres": [[1, 1]] * len(labels),
        "mother_assign": [0] * max([0, *labels]),
    }


def test_no_cells_is_an_empty_time_point():
    baby_sitter = pytest.importorskip("aliby.baby_sitter")
    result = baby_sitter.format_segmentation(
        [tile_result([]), tile_result([])], tp=3
    )
    assert all(len(value) == 0 for value in result.values())
    assert "timepoint" in result


def test_cells_are_given_their_trap_and_time_point():
    baby_sitter = pytest.importorskip("aliby.baby_sitter")
    result = baby_sitter.format_segmentation(
        [tile_result([1, 2]), tile_result([]), tile_result([4])], tp=3
    )
    assert result["trap"] == [0, 0, 2]
    assert result["cell_label"] == [1, 2, 4]
    assert result["timepoint"] == [3, 3, 3]


def test_outputs_of_different_lengths_are_refused():
    baby_sitter = pytest.importorskip("aliby.baby_sitter")
    tile = tile_result([1, 2])
    tile["centres"] = [[1, 1]]
    with pytest.raises(baby_sitter.InconsistentOutput, match="centres"):
        baby_sitter.format_segmentation([tile], tp=0)


def baby_result(traps, tp, size=8):
    """Make a formatted result of BABY with one cell in each trap given."""
    n = len(traps)
    return {
        "trap": list(traps),
        "cell_label": [1] * n,
        "timepoint": [tp] * n,
        "mother_assign_dynamic": [0] * n,
        "edgemasks": np.ones((n, size, size), dtype=bool),
    }


@pytest.mark.parametrize("empty_tp", [0, 1])
def test_a_time_point_with_no_cells_can_be_written_and_read(
    tmp_path, empty_tp
):
    path = tmp_path / "position.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("trap_info/trap_locations", data=np.zeros((2, 2)))
    writer = BabyWriter(path)
    for tp in range(3):
        traps = [] if tp == empty_tp else [0, 1]
        writer.write(baby_result(traps, tp), overwrite=[], tp=tp, tile_size=8)
    cells = Cells(path)
    assert sorted(set(cells["timepoint"])) == sorted({0, 1, 2} - {empty_tp})
    assert cells.at_time(empty_tp) == {0: [], 1: []}
    assert cells.labels_at_time(2) == {0: [1], 1: [1]}
    with h5py.File(path, "r") as f:
        assert f["cell_info/edgemasks"].shape == (4, 8, 8)


###
# cells
###


def cells_with(tmp_path, rows, mothers=None, ntraps=2):
    """Write an h5 of (trap, time point, label) rows and return its Cells."""
    path = tmp_path / "cells.h5"
    trap, tp, label = (
        np.array(column, dtype="uint16") for column in zip(*rows)
    )
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "trap_info/trap_locations", data=np.zeros((ntraps, 2))
        )
        f.create_dataset("cell_info/trap", data=trap)
        f.create_dataset("cell_info/timepoint", data=tp)
        f.create_dataset("cell_info/cell_label", data=label)
        f.create_dataset(
            "cell_info/mother_assign_dynamic",
            data=np.array(mothers or [0] * len(rows), dtype="uint16"),
        )
    return Cells(path)


def test_a_cell_takes_the_mother_it_was_last_given(tmp_path):
    rows = [(0, 0, 1), (0, 0, 2), (0, 1, 2), (1, 1, 5), (0, 2, 2)]
    cells = cells_with(tmp_path, rows, mothers=[0, 0, 1, 0, 0], ntraps=3)
    # cell 2 of trap 0 was given mother 1 and then none
    assert [list(map(int, m)) for m in cells.mothers] == [[0, 0], [0], []]


###
# images read
###


class CountingImage:
    """Serve an image as a zarr store does and count what is read."""

    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self.data[key]


def counting_tiler(n_tps=3):
    """Make a Tiler of two tiles over an image that counts its reads."""
    data = np.random.default_rng(0).integers(
        0, 1000, (n_tps, 2, 3, 64, 64), dtype=np.uint16
    )
    tiler = Tiler(
        CountingImage(data),
        {"channels": ["Brightfield", "GFP"]},
        TilerParameters.default(tile_size=16, ref_z=1),
    )
    tiler.tile_locs = TileLocations([[20, 20], [40, 40]], tile_size=16)
    tiler.no_processed = 1
    return tiler, data


def test_each_image_is_read_once_at_a_time_point():
    tiler, data = counting_tiler()
    for tp in range(3):
        tiler.find_drift(tp)
        # for segmenting and then for extracting
        first = tiler.get_tp_data_for_one_channel(tp, 0, lazy=False)
        second = tiler.get_tp_data_for_one_channel(tp, 0, lazy=False)
        np.testing.assert_array_equal(first, second)
    reads = tiler.image.reads
    # at each time point, one plane to find the drift and one z stack
    assert reads == [key for tp in range(3) for key in ((tp, 0, 1), (tp, 0))]


def test_tiles_are_the_same_from_numpy_and_from_dask():
    tiler, data = counting_tiler()
    # a tile that hangs off the image, to be padded
    tiler.tile_locs = TileLocations([[20, 20], [60, 5]], tile_size=16)
    tiler.tile_locs.drifts = [[0, 0]]
    from_numpy = tiler.get_tp_data_for_one_channel(0, 1, lazy=False)
    tiler.image = da.from_array(data)
    lazy = tiler.get_tp_data_for_one_channel(0, 1, lazy=True)
    assert isinstance(lazy, da.Array)
    np.testing.assert_array_equal(from_numpy, lazy.compute())
    assert from_numpy.shape == (2, 3, 16, 16)


###
# faults that ended a run, or would have
###


def test_cells_with_no_mothers_assigned_are_formatted():
    baby_sitter = pytest.importorskip("aliby.baby_sitter")
    tiles = [tile_result([1, 2]), tile_result([3])]
    for tile in tiles:
        del tile["mother_assign"]
    result = baby_sitter.format_segmentation(tiles, tp=3)
    assert result["cell_label"] == [1, 2, 3]
    assert "mother_assign_dynamic" not in result


def test_a_model_set_of_our_own_files_is_given_to_the_brain(monkeypatch):
    baby_sitter = pytest.importorskip("aliby.baby_sitter")
    made = []
    monkeypatch.setattr(
        baby_sitter, "BabyBrain", lambda **kwargs: made.append(kwargs)
    )
    files = {
        "flattener_file": Path("retrained/flattener.json"),
        "morph_model_file": Path("retrained/weights/weights.h5"),
    }
    baby_sitter.get_brain(files, clogging_thresh=1)
    assert made == [{**files, "clogging_thresh": 1}]


def test_a_dict_is_not_a_source_of_images():
    from aliby.io.image import dispatch_image

    with pytest.raises(ValueError, match="Invalid data source"):
        dispatch_image({"Brightfield": "pos001"})


def local_dataset(folder: Path, positions: list[str]) -> Path:
    """Write a data set of TIFF folders, each with a log of its own."""
    for position in positions:
        (folder / position).mkdir()
        (folder / position / "pos_t0001_Brightfield_z01.tif").write_bytes(b"")
        (folder / position / "notes.txt").write_text(position)
    return folder


def test_a_folder_is_not_a_metadata_file(tmp_path):
    from aliby.io.dataset import DatasetLocalDir

    data = tmp_path / "data"
    data.mkdir()
    (data / "catalog").mkdir()
    (data / "weblog").write_text("not a log")
    (data / "expt.log").write_text("a log")
    out = tmp_path / "out"
    out.mkdir()
    dataset = DatasetLocalDir(data)
    assert [file.name for file in dataset.files] == ["expt.log"]
    dataset.cache_logs(out)
    assert [file.name for file in out.iterdir()] == ["expt.log"]


def test_of_logs_with_one_name_the_one_nearest_the_top_is_kept(tmp_path):
    from aliby.io.dataset import DatasetLocalDir

    data = tmp_path / "data"
    data.mkdir()
    local_dataset(data, ["pos002", "pos001"])
    (data / "notes.txt").write_text("top")
    out = tmp_path / "out"
    out.mkdir()
    DatasetLocalDir(data).cache_logs(out)
    assert (out / "notes.txt").read_text() == "top"


def test_local_positions_are_in_order(tmp_path):
    from aliby.io.dataset import DatasetLocalDir

    names = ["pos010", "pos002", "pos001", "pos003"]
    local_dataset(tmp_path, names)
    assert list(DatasetLocalDir(tmp_path).get_position_ids()) == sorted(names)


def bare_pipeline(tmp_path, **general):
    """Make a Pipeline with only its general parameters."""
    pipeline = pytest.importorskip("aliby.pipeline")
    parameters = pipeline.PipelineParameters(
        general={"expt_id": "expt", "directory": str(tmp_path), **general},
        tiler={},
        baby={},
        extraction={},
        postprocessing={},
    )
    return pipeline.Pipeline(parameters)


@pytest.fixture
def aliby_logger():
    """Give the aliby logger and put its handlers back afterwards."""
    logger = logging.getLogger("aliby")
    handlers, level, propagate = (
        list(logger.handlers),
        logger.level,
        logger.propagate,
    )
    yield logger
    for handler in logger.handlers:
        handler.close()
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_the_log_file_takes_the_level_asked_for(tmp_path, aliby_logger):
    bare_pipeline(tmp_path, logfile_level="DEBUG")
    (file_handler,) = [
        handler
        for handler in aliby_logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]
    assert file_handler.level == logging.DEBUG
    aliby_logger.debug("a detail")
    file_handler.flush()
    assert "a detail" in (tmp_path / "aliby_expt.log").read_text()


def test_one_position_s_error_leaves_the_others_to_run(
    tmp_path, aliby_logger, monkeypatch
):
    pipe = bare_pipeline(tmp_path)
    ran = []

    def run_one_position(name_image_id):
        ran.append(name_image_id[0])
        if name_image_id[0] == "pos002":
            raise ValueError("Catastrophic Baby error!")

    positions = {"pos001": 1, "pos002": 2, "pos003": 3}
    monkeypatch.setattr(pipe, "run_one_position", run_one_position)
    monkeypatch.setattr(pipe, "setup", lambda: positions)
    monkeypatch.setattr(pipe, "report_h5_sizes", lambda: {})
    pipe.parameters.general.update(filter=None, distributed=0)
    pipe.parameters.metadata = {"minimal": {}}
    pipe.parameters.extraction = {"tree": {}}
    pipe.run()
    assert ran == list(positions)
    log = (tmp_path / "aliby_expt.log").read_text()
    assert "pos002 raised an error" in log
    assert "Traceback" in log and "Catastrophic Baby error!" in log


def test_an_error_at_every_position_is_raised(
    tmp_path, aliby_logger, monkeypatch
):
    pipe = bare_pipeline(tmp_path)

    def run_one_position(name_image_id):
        raise KeyError("no Brightfield")

    monkeypatch.setattr(pipe, "run_one_position", run_one_position)
    monkeypatch.setattr(pipe, "setup", lambda: {"pos001": 1, "pos002": 2})
    monkeypatch.setattr(pipe, "report_h5_sizes", lambda: {})
    pipe.parameters.general.update(filter=None, distributed=0)
    pipe.parameters.metadata = {"minimal": {}}
    pipe.parameters.extraction = {"tree": {}}
    with pytest.raises(RuntimeError, match="Every position raised"):
        pipe.run()


###
# work done once
###


def test_a_lazy_tile_reads_only_its_own_pixels():
    tiler, data = counting_tiler()
    image = tiler.image
    image.dtype, image.ndim = data.dtype, data.ndim
    for get_tile in (tiler.get_lazy_tile_view, tiler.get_tile_data):
        image.reads.clear()
        tile = np.asarray(get_tile(1, tp=2, c=1))
        np.testing.assert_array_equal(tile, data[2, 1, :, 32:48, 32:48])
        # the tile's three sections, not the stack they were cut from
        assert sum(data[key].size for key in image.reads) == 3 * 16 * 16


def test_lazy_tiles_are_the_same_from_an_array_and_from_dask():
    tiler, data = counting_tiler()
    tiler.image.dtype, tiler.image.ndim = data.dtype, data.ndim
    from_array = np.asarray(tiler.get_tiles_lazy(1, 0))
    tiler.image = da.from_array(data)
    np.testing.assert_array_equal(
        from_array, np.asarray(tiler.get_tiles_lazy(1, 0))
    )


def test_a_zarr_store_is_not_searched_for_logs(tmp_path, monkeypatch):
    import os

    from aliby.io.dataset import DatasetLocalDir

    data = tmp_path / "data"
    chunks = data / "pos001.zarr" / "0" / "0"
    chunks.mkdir(parents=True)
    (chunks / "chunk.txt").write_text("pixels")
    (data / "pos001.zarr" / "notes.txt").write_text("in the store")
    local_dataset(data, ["pos002"])
    (data / "expt.log").write_text("a log")
    scanned = []
    scandir = os.scandir
    monkeypatch.setattr(
        os, "scandir", lambda path: scanned.append(str(path)) or scandir(path)
    )
    files = DatasetLocalDir(data).files
    assert sorted(str(file.relative_to(data)) for file in files) == [
        "expt.log",
        "pos002/notes.txt",
    ]
    assert not any(".zarr" in path for path in scanned)


class CountingDataset:
    """Stand in for a data set and count how often its logs are cached."""

    def __init__(self):
        self.cached = 0

    def cache_logs(self, directory):
        self.cached += 1
        (directory / "expt.log").write_text("a log")


def test_logs_are_cached_once_for_a_folder(tmp_path):
    pipeline = pytest.importorskip("aliby.pipeline")
    dataset = CountingDataset()
    pipeline.cache_logs_once(dataset, "expt", tmp_path)
    pipeline.cache_logs_once(dataset, "expt", tmp_path)
    assert dataset.cached == 1
    # another folder, and a folder since emptied, are cached afresh
    other = tmp_path / "other"
    other.mkdir()
    pipeline.cache_logs_once(dataset, "expt", other)
    assert dataset.cached == 2
    (other / "expt.log").unlink()
    pipeline.cache_logs_once(dataset, "expt", other)
    assert dataset.cached == 3


def test_an_h5_file_is_named_without_opening_the_image(
    tmp_path, aliby_logger, monkeypatch
):
    pipeline = pytest.importorskip("aliby.pipeline")
    pipe = bare_pipeline(tmp_path, use_explog=False)

    def no_image(image_id):
        raise AssertionError("the image was opened")

    monkeypatch.setattr(pipeline, "dispatch_image", no_image)
    stale = tmp_path / "pos001.h5"
    stale.write_text("from an earlier run")
    assert pipe.generate_h5file("pos001") == stale
    assert not stale.exists()
