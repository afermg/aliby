"""
End-to-end coverage for the three published examples against the shared
pooch-fetched Zenodo test dataset.

Each test exercises one example script's pipeline shape (without invoking
its CLI) so the published examples are pinned against the canonical test
data. The nahual_embed example only exercises the ndarray-branch wiring in
``get_profiles_from_state`` -- a live Nahual embedder server is not started.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from aliby.io.dataset import DatasetDir, DatasetZarr, dispatch_dataset
from aliby.pipe import run_pipeline_and_post
from aliby.pipe_builder import build_pipeline_steps
from aliby.pipe_core import get_profiles_from_state
from aliby.test_data import get_dataset, get_dataset_path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _import_example(stem: str):
    """Load an example module by stem so its helpers are reusable here."""
    path = EXAMPLES_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module


def test_example_01_cell_painting_tiff(tmp_path):
    """01_cell_painting_tiff: TIFF dir -> cellpose -> cp_measure."""
    entry = get_dataset("crop_cellpainting_256")
    path = get_dataset_path(entry["name"])
    if not path.exists():
        pytest.skip(f"Test dataset not present: {path}")

    example = _import_example("01_cell_painting_tiff")

    datasets = DatasetDir(
        path, regex=entry["regex"], capture_order=entry["capture_order"]
    )
    positions = datasets.get_position_ids()
    assert positions, f"No positions discovered in {path}"

    base_pipeline = build_pipeline_steps(
        channels_to_segment={"nuclei": entry["channels"]["DNA"]},
        channels_to_extract=[entry["channels"]["DNA"]],
        features_to_extract=("intensity", "sizeshape"),
    )
    pipeline = example.build_pipeline_for_position(
        base_pipeline,
        positions[0],
        nahual_addresses=[],
        position_index=0,
    )
    run_pipeline_and_post(
        pipeline=pipeline,
        pipeline_name=positions[0]["key"],
        output_path=tmp_path,
        overwrite=True,
    )

    profiles = tmp_path / "profiles" / f"{positions[0]['key']}.parquet"
    assert profiles.exists() and profiles.stat().st_size > 0


def test_example_02_zarr_deep_embeddings_pipeline_shape(tmp_path):
    """02_zarr_deep_embeddings: pipeline dict is well-formed and the
    nahual_embed terminal pathway accepts a stub embedding without an
    actual remote.

    The end-to-end run requires a Nahual model server. Here we build the
    pipeline dict against the real monozarr fixture and verify the PR #20
    fix path by feeding a stub ndarray into ``get_profiles_from_state``."""
    entry = get_dataset("crop_cellpainting_256.zarr")
    path = get_dataset_path(entry["name"])
    if not path.exists():
        pytest.skip(f"Test dataset not present: {path}")

    example = _import_example("02_zarr_deep_embeddings")

    dataset = dispatch_dataset(path, is_zarr=True)
    positions = dataset.get_position_ids()
    assert positions, f"No zarr groups under {path}"

    model_config = example.MODEL_REGISTRY["dinov2"]
    pipeline = example.build_embed_pipeline(
        positions[0],
        address="ipc:///tmp/not_a_real_server.ipc",
        model_config=model_config,
    )

    assert "tile" in pipeline["steps"]
    embed_steps = [k for k in pipeline["steps"] if k.startswith("nahual_embed_")]
    assert len(embed_steps) == 1
    assert pipeline["steps"][embed_steps[0]]["model_group"] == "dinov2"
    assert pipeline["passed_data"][embed_steps[0]] == [("pixels", "tile", "data")]

    # PR #20 regression check on the resulting ndarray contract: the embed
    # step's output is wrapped by get_profiles_from_state into a length-1
    # instructions / length-1 metrics tuple before reaching format_extraction.
    stub_embedding = np.arange(12, dtype=np.float32).reshape(3, 4)
    state = {"data": {embed_steps[0]: [stub_embedding]}}
    profiles = get_profiles_from_state(state, pipeline)
    assert isinstance(profiles, pa.Table)
    assert profiles.num_rows > 0


def test_example_03_yeast_timelapse_baby_pipeline_shape(tmp_path):
    """03_yeast_timelapse_baby: build_pipeline shape and zgroup wrapper
    work against the real yeast time-series zarr.

    The end-to-end BABY run requires a baby-phone server; here we exercise
    the example's :func:`wrap_zarr_as_group` helper and confirm the
    pipeline dict carries the BABY-specific steps and passed_methods.
    """
    entry = get_dataset("crop_timeseries_alcatras_square_same_channels_293.zarr")
    root = get_dataset_path(entry["name"])
    if not root.exists():
        pytest.skip(f"Test dataset not present: {root}")

    example = _import_example("03_yeast_timelapse_baby")

    # Pick the first per-position group inside the zarr root.
    dataset = DatasetZarr(root)
    positions = dataset.get_position_ids()
    assert positions, f"No zarr groups under {root}"
    pos_path = Path(positions[0]["path"]) / positions[0]["key"]

    group_path, key = example.wrap_zarr_as_group(pos_path)
    assert (group_path / ".zgroup").exists()
    assert key == pos_path.name

    pipeline = example.build_pipeline(
        pos_path,
        baby_address="ipc:///tmp/not_a_real_baby.ipc",
        baby_modelset="placeholder_modelset",
        ntps=2,
        tile_size=128,
        ref_channel=0,
        ref_z=0,
    )
    assert pipeline["ntps"] == 2
    assert "segment_cell" in pipeline["steps"]
    assert pipeline["steps"]["segment_cell"]["segmenter_kwargs"]["kind"] == "nahual_baby"
    assert pipeline["passed_methods"] == {"segment_cell": ("tile", "get_fczyx")}
