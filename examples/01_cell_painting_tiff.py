"""
Cell Painting profiling: TIFF directory -> cellpose -> cp_measure features.

This example consolidates the pattern used by two downstream applications
(``afermg/uoe-kenneth-amiodarone`` and an internal GSK-equivalent pipeline).
Both produce per-cell profiles from multi-channel epifluorescence Cell
Painting stacks and parallelise per-position with joblib.

Run top-to-bottom. The default input is the small bundled Zenodo fixture
(``aliby.test_data.get_dataset("crop_cellpainting_256")``) which downloads
on first use via pooch and caches at ``~/.cache/pooch/aliby_tests/``.

To swap in your own data, replace ``DATA_PATH`` / ``REGEX`` /
``CAPTURE_ORDER`` below with the TIFF directory + filename convention you
want to process.
"""

from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

import pyarrow.parquet
from joblib import Parallel, delayed

from aliby.io.dataset import DatasetDir
from aliby.pipe import run_pipeline_and_post
from aliby.pipe_builder import build_pipeline_steps
from aliby.pipe_core import configure_logging
from aliby.test_data import get_dataset, get_dataset_path

# ---------------------------------------------------------------------------
# 1. Choose input data and its filename convention.
# ---------------------------------------------------------------------------
# The bundled fixture is a 5-channel Cell Painting set (DNA/ER/RNA/AGP/Mito)
# at 256x256, one position (well A01, field 1). 516 KB on disk.
ENTRY = get_dataset("crop_cellpainting_256")
DATA_PATH = get_dataset_path(ENTRY["name"])
REGEX = ENTRY["regex"]                  # r".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+)\.tif"
CAPTURE_ORDER = ENTRY["capture_order"]  # "WFC" -- well, field, channel
CHANNELS = ENTRY["channels"]            # {"DNA": 0, "ER": 1, "RNA": 2, "AGP": 3, "Mito": 4}

# ---------------------------------------------------------------------------
# 2. Discover positions in the dataset.
# ---------------------------------------------------------------------------
datasets = DatasetDir(DATA_PATH, regex=REGEX, capture_order=CAPTURE_ORDER)
positions = datasets.get_position_ids()
# positions is a list[dict] of {"key": "<well>__<field>", "path": <DATA_PATH>},
# one entry per (well, field). For the bundled single-position fixture, one
# entry. For real plates this would be hundreds to thousands.
print(f"Discovered {len(positions)} positions: {[p['key'] for p in positions]}")
# Expected: Discovered 1 positions: ['A01__1']

# ---------------------------------------------------------------------------
# 3. Build the cellpose + cp_measure pipeline definition.
# ---------------------------------------------------------------------------
# `build_pipeline_steps` is the public builder for the standard pipeline
# (cellpose local or `nahual_cellpose` remote when `nahual_addresses=...`).
# It bundles per-channel sizeshape + intensity extraction plus per-pair
# colocalisation (extractmulti_*). Skipping edge-pixel intensity measurements
# (`cp_measure_feature_kwargs={"intensity":{"edge_measurements":False}}`)
# roughly halves runtime on dense fields.
NAHUAL_ADDRESSES: list[str] = []  # Set to e.g. ["ipc:///tmp/cellpose0.ipc", ...]

base_pipeline = build_pipeline_steps(
    channels_to_segment={"nuclei": CHANNELS["DNA"], "cell": CHANNELS["AGP"]},
    channels_to_extract=list(CHANNELS.values()),
    features_to_extract=("intensity", "sizeshape"),
    nahual_addresses=NAHUAL_ADDRESSES or None,
    cp_measure_feature_kwargs={"intensity": {"edge_measurements": False}},
)
# `base_pipeline` is a dict with top-level keys "steps", "passed_data",
# "passed_methods", "save", "save_interval". `steps` here contains:
#   tile, segment_nuclei, segment_cell, extract_nuclei, extract_cell,
#   extractmulti_nuclei, extractmulti_cell
print("Pipeline steps:", list(base_pipeline["steps"].keys()))


# ---------------------------------------------------------------------------
# 4. Per-position helper -- stamp the pipeline with image_kwargs and run it.
# ---------------------------------------------------------------------------
def build_pipeline_for_position(
    base_pipeline: dict,
    position: dict,
    nahual_addresses: list[str],
    position_index: int,
    regex: str = REGEX,
    capture_order: str = CAPTURE_ORDER,
) -> dict:
    """Return a deepcopy of ``base_pipeline`` configured for one position."""
    pipeline = deepcopy(base_pipeline)
    pipeline["io"] = {
        "input_path": {"key": position["key"], "path": position["path"]},
        "capture_order": capture_order,
    }
    pipeline["steps"]["tile"]["image_kwargs"] = {
        "source": {"key": position["key"], "path": position["path"]},
        "regex": regex,
        "capture_order": capture_order,
    }
    if nahual_addresses:
        addr = nahual_addresses[position_index % len(nahual_addresses)]
        for step_name, step_params in pipeline["steps"].items():
            if step_name.startswith("segment_") and "segmenter_kwargs" in step_params:
                step_params["segmenter_kwargs"]["address"] = addr
    return pipeline


def run_one_position(
    base_pipeline: dict,
    position: dict,
    output_path: Path,
    nahual_addresses: list[str],
    position_index: int,
    regex: str = REGEX,
    capture_order: str = CAPTURE_ORDER,
) -> None:
    configure_logging(output_path / "log.txt")
    pipeline = build_pipeline_for_position(
        base_pipeline,
        position,
        nahual_addresses,
        position_index,
        regex=regex,
        capture_order=capture_order,
    )
    run_pipeline_and_post(
        pipeline=pipeline,
        pipeline_name=position["key"],
        output_path=output_path,
        overwrite=False,
    )


if __name__ == "__main__":
    # -------------------------------------------------------------------
    # 5. Run all positions in parallel (one joblib worker per position).
    # -------------------------------------------------------------------
    OUTPUT_DIR = Path(mkdtemp(prefix="aliby_cellpainting_out_"))
    print(f"Writing per-position outputs under {OUTPUT_DIR}")

    Parallel(n_jobs=1, backend="loky")(
        delayed(run_one_position)(
            base_pipeline, pos, OUTPUT_DIR, NAHUAL_ADDRESSES, i
        )
        for i, pos in enumerate(positions)
    )

    # -------------------------------------------------------------------
    # 6. Inspect one of the parquet outputs.
    # -------------------------------------------------------------------
    profiles_files = sorted((OUTPUT_DIR / "profiles").glob("*.parquet"))
    print(f"Wrote {len(profiles_files)} profiles parquet files.")
    table = pyarrow.parquet.read_table(profiles_files[0])
    print(f"First profile: {profiles_files[0].name} -- {table.num_rows} rows, "
          f"{len(table.column_names)} columns")
    # Expected output on the bundled fixture, approximately:
    #   Wrote 1 profiles parquet files.
    #   First profile: A01__1.parquet -- 26 rows, 632 columns
    # Columns include metadata_tile / metadata_label / metadata_object /
    # metadata_tp plus per-channel cp_measure features like
    # "0/max/intensity/Intensity_IntegratedIntensity" and pairwise
    # colocalisation features like "(0, 3)/None/max/pearson".
    print("Sample columns:", table.column_names[:8], "...")
