# /// script
# requires-python = ">=3.11, <3.14"
# dependencies = [
#     "marimo>=0.21.1",
#     "matplotlib>=3.10.8",
#     "numpy>=1.18",
#     "pooch>=1.8.2",
#     "aliby",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # nb02 — Cellpose Segmentation Pipeline

        Runs a full **Cellpose segmentation + feature extraction** pipeline on
        test data. This notebook demonstrates:

        1. Loading data using helpers from `nb01_data_loading`
        2. Building a pipeline with `build_pipeline_steps`
        3. Running segmentation and extraction via `run_pipeline_and_post`
        4. Inspecting the resulting profiles

        Based on `examples/basic_cellpose_cpmeasure_monozarr.py` and
        `tests/test_cellpose_cpmeasure_minimal.py`.
        """
    )
    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np

    _src = Path(__file__).resolve().parents[1] / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    # Import nb01 as a module so we can reuse its @app.function helpers
    _notebooks = Path(__file__).resolve().parent
    if str(_notebooks) not in sys.path:
        sys.path.insert(0, str(_notebooks))
    return Path, np, sys


@app.cell
def _(mo):
    mo.md("## 1. Load Test Data (via nb01)")
    return


@app.cell
def _(Path):
    import nb01_data_loading as nb01

    data_path = nb01.get_data_path(Path)
    catalog = nb01.dataset_catalog()
    return catalog, data_path, nb01


@app.cell
def _(catalog, mo):
    ds_dropdown = mo.ui.dropdown(
        options={d["name"]: d for d in catalog},
        value=catalog[0]["name"],
        label="Select dataset",
    )
    ds_dropdown
    return (ds_dropdown,)


@app.cell
def _(data_path, ds_dropdown, mo, nb01):
    ds_info = ds_dropdown.value
    _dset, positions = nb01.load_dataset_dir(data_path, ds_info)

    mo.md(
        f"**Dataset:** `{ds_info['name']}` — "
        f"**{len(positions)}** position(s), "
        f"capture order `{ds_info['capture_order']}`"
    )
    return ds_info, positions


@app.cell
def _(mo):
    mo.md("## 2. Configure Pipeline")
    return


@app.cell
def _(ds_info, mo):
    _is_timeseries = "T" in ds_info["capture_order"]

    # Sensible defaults per dataset type
    if ds_info["name"] == "crop_cellpainting_256":
        _default_seg = "nuclei:1, cell:0"
        _default_ext = "0, 1"
    else:
        _default_seg = "cells:0"
        _default_ext = "0, 1"

    seg_channels_input = mo.ui.text(
        value=_default_seg,
        label="Channels to segment (name:index, ...)",
    )
    ext_channels_input = mo.ui.text(
        value=_default_ext,
        label="Channels to extract (comma-separated indices)",
    )
    features_input = mo.ui.multiselect(
        options=[
            "intensity",
            "sizeshape",
            "radial_zernikes",
            "feret",
            "texture",
            "radial_distribution",
            "zernike",
        ],
        value=["intensity", "sizeshape"],
        label="Features to extract",
    )

    mo.vstack([seg_channels_input, ext_channels_input, features_input])
    return ext_channels_input, features_input, seg_channels_input


@app.function
def parse_seg_channels(text):
    """Parse 'nuclei:1, cell:0' into {'nuclei': 1, 'cell': 0}."""
    result = {}
    for pair in text.split(","):
        pair = pair.strip()
        if ":" in pair:
            name, idx = pair.split(":")
            result[name.strip()] = int(idx.strip())
    return result


@app.function
def parse_ext_channels(text):
    """Parse '0, 1, 2' into [0, 1, 2]."""
    return [int(x.strip()) for x in text.split(",") if x.strip()]


@app.cell
def _(
    ds_info,
    ext_channels_input,
    features_input,
    mo,
    parse_ext_channels,
    parse_seg_channels,
    positions,
    seg_channels_input,
):
    from aliby.pipe_builder import build_pipeline_steps

    channels_to_segment = parse_seg_channels(seg_channels_input.value)
    channels_to_extract = parse_ext_channels(ext_channels_input.value)

    _base = build_pipeline_steps(
        channels_to_segment=channels_to_segment,
        channels_to_extract=channels_to_extract,
        features_to_extract=features_input.value,
    )

    # Build a complete pipeline dict (avoid in-place mutation)
    pos = positions[0]
    key = pos["key"]
    path = pos["path"]

    _image_kwargs = {
        "source": {"key": key, "path": path},
        "regex": ds_info["regex"],
        "capture_order": ds_info["capture_order"],
    }

    # Reconstruct the tile step with image_kwargs included
    _tile_step = {**_base["steps"]["tile"], "image_kwargs": _image_kwargs}
    _steps = {**_base["steps"], "tile": _tile_step}

    pipeline = {
        **_base,
        "io": {
            "input_path": {"key": key, "path": path},
            "capture_order": ds_info["capture_order"],
        },
        "steps": _steps,
    }

    # Limit time points for time-series to keep the demo fast
    if "T" in ds_info["capture_order"]:
        pipeline["ntps"] = 2

    mo.md(
        f"### Pipeline Configuration\n\n"
        f"- **Segmentation channels**: {channels_to_segment}\n"
        f"- **Extraction channels**: {channels_to_extract}\n"
        f"- **Features**: {features_input.value}\n"
        f"- **Position**: `{key}`\n"
        f"- **Steps**: {list(pipeline['steps'].keys())}"
    )
    return channels_to_segment, key, path, pipeline, pos


@app.cell
def _(mo):
    mo.md("## 3. Run Pipeline")
    return


@app.cell
def _(Path, key, mo, pipeline):
    import tempfile

    from aliby.pipe import run_pipeline_and_post

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir)
        profiles, post_results = run_pipeline_and_post(
            pipeline=pipeline,
            pipeline_name=key,
            output_path=output_path,
        )

    if profiles is not None:
        n_rows = profiles.num_rows
        n_cols = profiles.num_columns
        col_names = profiles.column_names
        mo.md(
            f"### Pipeline Complete\n\n"
            f"- **Profiles**: {n_rows} rows x {n_cols} columns\n"
            f"- **Column names** (first 20): `{col_names[:20]}`"
        )
    else:
        mo.md("Pipeline returned no profiles.")
    return post_results, profiles


@app.cell
def _(mo):
    mo.md("## 4. Inspect Profiles")
    return


@app.cell
def _(mo, profiles):
    if profiles is not None and profiles.num_rows > 0:
        _df = profiles.to_pandas()
        mo.output.replace(mo.ui.table(_df.head(50), label="Extracted profiles (first 50 rows)"))
    else:
        mo.output.replace(mo.md("_No profile data to display._"))
    return


@app.cell
def _(mo, profiles):
    import matplotlib.pyplot as plt

    if profiles is not None and profiles.num_rows > 0:
        _df = profiles.to_pandas()

        # Find numeric feature columns (exclude metadata columns)
        _meta_cols = [c for c in _df.columns if c.startswith("metadata_")]
        _feat_cols = [c for c in _df.columns if c not in _meta_cols]

        if len(_feat_cols) > 0:
            _n_features = min(6, len(_feat_cols))
            _fig, _axes = plt.subplots(
                1, _n_features, figsize=(4 * _n_features, 3), squeeze=False
            )
            for _i, _col in enumerate(_feat_cols[:_n_features]):
                _ax = _axes[0][_i]
                _vals = _df[_col].dropna()
                if len(_vals) > 0:
                    _ax.hist(_vals, bins=20, edgecolor="black", alpha=0.7)
                _ax.set_title(_col, fontsize=8)
                _ax.tick_params(labelsize=7)

            _fig.suptitle("Feature Distributions", fontsize=12)
            plt.tight_layout()
            mo.output.replace(mo.mpl.interactive(_fig))
        else:
            mo.output.replace(mo.md("_No feature columns found._"))
    else:
        mo.output.replace(mo.md("_No profile data to visualize._"))
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Pipeline Dictionary Reference

        The pipeline is a plain dictionary with these keys:

        | Key | Purpose |
        |---|---|
        | `steps` | Step name -> parameters (tile, segment_X, extract_X) |
        | `passed_data` | Step name -> list of (param, source_step) tuples |
        | `passed_methods` | Step name -> (source_step, method_name) |
        | `save` | Which steps write intermediate results to disk |
        | `save_interval` | How often to flush saves (in time points) |

        Use `build_pipeline_steps()` for common configurations, or construct
        the dict manually for full control (as shown in
        `basic_cellpose_cpmeasure_monozarr.py`).
        """
    )
    return


if __name__ == "__main__":
    app.run()
