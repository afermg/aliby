# /// script
# requires-python = ">=3.11, <3.14"
# dependencies = [
#     "marimo>=0.21.1",
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
        # nb03 — Deep Learning Embeddings

        Demonstrates how to run **deep learning embedding pipelines** using
        [Nahual](https://github.com/afermg/nahual) as the inference backend.
        All available embedding models are supported:

        | Model | Server repo | model_group |
        |---|---|---|
        | **OpenPhenom** | [afermg/nahual_vit](https://github.com/afermg/nahual_vit) | `vit` |
        | **MorphEM** | [afermg/nahual_vit](https://github.com/afermg/nahual_vit) | `vit` |
        | **DINOv2** | [afermg/dinov2](https://github.com/afermg/dinov2) | `dinov2` |
        | **SubCell** | [afermg/SubCellPortable](https://github.com/afermg/SubCellPortable) | `subcell` |

        Based on `examples/deep_learning.py`, `examples/deep_learning_dinov2.py`,
        and the [Nahual examples](https://github.com/afermg/nahual/tree/main/examples).

        > **Prerequisite:** A Nahual model server must be running in a
        > separate process. See Section 4 below for setup instructions.
        """
    )
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import numpy as np

    return (Path,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Data Source

    Point to **any folder** of images with a matching regex, or use the
    built-in test dataset from nb01.
    """)
    return


@app.cell
def _(Path, mo):
    import nb01_data_loading as nb01

    # Zenodo test datasets (downloaded on first use)
    test_data_path = nb01.get_data_path(Path)
    catalog = nb01.dataset_catalog()

    dataset_dropdown = mo.ui.dropdown(
        options={"(custom path)": None, **{d["name"]: d for d in catalog}},
        value=catalog[0]["name"],
        label="Test dataset (from Zenodo)",
    )
    dataset_dropdown
    return dataset_dropdown, test_data_path


@app.cell(hide_code=True)
def _(dataset_dropdown, mo, test_data_path):
    # Pre-fill from test dataset or enter custom values
    _selected = dataset_dropdown.value
    if _selected is not None:
        _default_folder = str(test_data_path / _selected["name"])
        _default_regex = _selected["regex"]
        _default_capture = _selected["capture_order"]
    else:
        _default_folder = ""
        _default_regex = ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"
        _default_capture = "WFC"

    folder_input = mo.ui.text(
        value=_default_folder,
        label="Image folder",
        full_width=True,
    )
    regex_input = mo.ui.text(
        value=_default_regex,
        label="Filename regex (capture groups define position grouping)",
        full_width=True,
    )
    capture_order_input = mo.ui.text(
        value=_default_capture,
        label="Capture order (e.g. WFC, FTCZ, WTFZC)",
    )

    mo.vstack([folder_input, regex_input, capture_order_input])
    return capture_order_input, folder_input, regex_input


@app.cell
def _(Path, capture_order_input, folder_input, mo, regex_input):
    from aliby.io.dataset import DatasetDir

    _folder = Path(folder_input.value)
    ds_info = {
        "name": _folder.name,
        "regex": regex_input.value,
        "capture_order": capture_order_input.value,
    }

    dset = DatasetDir(
        _folder,
        regex=ds_info["regex"],
        capture_order=ds_info["capture_order"],
    )
    positions = dset.get_position_ids()

    mo.md(
        f"**Dataset:** `{ds_info['name']}` — **{len(positions)}** position(s)\n\n"
        + "\n".join(
            f"- `{p['key']}`: {len(p['path']) if isinstance(p['path'], list) else 1} file(s)"
            for p in positions[:5]
        )
    )
    return ds_info, positions


@app.cell
def _(mo):
    mo.md("""
    ## 2. Model Selection
    """)
    return


@app.cell
def _():
    MODEL_REGISTRY = {
        "openphenom": {
            "label": "OpenPhenom (recursionpharma/OpenPhenom)",
            "model_group": "vit",
            "setup_params": {"model_name": "recursionpharma/OpenPhenom"},
            "default_address": "ipc:///tmp/vit.ipc",
            "default_tile_size": 256,
            "selected_channels": None,
            "server_repo": "afermg/nahual_vit",
            "nix_run": "nix run github:afermg/nahual_vit -- ipc:///tmp/vit.ipc",
            "non_nix": "git clone https://github.com/afermg/nahual_vit.git && cd nahual_vit\nuv sync && uv run python src/vit/server.py ipc:///tmp/vit.ipc",
            "notes": "Accepts up to 6 channels, zero-pads if fewer are provided.",
        },
        "morphem": {
            "label": "MorphEM (CaicedoLab/MorphEm)",
            "model_group": "vit",
            "setup_params": {"model_name": "CaicedoLab/MorphEm"},
            "default_address": "ipc:///tmp/morphem.ipc",
            "default_tile_size": 256,
            "selected_channels": None,
            "server_repo": "afermg/nahual_vit",
            "nix_run": "nix run github:afermg/nahual_vit -- ipc:///tmp/morphem.ipc",
            "non_nix": "git clone https://github.com/afermg/nahual_vit.git && cd nahual_vit\nuv sync && uv run python src/vit/morphem.py ipc:///tmp/morphem.ipc",
            "notes": "Accepts up to 6 channels, zero-pads if fewer. Same server as OpenPhenom.",
        },
        "dinov2": {
            "label": "DINOv2 (facebookresearch/dinov2)",
            "model_group": "dinov2",
            "setup_params": {
                "repo_or_dir": "facebookresearch/dinov2",
                "model_name": "dinov2_vits14_lc",
            },
            "default_address": "ipc:///tmp/dinov2.ipc",
            "default_tile_size": 224,
            "selected_channels": [0, 1, 2],
            "server_repo": "afermg/dinov2",
            "nix_run": "nix run github:afermg/dinov2 -- ipc:///tmp/dinov2.ipc",
            "non_nix": "git clone https://github.com/afermg/dinov2.git && cd dinov2\nuv sync && uv run python server.py ipc:///tmp/dinov2.ipc",
            "notes": "Expects 3 RGB channels (use selected_channels). Tile size must be a multiple of 14.",
        },
        "subcell": {
            "label": "SubCell (SubCellPortable)",
            "model_group": "subcell",
            "setup_params": {
                "model_type": "mae_contrast_supcon_model",
                "model_channels": "rybg",
            },
            "default_address": "ipc:///tmp/subcell.ipc",
            "default_tile_size": 256,
            "selected_channels": [0, 1, 2, 3],
            "server_repo": "afermg/SubCellPortable",
            "nix_run": "nix run github:afermg/subcellportable -- ipc:///tmp/subcell.ipc",
            "non_nix": "git clone https://github.com/afermg/SubCellPortable.git && cd SubCellPortable\nuv sync && uv run python server.py ipc:///tmp/subcell.ipc",
            "notes": "Expects 4 channels (RYBG order). Encodes single-cell morphology and protein localisation.",
        },
    }
    return (MODEL_REGISTRY,)


@app.cell
def _(MODEL_REGISTRY, mo):
    model_selector = mo.ui.dropdown(
        options={v["label"]: k for k, v in MODEL_REGISTRY.items()},
        value=MODEL_REGISTRY["openphenom"]["label"],
        label="Embedding model",
    )
    model_selector
    return (model_selector,)


@app.cell
def _(MODEL_REGISTRY, mo, model_selector):
    _model = MODEL_REGISTRY[model_selector.value]

    address_input = mo.ui.text(
        value=_model["default_address"],
        label="Nahual server address",
    )
    tile_size_input = mo.ui.slider(
        start=128, stop=512, step=64,
        value=_model["default_tile_size"],
        label="Tile size",
    )

    mo.vstack([address_input, tile_size_input])
    return address_input, tile_size_input


@app.cell
def _(mo):
    mo.md("""
    ## 3. Pipeline Configuration
    """)
    return


@app.function
def build_embed_pipeline(input_path, address, tile_size, ds_info, model_config):
    """Build an embedding pipeline for any Nahual model.

    Parameters
    ----------
    input_path : list or str
        Image source (from dataset position).
    address : str
        IPC address of the Nahual server.
    tile_size : int
        Size of image tiles to feed to the model.
    ds_info : dict
        Dataset info with capture_order and regex.
    model_config : dict
        Entry from MODEL_REGISTRY with model_group, setup_params, etc.
    """
    fluo_base_config = {
        "input_path": input_path,
        "image_kwargs": {
            "capture_order": ds_info["capture_order"],
            "regex": ds_info["regex"],
        },
        "ntps": 1,
        "tile": {
            "kind": "crop",
            "tile_size": tile_size,
            "calculate_drift": False,
        },
    }

    embed_params = dict(
        address=address,
        model_group=model_config["model_group"],
        setup_params=model_config["setup_params"],
    )
    if model_config["selected_channels"] is not None:
        embed_params["selected_channels"] = model_config["selected_channels"]

    step_name = f"nahual_embed_{model_config['model_group']}"

    return {
        "io": {**fluo_base_config},
        "steps": dict(
            tile=dict(
                **fluo_base_config["tile"],
                image_kwargs=dict(
                    source=input_path,
                    **fluo_base_config["image_kwargs"],
                ),
            ),
            **{step_name: embed_params},
        ),
        "passed_data": {step_name: [("pixels", "tile", "data")]},
        "save": (),
        "save_interval": 1,
    }


@app.cell
def _(
    MODEL_REGISTRY,
    address_input,
    ds_info,
    mo,
    model_selector,
    positions,
    tile_size_input,
):
    pos = positions[0]
    input_path = pos["path"]

    model_key = model_selector.value
    model_config = MODEL_REGISTRY[model_key]
    address = address_input.value
    tile_size = tile_size_input.value

    dl_pipeline = build_embed_pipeline(
        input_path, address, tile_size, ds_info, model_config,
    )
    embed_step = [s for s in dl_pipeline["steps"] if s.startswith("nahual_")][0]

    mo.md(
        f"### Pipeline for **{model_config['label']}**\n\n"
        f"- **model_group:** `{model_config['model_group']}`\n"
        f"- **Address:** `{address}`\n"
        f"- **Tile size:** {tile_size}\n"
        f"- **Embed step:** `{embed_step}`\n"
        f"- **selected_channels:** {model_config['selected_channels']}\n"
        f"- **Steps:** {list(dl_pipeline['steps'].keys())}"
    )
    return (model_config,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Starting a Nahual Server
    """)
    return


@app.cell
def _(mo, model_config):
    _lines = [
        "**A Nahual model server must be running before executing the "
        "pipeline.** Each model lives in its own repository with its own "
        "dependencies. Nahual uses IPC sockets (via pynng) so the server "
        "and client can run in completely different Python environments.",
        "",
        f"### {model_config['label']}",
        "",
        f"**Server repo:** [{model_config['server_repo']}]"
        f"(https://github.com/{model_config['server_repo']})",
        "",
        "**With Nix** (recommended — handles all CUDA/PyTorch deps):",
        f"```bash\n{model_config['nix_run']}\n```",
        "",
        "**Without Nix:**",
        f"```bash\n{model_config['non_nix']}\n```",
        "",
        f"_{model_config['notes']}_",
        "",
        "Press `Ctrl-C` in the server terminal to stop it.",
    ]
    mo.callout(mo.md("\n".join(_lines)), kind="warn")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Run Embedding Pipeline
    """)
    return


@app.cell
def _(mo):
    # Uncomment to run (requires a Nahual server):
    #
    # import tempfile
    # from aliby.pipe import run_pipeline_and_post
    #
    # with tempfile.TemporaryDirectory() as tmp_dir:
    #     profiles, post = run_pipeline_and_post(
    #         pipeline=dl_pipeline,
    #         pipeline_name=pos["key"],
    #         output_path=Path(tmp_dir),
    #     )
    #
    # if profiles is not None:
    #     _df = profiles.to_pandas()
    #     mo.output.replace(mo.ui.table(_df.head(20)))

    mo.md("""
    _Pipeline execution is commented out — uncomment after starting a Nahual server._
    """)
    return


@app.cell
def _(MODEL_REGISTRY, mo):
    _rows = []
    for _k, _m in MODEL_REGISTRY.items():
        _ch = str(_m["selected_channels"]) if _m["selected_channels"] else "All"
        _rows.append(
            f"| **{_m['label'].split(' (')[0]}** "
            f"| [{_m['server_repo']}](https://github.com/{_m['server_repo']}) "
            f"| `\"{_m['model_group']}\"` "
            f"| {_ch} "
            f"| {_m['default_tile_size']} "
            f"| `{_m['default_address']}` |"
        )

    mo.md(
        "## 6. All Embedding Models\n\n"
        "| Model | Server repo | model_group | Input channels | Tile size | Default IPC |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(_rows)
        + "\n\n"
        "All pipelines follow the same pattern:\n"
        "1. **Tile** the input image into crops of the configured size\n"
        "2. **Send pixel data** to the Nahual server via IPC\n"
        "3. **Receive** per-tile embedding vectors as profiles\n\n"
        "`model_group` must match the server type — this is how "
        "`nahual.process.dispatch_setup_process` selects the correct "
        "serialization protocol.\n\n"
        "OpenPhenom and MorphEM share the same server repo "
        "([nahual_vit](https://github.com/afermg/nahual_vit)) "
        "but use different HuggingFace model weights."
    )
    return


if __name__ == "__main__":
    app.run()
