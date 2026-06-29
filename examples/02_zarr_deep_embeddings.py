"""
Deep-learning embeddings: monozarr -> nahual_embed_* via Nahual IPC.

This example exercises the ``nahual_embed_*`` (arbitrary-embedder) terminal
pathway in ``aliby.pipe_core.get_profiles_from_state``. It mirrors the
publication-side pattern from ``afermg/JUMP_lite`` -- run a foundation-model
embedder (DINOv2, OpenPhenom, MorphEM, SubCell, ViT, ...) over every tile of
a monozarr Cell Painting dataset, with the model hosted in a separate
process and reached over Nahual's pynng IPC.

Default input is the bundled Zenodo Cell Painting monozarr
(``crop_cellpainting_256.zarr``, 540 KB), downloaded on first use by
``aliby.test_data``.

End-to-end execution against a real embedder requires a running Nahual
server (see Section 4 below). When no server is reachable the script stops
after building the pipeline dict and stubbing an ndarray through
``get_profiles_from_state`` to demonstrate the PR #20 fix path.
"""

from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pyarrow as pa

from aliby.io.dataset import dispatch_dataset
from aliby.pipe import run_pipeline_and_post
from aliby.pipe_core import configure_logging, get_profiles_from_state
from aliby.test_data import get_dataset, get_dataset_path

# ---------------------------------------------------------------------------
# 1. Choose input data.
# ---------------------------------------------------------------------------
# The bundled monozarr is the same content as crop_cellpainting_256 but
# written as a zarr root with one (C=5, Y=256, X=256) group per position.
ENTRY = get_dataset("crop_cellpainting_256.zarr")
MONOZARR_PATH = get_dataset_path(ENTRY["name"])
print(f"Monozarr root: {MONOZARR_PATH}")

dataset = dispatch_dataset(MONOZARR_PATH, is_zarr=True)
positions = dataset.get_position_ids()
print(f"Discovered {len(positions)} zarr groups: {[p['key'] for p in positions]}")
# Expected on the bundled fixture:
#   Discovered 1 zarr groups: ['source_3__CP_25_all_Phenix1__JCPQC016__A01__1__']

# ---------------------------------------------------------------------------
# 2. Pick an embedder. Each model is hosted in its own server repo and
#    bound to a pynng IPC socket. Start one socket per GPU you want to feed.
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    "dinov2": {
        "model_group": "dinov2",
        "tile_size": 224,
        "selected_channels": [0, 1, 2],
        "setup_params": {
            "repo_or_dir": "facebookresearch/dinov2",
            "model_name": "dinov2_vits14",
            "pretrained": True,
            "device": -1,  # -1 tells the server to round-robin its GPUs
        },
        "server": "github:afermg/dinov2",
    },
    "openphenom": {
        "model_group": "vit",
        "tile_size": 256,
        "selected_channels": [0, 1, 2, 3, 4],
        "setup_params": {"model_name": "openphenom", "device": -1},
        "server": "github:afermg/nahual_vit",
    },
    "morphem": {
        "model_group": "vit",
        "tile_size": 224,
        "selected_channels": [0, 1, 2, 3, 4],
        "setup_params": {"model_name": "morphem", "device": -1},
        "server": "github:afermg/nahual_vit",
    },
    "subcell": {
        "model_group": "subcell",
        "tile_size": 448,
        "selected_channels": [3, 2, 1, 0],
        "setup_params": {"model_name": "subcell", "device": -1},
        "server": "github:afermg/SubCellPortable",
    },
}

MODEL_NAME = "dinov2"
MODEL_CONFIG = deepcopy(MODEL_REGISTRY[MODEL_NAME])

# Replace with the address(es) of running Nahual servers. Round-robined
# across positions. Empty list -> live run is skipped (see Section 5).
ADDRESSES: list[str] = []  # e.g. ["ipc:///tmp/dinov2_0.ipc", "ipc:///tmp/dinov2_1.ipc"]


# ---------------------------------------------------------------------------
# 3. Per-position pipeline builder.
# ---------------------------------------------------------------------------
def build_embed_pipeline(
    position: dict,
    address: str,
    model_config: dict,
) -> dict:
    """Tile -> nahual_embed_* pipeline dict for a single zarr position."""
    step_name = f"nahual_embed_{model_config['model_group']}"
    embed_params: dict = {
        "address": address,
        "model_group": model_config["model_group"],
        "setup_params": model_config["setup_params"],
    }
    if model_config.get("selected_channels") is not None:
        embed_params["selected_channels"] = model_config["selected_channels"]

    return {
        "io": {"input_path": position, "capture_order": "CYX"},
        "ntps": 1,
        "steps": {
            "tile": {
                "kind": "crop",
                "tile_size": model_config["tile_size"],
                "calculate_drift": False,
                "image_kwargs": {"source": position, "capture_order": "CYX"},
            },
            step_name: embed_params,
        },
        "passed_data": {step_name: [("pixels", "tile", "data")]},
        "save": (),
        "save_interval": 1,
    }


def run_one_position(args: tuple) -> None:
    position, model_config, addresses, output_path, position_index = args
    address = addresses[position_index % len(addresses)]
    pipeline = build_embed_pipeline(position, address, model_config)
    configure_logging(output_path / "log.txt")
    run_pipeline_and_post(
        pipeline=pipeline,
        pipeline_name=position["key"],
        output_path=output_path,
        overwrite=False,
    )


# ---------------------------------------------------------------------------
# 4. Server-launch hint (run in a separate terminal!).
# ---------------------------------------------------------------------------
# Each model lives in its own repository. With Nix (handles CUDA/PyTorch):
#   nix run github:afermg/dinov2#nahual -- --address ipc:///tmp/dinov2_0.ipc
# Then set ADDRESSES = ["ipc:///tmp/dinov2_0.ipc"] above and re-run.

# ---------------------------------------------------------------------------
# 5. Either run the pipeline against the live server, or stub the embedder
#    output to demonstrate the PR #20 ndarray-branch fix path.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = build_embed_pipeline(
        positions[0],
        address=(ADDRESSES[0] if ADDRESSES else "ipc:///tmp/not_a_real_server.ipc"),
        model_config=MODEL_CONFIG,
    )
    embed_step = next(k for k in pipeline["steps"] if k.startswith("nahual_embed_"))
    print(f"Built pipeline with steps: {list(pipeline['steps'])}")
    print(f"Embed step: {embed_step}  tile_size={MODEL_CONFIG['tile_size']}  "
          f"selected_channels={MODEL_CONFIG['selected_channels']}")
    # Expected: Built pipeline with steps: ['tile', 'nahual_embed_dinov2']
    #           Embed step: nahual_embed_dinov2  tile_size=224  selected_channels=[0, 1, 2]

    if ADDRESSES:
        # Live run -- one process per position, round-robined across servers.
        output_dir = Path(mkdtemp(prefix="aliby_embeddings_out_"))
        print(f"Writing embeddings under {output_dir}")
        payloads = [
            (pos, MODEL_CONFIG, ADDRESSES, output_dir, i)
            for i, pos in enumerate(positions)
        ]
        with Pool(processes=min(len(payloads), len(ADDRESSES))) as pool:
            pool.map(run_one_position, payloads)
        # Per-position parquet files in <output_dir>/profiles/. Columns are
        # X_0 ... X_{D-1} for embedding dimension D (e.g. 384 for dinov2_vits14)
        # plus metadata_tile / metadata_label / metadata_object / metadata_tp.
    else:
        # No live server -- stub an embedding ndarray through the same
        # `get_profiles_from_state` codepath the live run would hit. This is
        # exactly the path PR #20 fixes: without the fix, the strict-zip in
        # `format_extraction` raises ValueError because the instructions side
        # was wrapped in an infinite `itertools.cycle`.
        stub_embedding = np.arange(12, dtype=np.float32).reshape(3, 4)
        state = {"data": {embed_step: [stub_embedding]}}
        profiles = get_profiles_from_state(state, pipeline)
        print(f"Stubbed profiles: {profiles.num_rows} rows, "
              f"columns={profiles.column_names}")
        assert isinstance(profiles, pa.Table) and profiles.num_rows > 0
        # Expected: Stubbed profiles: 3 rows,
        # columns=['tile', 'label', 'X_0', 'X_1', 'X_2', 'X_3',
        #          'metadata_object', 'metadata_tp']
