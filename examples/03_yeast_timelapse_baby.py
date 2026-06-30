"""
Yeast time-lapse profiling: zarr -> BABY -> intensity/sizeshape + tracking.

This example consolidates the pattern used by ``afermg/aliby_baby_processing``
to process Zenodo-deposited yeast micro-colony time-lapses with the BABY
segmenter, run remotely via a Nahual baby-phone server.

Default input is the bundled Zenodo yeast time-series monozarr
(``crop_timeseries_alcatras_square_same_channels_293.zarr``, ~5.7 MB).
Each per-position group inside the zarr is itself a TCZYX array; the
``wrap_zarr_as_group`` helper ensures aliby's ``ImageZarr`` opens it via
the expected ``(group_path, key)`` shape (mirrors the wrapping
aliby_baby_processing patches in for Zenodo deposits without a root
``.zgroup``).

End-to-end execution requires a Nahual baby-phone server (see Section 4).
When the server isn't reachable the script stops after building the pipeline
dict and reports its shape.
"""

import json
from pathlib import Path

import zarr

from aliby.io.dataset import DatasetZarr
from aliby.pipe_builder_baby import build_pipeline_steps
from aliby.test_data import get_dataset, get_dataset_path

# ---------------------------------------------------------------------------
# 1. Choose input data.
# ---------------------------------------------------------------------------
# A TCZYX zarr root with one (T, C, Z, Y, X) array per yeast position.
ENTRY = get_dataset("crop_timeseries_alcatras_square_same_channels_293.zarr")
ROOT_PATH = get_dataset_path(ENTRY["name"])
print(f"Zarr root: {ROOT_PATH}")

dataset = DatasetZarr(ROOT_PATH)
positions = dataset.get_position_ids()
print(
    f"Discovered {len(positions)} per-position zarr arrays: "
    f"{[p['key'] for p in positions]}"
)
# Expected: Discovered 2 per-position zarr arrays: ['PDR5_GFP_001', 'PDR5_GFP_002']
POSITION_PATH = Path(positions[0]["path"]) / positions[0]["key"]
print(f"Working on: {POSITION_PATH}")

# Each per-position group contains a single ``dataset`` array of shape
# (T, C, Z, Y, X) uint16. For the fixture: T=4, C=3, Z=3, Y=293, X=293.
arr_shape = zarr.open(str(POSITION_PATH), mode="r")["dataset"].shape
print(f"Position array shape: {arr_shape}  (T, C, Z, Y, X)")


# ---------------------------------------------------------------------------
# 2. zarr-v2 group wrapper for arrays that ship without a parent .zgroup.
# ---------------------------------------------------------------------------
def wrap_zarr_as_group(zarr_path: Path) -> tuple[Path, str]:
    """Synthesise a ``.zgroup`` next to the array, return (group_path, key)."""
    parent = zarr_path.parent
    zgroup = parent / ".zgroup"
    if not zgroup.exists():
        zgroup.write_text(json.dumps({"zarr_format": 2}))
    return parent, zarr_path.name


GROUP_PATH, KEY = wrap_zarr_as_group(POSITION_PATH)
print(f"Wrapped: group_path={GROUP_PATH}  key={KEY}")
# Expected: Wrapped: group_path=<ROOT_PATH>  key=PDR5_GFP_001


# ---------------------------------------------------------------------------
# 3. BABY pipeline definition.
# ---------------------------------------------------------------------------
# BABY runs in a separate process behind a baby-phone Nahual server.
# `baby_address` and `baby_modelset` are both required by
# `pipe_builder_baby.build_pipeline_steps`.
BABY_ADDRESS = "ipc:///tmp/baby.ipc"  # Replace if your server differs
BABY_MODELSET = "yeast_alcatras_brightfield_60x_5z"  # See BABY docs for the catalogue
REF_CHANNEL = 0
REF_Z = 0
TILE_SIZE = 117
NTPS = min(2, arr_shape[0])

# ``parallel=false`` plays nicely with joblib loky at the outer per-site
# level (matches the override aliby_baby_processing uses).
BABY_EXTRA_ARGS: list[tuple[str, tuple[str, str]]] = [
    ("refine_outlines", ("", "true")),
    ("with_edgemasks", ("", "true")),
    ("with_masks", ("", "true")),
    ("assign_mothers", ("", "true")),
    ("parallel", ("", "false")),
]


def build_pipeline(
    zarr_path: Path,
    baby_address: str,
    baby_modelset: str,
    ntps: int,
    tile_size: int = TILE_SIZE,
    ref_channel: int = REF_CHANNEL,
    ref_z: int = REF_Z,
) -> dict:
    group_path, key = wrap_zarr_as_group(zarr_path)
    pipeline = build_pipeline_steps(
        baby_address=baby_address,
        baby_modelset=baby_modelset,
        channels_to_segment={"cell": ref_channel},
        features_to_extract=("intensity", "sizeshape"),
    )
    pipeline["ntps"] = ntps
    pipeline["steps"]["tile"].update(
        tile_size=tile_size,
        ref_channel=ref_channel,
        ref_z=ref_z,
        image_kwargs={
            "source": {"path": str(group_path), "key": key},
            "capture_order": "TCZYX",
            "dimorder": "TCZYX",
        },
    )
    pipeline["steps"]["segment_cell"]["segmenter_kwargs"]["extra_args"] = (
        BABY_EXTRA_ARGS
    )
    # pipe_builder_baby leaves passed_methods empty (BABY pulls pixels via
    # its own tiler). Re-wire the cellpose-style accessor so the segmenter
    # receives the per-timepoint stack.
    pipeline["passed_methods"] = {"segment_cell": ("tile", "get_fczyx")}
    return pipeline


# ---------------------------------------------------------------------------
# 4. Server-launch hint (run in a separate terminal!).
# ---------------------------------------------------------------------------
# nix run github:afermg/nahual_baby_phone -- --address ipc:///tmp/baby.ipc

# ---------------------------------------------------------------------------
# 5. Build the pipeline; run end-to-end only if a baby-phone server is up.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = build_pipeline(
        POSITION_PATH,
        baby_address=BABY_ADDRESS,
        baby_modelset=BABY_MODELSET,
        ntps=NTPS,
    )
    print(f"Pipeline steps: {list(pipeline['steps'])}")
    print(
        f"ntps={pipeline['ntps']}  "
        f"segmenter={pipeline['steps']['segment_cell']['segmenter_kwargs']['kind']}  "
        f"baby_address={pipeline['steps']['segment_cell']['segmenter_kwargs']['address']}"
    )
    # Expected:
    #   Pipeline steps: ['tile', 'segment_cell', 'extract_cell']
    #   ntps=2  segmenter=nahual_baby  baby_address=ipc:///tmp/baby.ipc

    # The line below is what actually launches BABY -- left commented because
    # it requires a running nahual_baby_phone server. Uncomment after starting
    # one (see Section 4).
    #
    # output_dir = Path(mkdtemp(prefix="aliby_baby_out_"))
    # print(f"Writing BABY outputs under {output_dir}")
    # configure_logging(output_dir / "log.txt")
    # run_pipeline_and_post(
    #     pipeline=pipeline,
    #     pipeline_name=POSITION_PATH.stem,
    #     output_path=output_dir,
    #     overwrite=True,
    # )
    # Outputs:
    #   <output_dir>/profiles/<position>.parquet  -- per-cell intensity + sizeshape
    #   <output_dir>/tracking/<position>_segment_cell.parquet
    #     -- BABY tracking/lineage (mother-daughter assignments per timepoint)
