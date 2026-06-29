"""
Shared test-data loader for aliby tests, examples, and notebooks.

A small public dataset hosted on Zenodo (record 19411429) covers every
input modality aliby supports: TIFF directories (single-timepoint Cell
Painting and multi-timepoint yeast), per-position zarr v3 groups (CYX for
Cell Painting, TCZYX for time-series), and the matching FTCZ filename
convention. The dataset is fetched once on first use via pooch and cached
under ``~/.cache/pooch/aliby_tests/``.

Public API
----------
- :data:`DATASETS` -- list of catalogue entries, one per sub-dataset.
- :func:`get_data_root` -- absolute path to the unpacked dataset root.
- :func:`get_dataset` -- look up a single catalogue entry by name.
- :func:`get_dataset_path` -- absolute path to a sub-dataset.

Catalogue entry schema
----------------------
- ``name`` -- subdirectory name under the dataset root.
- ``modality`` -- "cell_painting", "yeast_timelapse", or
  "cell_painting_zarr"/"yeast_timelapse_zarr" for the zarr variants.
- ``layout`` -- "tiff_dir" or "zarr".
- ``regex`` / ``capture_order`` -- aliby ``DatasetDir`` arguments
  (TIFF datasets only; ``None`` for zarr).
- ``channels`` -- channel-name → index mapping, when applicable.
- ``description`` -- human-readable summary.
"""

from __future__ import annotations

from pathlib import Path

ZENODO_URL = (
    "https://zenodo.org/api/records/19411429/files/"
    "aliby_test_dataset.tar.gz/content"
)
ZENODO_HASH = "3a8b1b7b362f002098ba44e65622862057cfe46f0b459514bf270349c8bce4a7"
DOWNLOAD_NAME = "aliby_test_dataset.tar.gz"
EXTRACT_DIR = "aliby_tests"
LEGACY_LOCAL_ROOT = Path("/datastore/alan/aliby/test_dataset/data/")


DATASETS: list[dict] = [
    {
        "name": "crop_cellpainting_256",
        "modality": "cell_painting",
        "layout": "tiff_dir",
        "regex": r".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+)\.tif",
        "capture_order": "WFC",
        "channels": {"DNA": 0, "ER": 1, "RNA": 2, "AGP": 3, "Mito": 4},
        "description": (
            "5-channel Cell Painting (DNA/ER/RNA/AGP/Mito), single-timepoint, "
            "256x256 crops. Standard pattern for the cellpose + cp_measure pipeline."
        ),
    },
    {
        "name": "crop_cellpainting_256.zarr",
        "modality": "cell_painting_zarr",
        "layout": "zarr",
        "regex": None,
        "capture_order": "CYX",
        "channels": {"DNA": 0, "ER": 1, "RNA": 2, "AGP": 3, "Mito": 4},
        "description": (
            "Same crops as crop_cellpainting_256 written as a zarr root with "
            "one (5, Y, X) group per position. Drives the deep-embeddings "
            "(nahual_embed_*) example."
        ),
    },
    {
        "name": "crop_timeseries_alcatras_round_diff_dims_293",
        "modality": "yeast_timelapse",
        "layout": "tiff_dir",
        "regex": r".*/([^/]+)/.+_([0-9]{6})_([A-Za-z0-9]+)_(?:.*_)?([0-9]+)\.tif",
        "capture_order": "FTCZ",
        "channels": None,
        "description": (
            "Yeast time-series in round alcatras traps, per-position TIFF "
            "subdirectories, three channels with z-stacks of varying dimensions."
        ),
    },
    {
        "name": "crop_timeseries_alcatras_square_same_channels_293",
        "modality": "yeast_timelapse",
        "layout": "tiff_dir",
        "regex": r".*/([^/]+)/.+_([0-9]{6})_([A-Za-z0-9]+)_(?:.*_)?([0-9]+)\.tif",
        "capture_order": "FTCZ",
        "channels": None,
        "description": (
            "Yeast time-series in square alcatras traps, per-position TIFF "
            "subdirectories. Same channel shape across positions."
        ),
    },
    {
        "name": "crop_timeseries_alcatras_square_same_channels_293.zarr",
        "modality": "yeast_timelapse_zarr",
        "layout": "zarr",
        "regex": None,
        "capture_order": "TCZYX",
        "channels": None,
        "description": (
            "Same content as the square-traps TIFF dataset, written as a zarr "
            "root with one (T, C, Z, Y, X) array per position. Drives the "
            "BABY time-lapse example without a .zgroup wrapper step."
        ),
    },
]


def _build_pooch_call() -> tuple[str, str, str, object]:
    """Return arguments for ``pooch.retrieve`` without importing pooch eagerly."""
    import pooch

    return (
        ZENODO_URL,
        ZENODO_HASH,
        DOWNLOAD_NAME,
        pooch.Untar(extract_dir=EXTRACT_DIR),
    )


def get_data_root() -> Path:
    """Return the absolute path to the unpacked dataset root.

    Honours the legacy on-disk copy at :data:`LEGACY_LOCAL_ROOT` when
    present (e.g. on the maintainer's workstation). Otherwise fetches the
    Zenodo tarball via pooch on first call and reuses the cache thereafter.
    """
    if LEGACY_LOCAL_ROOT.exists():
        return LEGACY_LOCAL_ROOT

    try:
        import pooch
    except ImportError as e:
        raise ImportError(
            "aliby.test_data requires `pooch` to download the Zenodo "
            "dataset. Install with `uv add pooch` or `pip install pooch`."
        ) from e

    url, known_hash, fname, processor = _build_pooch_call()
    files = pooch.retrieve(
        url=url,
        known_hash=known_hash,
        fname=fname,
        processor=processor,
    )
    return Path(files[0].split(EXTRACT_DIR + "/")[0]) / EXTRACT_DIR


def get_dataset(name: str) -> dict:
    """Look up a catalogue entry by ``name``."""
    for entry in DATASETS:
        if entry["name"] == name:
            return entry
    valid = ", ".join(d["name"] for d in DATASETS)
    raise KeyError(f"Unknown dataset {name!r}. Valid names: {valid}")


def get_dataset_path(name: str) -> Path:
    """Return the absolute path to a sub-dataset, fetching it if needed."""
    return get_data_root() / name
