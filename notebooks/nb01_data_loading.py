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
        # nb01 — Data Loading & Image Exploration

        Foundation notebook for the aliby pipeline. Demonstrates how to:

        1. **Locate test data** (local path or Zenodo download)
        2. **Discover datasets** using `DatasetDir`, `DatasetZarr`, and `dispatch_dataset`
        3. **Load images** using `ImageList` and `ImageZarr`
        4. **Visualize** loaded image data

        Later notebooks import `@app.function` helpers defined here.
        """
    )
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import numpy as np

    return Path, np


@app.cell
def _(mo):
    mo.md("""
    ## 1. Data Source

    Pick a **test dataset** from Zenodo or point to any folder with a matching regex.
    """)
    return


@app.function
def get_data_path(Path):
    """Return the path to the test dataset, downloading from Zenodo if needed."""
    import pooch

    local_path = Path("/datastore/alan/aliby/test_dataset/data/")
    if local_path.exists():
        return local_path

    marker = "aliby_tests/"
    files = pooch.retrieve(
        url="https://zenodo.org/api/records/19411429/files/aliby_test_dataset.tar.gz/content",
        known_hash="3a8b1b7b362f002098ba44e65622862057cfe46f0b459514bf270349c8bce4a7",
        fname="aliby_test_dataset.tar.gz",
        processor=pooch.Untar(extract_dir="aliby_tests"),
    )
    return Path(files[0].split(marker)[0] + marker)


@app.cell
def _(Path, mo):
    # Zenodo test datasets (downloaded on first use)
    test_data_path = get_data_path(Path)
    catalog_list = dataset_catalog()

    dataset_dropdown = mo.ui.dropdown(
        options={"(custom path)": None, **{d["name"]: d for d in catalog_list}},
        value=catalog_list[0]["name"],
        label="Test dataset (from Zenodo)",
    )
    dataset_dropdown  # noqa: B018
    return dataset_dropdown, test_data_path


@app.cell
def _(mo):
    mo.md("""
    ## 2. Dataset Discovery
    """)
    return


@app.function
def dataset_catalog():
    """Return the catalog of test datasets with their regex and capture order."""
    return [
        {
            "name": "crop_cellpainting_256",
            "regex": ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif",
            "capture_order": "WFC",
            "description": "Cell Painting (5-channel fluorescence, single timepoint)",
        },
        {
            "name": "crop_timeseries_alcatras_round_diff_dims_293",
            "regex": ".*/([^/]+)/.+_([0-9]{6})_([A-Za-z0-9]+)_(?:.*_)?([0-9]+).tif",
            "capture_order": "FTCZ",
            "description": "Time-series (round trap, different image dimensions)",
        },
        {
            "name": "crop_timeseries_alcatras_square_same_channels_293",
            "regex": ".*/([^/]+)/.+_([0-9]{6})_([A-Za-z0-9]+)_(?:.*_)?([0-9]+).tif",
            "capture_order": "FTCZ",
            "description": "Time-series (square trap, same channel dimensions)",
        },
    ]


@app.cell
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


@app.function
def load_dataset_dir(data_path, ds_info):
    """Load a DatasetDir from a dataset info dict and return positions."""
    from aliby.io.dataset import DatasetDir

    dset = DatasetDir(
        data_path / ds_info["name"],
        regex=ds_info["regex"],
        capture_order=ds_info["capture_order"],
    )
    positions = dset.get_position_ids()
    return dset, positions


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
        f"**Dataset:** `{dset.name}` — found **{len(positions)}** position(s)\n\n"
        + "\n".join(
            f"- `{p['key']}`: {len(p['path']) if isinstance(p['path'], list) else 1} file(s)"
            for p in positions[:5]
        )
    )
    return ds_info, positions


@app.cell
def _(mo):
    mo.md("""
    ## 3. Zarr Dataset Discovery
    """)
    return


@app.cell
def _(Path, ds_info, folder_input, mo):
    from aliby.io.dataset import DatasetZarr

    zarr_name = f"{ds_info['name']}.zarr"
    _folder = Path(folder_input.value)
    zarr_path = _folder.parent / zarr_name

    if zarr_path.exists():
        dset_zarr = DatasetZarr(zarr_path)
        zarr_positions = dset_zarr.get_position_ids()
        mo.md(
            f"**Zarr dataset:** `{zarr_name}` — **{len(zarr_positions)}** position(s)\n\n"
            + "\n".join(f"- key=`{p['key']}`" for p in zarr_positions[:5])
        )
    else:
        dset_zarr = None
        zarr_positions = []
        mo.md(f"Zarr dataset `{zarr_name}` not found — skipping.")
    return dset_zarr, zarr_positions


@app.cell
def _(mo):
    mo.md("""
    ## 4. Image Loading
    """)
    return


@app.function
def load_image_from_position(position, ds_info):
    """Load an image from a position dict, returns an Image object with lazy data."""
    from aliby.io.image import ImageList

    img = ImageList(
        source=position["path"],
        regex=ds_info["regex"],
        capture_order=ds_info["capture_order"],
    )
    return img


@app.cell
def _(ds_info, mo, np, positions):
    pos = positions[0]
    img = load_image_from_position(pos, ds_info)
    data = img.get_data_lazy()

    dims = dict(zip("TCZYX", data.shape))
    mo.md(
        f"### Image: `{img.name}`\n\n"
        f"- **Shape** (TCZYX): `{data.shape}`\n"
        f"- **Dtype**: `{data.dtype}`\n"
        f"- **Dimensions**: {', '.join(f'{k}={v}' for k, v in dims.items())}\n"
        f"- **Value range**: [{np.min(data)}, {np.max(data)}]"
    )
    return data, dims


@app.cell
def _(mo):
    mo.md("""
    ## 5. Image Visualization
    """)
    return


@app.cell
def _(data, dims, mo, np):
    import matplotlib.pyplot as plt

    t_idx = 0
    nch = dims["C"]
    ncols = min(nch, 4)
    nrows = (nch + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
    )
    for c in range(nch):
        ax = axes[c // ncols][c % ncols]
        # Max-project over Z for display
        img_2d = np.max(data[t_idx, c], axis=0)
        ax.imshow(img_2d, cmap="gray")
        ax.set_title(f"Channel {c}")
        ax.axis("off")

    # Hide unused axes
    for idx in range(nch, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"Timepoint {t_idx} — max-Z projection", fontsize=14)
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Zarr Image Loading
    """)
    return


@app.cell
def _(dset_zarr, mo, np, zarr_positions):
    if dset_zarr is not None and len(zarr_positions) > 0:
        from aliby.io.image import ImageZarr

        zarr_pos = zarr_positions[0]
        img_zarr = ImageZarr(source=zarr_pos)
        zarr_data = img_zarr.get_data_lazy()
        mo.md(
            f"### Zarr Image: `{img_zarr.name}`\n\n"
            f"- **Shape** (TCZYX): `{zarr_data.shape}`\n"
            f"- **Dtype**: `{zarr_data.dtype}`\n"
            f"- **Value range**: [{np.min(zarr_data)}, {np.max(zarr_data)}]"
        )
    else:
        mo.md("_No zarr data to display._")
    return


if __name__ == "__main__":
    app.run()
