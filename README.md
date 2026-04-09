# ALIBY: End-to-end processing for high throughput microscopy

ALIBY (pronounced alib-bee) orchestrates various tools for processing large-scale imaging data. It supports tasks like segmentation (e.g., Cellpose), feature extraction, and deep learning model deployment.

## Main features

- End-to-end: From a folder of images into morphological profiles
- Multiple modalities: Time-series, Fluorescence (Cell Painting, Fluorophores), 2D and 3D.
- Interpretable or Deep Learning: Leverages [cp_measure](https://github.com/afermg/cp_measure?tab=readme-ov-file#bulk-api) for features or [nahual](https://github.com/afermg/nahual) for deep learning embeddings
- Local-first, but supports distributed: GPU-intensive models can run in a different server from the one loading the data
- Zarr support: Data loader for Zarr datasets, supporting compressed datasets
- Standard output: Single-object (i.e., cell, nuclei) profiles are stored as parquet files with

## Quick Start: Basic Pipeline for Local TIFFs

This example demonstrates how to run a segmentation (via [Cellpose](https://github.com/MouseLand/cellpose)) and feature extraction (via [cp_measure](https://github.com/afermg/cp_measure?tab=readme-ov-file#bulk-api)) pipeline on a local dataset of TIF files.

```python
from pathlib import Path
from aliby.io.dataset import DatasetDir
from aliby.pipe import run_pipeline_and_post
from aliby.pipe_builder import build_pipeline_steps

# 1. Setup paths and data identification
input_path = Path("data/my_experiment")
# Regex to capture experimental metadata from filenames
# example filename: `testdir/testimage__A01__1__DNA.tif` -> A01, 1, DNA
regex = ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"
capture_order = "WFC"  # e.g., Well, Field-of-view, Channel

# 2. Identify positions in the dataset
dataset = DatasetDir(input_path, regex=regex, capture_order=capture_order)
positions = datas.get_position_ids() # This will fail if the regex doesn't capture any image groups

# Take the first position for this example
key, path = positions[0]["key"], positions[0]["path"]

# 3. Build pipeline steps
# Define which channels to segment and which to extract features from
pipeline = build_pipeline_steps(
    channels_to_segment={"nuclei": 1},
    channels_to_extract=[0, 1], # If the channels DNA, RNA, etc. they will be alphabetically sorted 
    features_to_extract=["intensity", "sizeshape"], # cp_measure features
)

# 4. Tell the "tile" step which set of images to read
pipeline["steps"]["tile"]["image_kwargs"] = {
    "source": {"key": key, "path": path},
    "regex": regex,
    "capture_order": capture_order,
}

# 5. Run the pipeline
result = run_pipeline_and_post(
    pipeline=pipeline, 
    pipeline_name=key, 
    output_path="results"
)
```

## Installation

We recommend using [uv](https://github.com/astral-sh/uv):

```bash
git clone git@github.com:afermg/aliby.git
cd aliby
uv sync --all-groups
```

### Nix setup

To run in a self-contained shell:

```bash
nix develop .
```

## Notes

### Terms and assumptions
Different subfields renamed existing concepts. Here is a list of names and details to make things less confusing.

- **Position/Field of View/Site**: A set of images obtained in the same location, comprised of all available time points, z-stacks, and channels taken in that exact place.
- **Time point/Frame**: In this context, "frame" is short for time-frame. 
- **regex**: Regular expression; the pattern-matching logic ALIBY uses to map a list of filenames into an array representing the images using capture groups (patterns inside `()` brackets).
- **capture_order**: The order in which the capture groups appear in the filename, represented as a string composed of a subset of `TCZ` and other non-`YX` letters.
- **Dimensions**: Any list of images (or zarr array) will be converted to a 5-D array internally: `TCZYX`. T (Time), C (Channel), Z (z-stack), Y (Y-dimension), and X (X-dimension).
- **Other capture groups**: These are not encoded but are often used to group images when stored in a single folder, such as W (Well), F (Field-of-View/Site), and P (Plate).
- **Cell Painting**: An experimental assay in which cells are fixed and dyed with five different compounds to highlight different sets of organelles. This is usually converted into vectors that characterize the cell state to evaluate the effect of drugs at scale.

### Details about the project itself
- Due to a recent overhaul of ALIBY, detailed documentation is under construction.
- To keep the dependency tree as small as possible is mostly considered feature-complete, any further processing will be done in a separate library.
- ALIBY stands for (Analyser of Live-Cell Imaging for Budding Yeast), since it started in 2021 as a tool to quantify cell signalling in high throughput time series experiments. Over the years its scope increased to support Cell Painting assays on mammalian cells, growing into a generalist method for end-to-end processing of microscopy imaging.
