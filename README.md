# ALIBY

**End-to-end processing for high-throughput microscopy.**

ALIBY (pronounced _alib-bee_) orchestrates various tools for processing large-scale imaging data. It quantifies microscopy experiments, fluorescence and/or time series, through segmentation (e.g., Cellpose) and feature extraction or via deep learning models.

## Main Features

- **End-to-End:** Extract morphological profiles from a set of images.
- **Multimodal:** Handles Time-series, Fluorescence (Cell Painting, Fluorophores), and both 2D and 3D data.
- **Interpretable & Deep Learning:** Leverages [cp_measure](https://github.com/afermg/cp_measure) for classical features or [nahual](https://github.com/afermg/nahual) for deep learning embeddings.
- **Local-First, but also Distributed:** Can be run on a single machine, but also supports GPU-based processing across machines.
- **Standardized Output:** Object-level profiles (e.g., cells, nuclei) stored as efficient Parquet files.
- **Zarr support:** Zarr support for compressed, scalable datasets.

## Installation

### Using `pip`

```bash
pip install aliby
```

### Using `uv`

For development installations, we recommend [uv](https://github.com/astral-sh/uv).

```bash
git clone git@github.com:afermg/aliby.git
cd aliby
uv sync --all-groups
```

### Nix

For a fully reproducible, self-contained shell:

```bash
nix develop .
```

## Quick Start: Basic Pipeline for Local TIFFs

Process a local dataset of TIFF files using [Cellpose](https://github.com/MouseLand/cellpose) for segmentation and [cp_measure](https://github.com/afermg/cp_measure) for feature extraction.

```python
from pathlib import Path
from aliby.io.dataset import DatasetDir
from aliby.pipe import run_pipeline_and_post
from aliby.pipe_builder import build_pipeline_steps

# 1. Setup paths and metadata identification
input_path = Path("data/my_experiment")

# Regex to capture Well, Field-of-view, and Channel from filenames
# example: `testdir/testimage__A01__1__DNA.tif` -> A01, 1, DNA
regex = r".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"
capture_order = "WFC" 

# 2. Identify positions in the dataset
dataset = DatasetDir(input_path, regex=regex, capture_order=capture_order)
positions = dataset.get_position_ids()

# Take the first position for this example
key, path = positions[0]["key"], positions[0]["path"]

# 3. Build pipeline steps
pipeline = build_pipeline_steps(
    channels_to_segment={"nuclei": 1},
    channels_to_extract=[0, 1], 
    features_to_extract=["intensity", "sizeshape"], 
)

# 4. Configure the tiling step
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

_See the [examples/](examples/) directory for more advanced use cases.

## Core Concepts & Glossary

Microscopy terminology can vary. Here’s how ALIBY defines these concepts:

- **Position / Field of View (FOV) / Site**: A set of images (all time points, z-stacks, channels) captured at a single physical location.
- **Time point / Frame**: A single moment in a time-series experiment.
- **Regex & Capture Groups**: ALIBY uses Regular Expressions to map filenames into internal arrays. Capture groups `()` in your regex define metadata like Well (W), Channel (C), or Time (T).
- **Capture Order**: The order of metadata groups in your regex (e.g., `"WFC"`).
- **TCZYX Dimensions**: Internally, all data is handled as a 5-D array: **T**ime, **C**hannel, **Z**-stack, **Y**, and **X**.
- **Cell Painting**: A high-content screening assay using five dyes to highlight various organelles, typically used to generate phenotypic vectors.

### About the project

- Under Construction:Detailed documentation is currently being
- We aim to keep the dependency tree minimal. Further downstream processing is often relegated to specialized libraries.
- ALIBY (_Analyser of Live-Cell Imaging for Budding Yeast_) originated in [2021](https://gitlab.com/aliby/aliby) at the [Swain Lab](https://swainlab.bio.ed.ac.uk/) to quantify yeast signalling. It has since evolved into a general-purpose tool for high-throughput imaging, including mammalian Cell Painting assays.
