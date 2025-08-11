# CLAUDE.md

## Project Overview

ALIBYlite (Analyser of Live-cell Imaging for Budding Yeast) is an end-to-end processing pipeline for cell microscopy time-lapses. It automates segmentation, tracking, lineage predictions and post-processing for live-cell imaging data.

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n alibylite python=3.11
conda activate alibylite

# Configure poetry to not create virtual environments
poetry config virtualenvs.create false

# Install dependencies
poetry install --with baby

# Optional: Install OMERO support
poetry install --with omero
# or manually:
conda install -c conda-forge zeroc-ice==3.6.5
pip install omero-py
```

### Code Formatting
```bash
# Format code with Black (configured for 79 character line length)
black src/
```

### No test framework detected 

## Architecture Overview

The codebase follows a modular pipeline architecture with five main processing steps:

1. **Tiling** (`aliby.tile`): Extracts individual cell traps from microscopy images
2. **Segmentation** (`aliby.baby_sitter`): Uses Baby-seg neural networks to segment cells
3. **Extraction** (`extraction.core`): Extracts quantitative features from segmented cells
4. **Post-processing** (`postprocessor.core`): Applies lineage tracking, merging, and quality control
5. **I/O and Data Management** (`agora.io`, `aliby.io`): Handles data input/output, metadata, and storage

### Main Entry Points
- `aliby.pipeline.Pipeline`: Main orchestration class that runs the complete pipeline
- `examples/run_local.py`: Example script showing local pipeline execution
- `examples/run_jura.py`: Example for OMERO server integration

### Key Components

**Pipeline Configuration**
- `aliby.global_settings.GlobalSettings`: Hard-coded parameters and imaging specifications
- `aliby.pipeline.PipelineParameters`: Configurable parameters for all pipeline steps
- Parameters are organized hierarchically: general, tiler, baby, extraction, postprocessing

**Data Flow**
- Input: Microscopy images (local files or OMERO server)
- Output: HDF5 files containing extracted features and metadata
- Intermediate: Tile images, segmentation masks, cell tracking data

**Core Processing Classes**
- `aliby.tile.tiler.Tiler`: Handles image tiling and trap detection
- `aliby.baby_sitter.BabyRunner`: Manages Baby-seg segmentation
- `extraction.core.extractor.Extractor`: Extracts quantitative measurements
- `postprocessor.core.postprocessing.PostProcessor`: Applies post-processing steps

### Dependencies and Integration
- **Baby-seg**: Neural network segmentation (tensorflow backend)
- **OMERO**: Optional integration for microscopy server access
- **HDF5/Zarr**: Data storage formats
- **Pandas/NumPy**: Data manipulation and analysis
- Uses Poetry for dependency management with optional groups (baby, omero)

### Configuration Patterns
- All processing steps inherit from `agora.abc.ProcessABC` and use corresponding `ParametersABC` classes
- Configuration uses nested dictionaries that can be recursively merged
- Global settings are defined in `aliby.global_settings` and include imaging specifications and default functions

### Data Storage
- Primary output format is HDF5 with structured datasets
- Metadata from microscopy logs is preserved
- Signal extraction results are organized by channel and processing function
- Writers in `agora.io.writers` handle structured data output
