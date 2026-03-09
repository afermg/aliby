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

The codebase follows a modular pipeline architecture with main processing steps executed for each microscopy position and time point:

1. **Tiling** (`aliby.tile`): Extracts individual cell traps from microscopy images
2. **Segmentation** (`aliby.baby_sitter`): Uses Baby-seg neural networks to segment cells
3. **Extraction** (`extraction.core`): Extracts quantitative features from segmented cells
4. **Post-processing** (`postprocessor.core`): Applies lineage tracking, merging, and quality control
5. **I/O and Data Management** (`agora.io`, `aliby.io`): Handles data input/output, metadata, and storage

### Data Concepts

- **Position**: A microscope location during an experiment, potentially with different strains ("groups")
- **Tile/Trap**: Individual cell regions extracted from each position's images (terms used interchangeably)
- **Time Point**: Sequential images captured at each position throughout the experiment
- **Cells vs Signal**: Raw cell data per tile vs processed data aggregated per position
- **Record/Kymograph**: Cell tracking data over time (terms used interchangeably)
- **Bud/Daughter**: Newly divided cells (terms used interchangeably)

### Pipeline Workflow

The main pipeline (`aliby.pipeline.Pipeline`) orchestrates processing through:
1. **run()**: Loops through all positions
2. **run_one_position()**: Processes all time points for a position
3. **_run_tp()**: Each step processes one time point (wrapped by StepABC as run_tp with timing)

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
- `aliby.tile.tiler.Tiler`: Tiles images into regions of interest (one per trap), ignoring tiles without cells
- `aliby.baby_sitter.BabyRunner`: Interfaces with Baby-seg to return cell masks, mother-bud pairs, and tracking data
- `extraction.core.extractor.Extractor`: Extracts areas, volumes, and fluorescence data using cell masks (writes directly to HDF5)
- `postprocessor.core.postprocessing.PostProcessor`: Applies cell picking, track merging, and runs processes like budding analysis

**Data Access Classes**
- `agora.cells`: Accesses cell information and masks from HDF5 files (lazy loading)
- `agora.signal`: Gets extracted properties for all cells/timepoints from HDF5 (used in postprocessing)
- `agora.bridge`: Interface layer for HDF5 file operations
- `postprocessor.grouper`: Concatenates signals across positions to generate experiment-wide dataframes

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
- Primary output format is HDF5 with structured datasets (one file per position)
- Metadata from microscopy logs is preserved
- Signal extraction results are organized by channel and processing function
- Writers in `agora.io.writers` handle structured data output
- Cell properties stored as nested dictionaries: `{'general': {'None': ['area', 'volume', 'eccentricity']}}`
- Picker and merger choices are written to HDF5 for reproducibility

### Extraction Functions

**Cell Functions** (`extraction.core.functions.cell_functions`)
- Standard cell measurements: area, median fluorescence, eccentricity
- Multi-channel fluorescence functions

**Background Functions** (`extraction.core.functions.background_functions`)
- Per-tile background statistics (median, mean, std of pixels outside all cells)
- Used for background channels such as Cy5; results are stored with `cell_label = -1`
- Loaded by `load_background_functions()` in `loaders.py`

**Distributors** (`extraction.core.functions.distributors`)
- Collapse multiple z-sections to 2D images

**Defaults** (`extraction.core.functions.defaults`)
- Standard fluorescence signals and metrics via `aliby.global_settings`

### Extractor internals

`Extractor.extract_single_channel_functions` drives per-channel extraction and delegates to two private helpers:
- `_extract_channel`: raw image + optional `_bgsub` variant for one channel
- `_extract_intracellular`: vacuole/cytoplasm sub-mask extraction for one channel

`Extractor.extract_multichannel_functions` handles metrics that combine multiple channels.

Both call `reduce_extract` → `apply_extraction_functions` → `apply_extraction_function`.
`apply_extraction_function` checks `self.cell_fun_names` to decide whether to index results
per cell `(trap_id, cell_label)` or per trap `trap_id` (for background functions).

Background channels (Cy5) skip `_bgsub` extraction entirely — subtracting the background
from itself would produce near-zero signals.

### Vacuole Identification

Vacuoles (liquid-filled compartments) are detected using a U-net CNN (`VacuoleIdentifier` from the optional `maby` package) applied to brightfield images. Detection splits cell masks into vacuole and cytoplasm sub-regions, enabling separate extraction of fluorescence metrics for each compartment (e.g. `GFP_vacuole`, `GFP_cytoplasm`).

**Key files:**
- `extraction/core/extractor.py`: `ExtractorParameters.identify_vacuoles` flag (default `True`); `compute_intracellular_masks()` generates sub-masks per trap; `extract_single_channel_functions()` runs extraction on full-cell, vacuole, and cytoplasm masks
- `extraction/core/functions/cell_functions.py`: `identify_vacuole()` calls the CNN; `_get_model()` lazy-loads and caches models; `is_model_available()` checks for optional dependencies
- `extraction/core/functions/loaders.py`: filters out functions whose model package is not installed

**Integration:**
- Enabled by default for all fluorescence channels except Cy5
- Requires the `maby` package; gracefully disabled (with a logged warning) if not installed or if `identify_vacuoles=False`
- Brightfield ("mean" projection) is used as input to the vacuole CNN
- Cells with no detected vacuole receive empty sub-masks so extraction still runs

### Post-processing Components

**Picker** (`postprocessor.core.reshapers.picker`)
- Selects cells with lineage information and minimum track length (default: 3+ timepoints)
- Identifies mother-bud relationships using Baby's lineage data

**Merger** (`postprocessor.core.reshapers.merger`)
- Combines fragmented tracks that should represent the same cell

**Process Functions**
- `buddings`: Analyzes cell division events
- `bud_metric`: Calculates bud-specific measurements
- Applied to signals like volume to generate derived measurements

### Logging

- Use `logging.getLogger("aliby").warning(...)` throughout — never bare `print()` for warnings
- The aliby logger (with timestamp formatter) is configured in `pipeline.py:_setup_logging`, which runs *after* `MetaData.__init__`; warnings emitted before that point appear without timestamps
- `parse_microscopy_logs` is called from multiple places per pipeline run (`MetaData.__init__` and `BaseLocalImage.set_meta`); deduplication flags must be module-level state, not reset inside `parse_microscopy_logs`
- `logger.propagate = False` is set in `_setup_logging` to prevent duplicate output when running inside IPython/Jupyter (where the root logger already has a handler)

### OMERO Integration

**Image Handling** (`aliby.io.omero`)
- Extracts Image objects from OMERO image IDs
- Extracts Dataset objects from OMERO experiment IDs

**Preflight and metadata synthesis** (`aliby.io.omero.Dataset`, `aliby.pipeline`)
- `Pipeline.setup()` connects to OMERO inside the `with dispatcher as conn:` block and logs a summary (positions, timepoints, channels) — all OMERO calls must stay inside that block as the connection closes on exit
- `Dataset.get_minimal_meta()` synthesises the three required metadata fields (`channels`, `time_settings/ntimepoints`, `time_settings/timeinterval`) from the first image when no log files are attached to the dataset; returns `None` if the dataset has no images
- `Dataset.cache_logs()` catches `FileNotFoundError` (no annotations) and logs a warning rather than raising, so the pipeline continues even for datasets with no attached files
- `Dataset.get_channels()` returns `[]` for empty datasets (never raises `UnboundLocalError`)
- `listChildren()` on a Dataset returns raw `ImageI` model objects, not gateway wrappers — use `getPixels(0)` not `getPixels()` when accessing pixels on those objects
- `MetaData.__init__` takes `omero_meta` (dict) instead of `OMERO_channels` (list); OMERO channel order is always authoritative and overrides log-file channel order; raises `FileNotFoundError` for local sources with no log files and no supplied metadata
- `PipelineParameters.default()` raises `ValueError` early if `get_minimal_meta()` returns `None` (dataset has no images), before attempting to build pipeline parameters
