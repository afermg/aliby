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
- Tiles with no cells are *not* skipped: a tile with no cells is entirely background.
  Skipping them left gaps that became NaN in `Signal.dataset_to_df`'s pivot
- All three take an optional `exclude_mask` for the PDMS trap (see below)

### PDMS masking

The PDMS trap lies outside the cells and so is counted as background unless it is
excluded. In principle `median_background` is robust to it while the PDMS covers
under half the non-cell pixels, whereas `mean_background` and `std_background`
are not.

**How much this matters depends strongly on the channel.**

*GFP* — measured on four positions across experiments 3451 and 2179: the trap
covers 15.4-15.7% of a tile and is only **1.02-1.04x** as bright as the medium.
Masking shifts the median and mean by under 1% and moves the std by -2% to +7%.
Barely worth having.

*Cy5* — the dye **binds to the PDMS**, so the pillars grow brighter as dye
accumulates. Inferred from experiment 4802's stored median, mean and std
backgrounds together with the 15.5% coverage measured from brightfield: the
relative std of the Cy5 background rises from **0.06** early in the ramp to
**0.31** late, reaching **0.40-0.65** in chambers 1704 and 1705, implying pillars
of roughly **1.9-2.4x** the medium. This is not caused by unsegmented cells: within
each chamber the relative std is flat against the number of cells in a trap
(correlations +0.10, -0.11, -0.01), and the chamber with the *most* cells (1352,
5.6 per trap) has the *least* heterogeneity (0.07) and also the lowest dye.

So for Cy5, `mean_background` and `std_background` are materially wrong late in an
experiment and the masking is worth having. `median_background` stays protected
because the PDMS is well under half the non-cell pixels, which is why `median_cy5`
tracks a dye ramp cleanly over a 2.8x range.

- `compute_pdms_mask()` in `extractor.py` finds the trap from the **brightfield**
  tiles, which are always imaged, so the masking works for experiments with no Cy5
- Tiles are drift-corrected and centred on traps, so the trap lies at the same place
  in every tile; the median over tiles reinforces it
- Cells are blanked before the median, never averaged away: cells sit in the trap's
  pocket and so lie in similar places in different tiles
- Found once per position at the first extracted time point, stored at
  `/extraction/pdms_mask` by `ExtractorWriter.write_pdms_mask`, and reloaded on a
  resumed run so that it stays consistent
- Applied both to the background functions and to `get_background_masks`, which
  feeds the per-z median subtraction behind every channel's `_bgsub` images
- Controlled by `ExtractorParameters.mask_pdms` (default `True`); disabled
  automatically if the mask covers over 60% of a tile

### Cells exclude the Cy5 dye

Measured on experiment 4802, using every trap and every *segmented* cell from
`cell_info` — 261766 trap-timepoints. Normalising each trap-timepoint's
`median_background` by the median over traps at that timepoint, to strip out the
dye ramp:

| correlation with the background | |
| --- | --- |
| pooled, against cell count | +0.097 |
| within-trap, against cell count | +0.034 |
| within-trap, against cell **area** | **-0.099** |

Within a trap, the timepoints at which it holds more cell area are the timepoints
at which its Cy5 background is *lower*, and the effect strengthens with the dye
level (slope -0.006 per 1000 px at a dye level of 30, -0.017 at 70+). Cells
displace dye-bearing medium. Dye *binding to cells* would push the background up
with occupancy and is ruled out by the sign.

Beware of measuring this from a wela tsv: those hold only picked cells and only
the traps that contain them — 736 of about 1193 traps for 4802 — which inflates
the apparent effect roughly twofold and makes it look positive. Most of what
looks like an occupancy effect pooled across traps is a between-trap confound:
traps that habitually hold more cells sit where the background is brighter.

A consequence for the empty-trap fix above: a trap carrying a typical ~2000 px of
cells should read about 3% *higher* when empty, at high dye. Genuinely empty
traps cannot be measured from data extracted before that fix, because they are
exactly the rows that were missing, so this remains an extrapolation.

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
