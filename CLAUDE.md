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

### Tests
```bash
# unit tests mock omero, so they need no OMERO server
python -m pytest tests/
# include the slow integration tests
python -m pytest tests/ --run-slow
```

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

### Stopping early and empty time points

- `CloggingCheck` in `pipeline.py` holds the areas of the last `ntps_to_eval` time points as the extractor returns them, the latest included, and gives the fraction of tiles with cells that are clogged. `check_earlystop` gives the same from an h5 file. Until October 2026 the h5 file was read back at every time point and the latest time point was left out
- A time point with no cells in any trap is not a failure: nothing is added to `cell_info`, the backgrounds are extracted, and the position carries on. `format_segmentation` raises `InconsistentOutput` when BABY's outputs differ in length, which is a failure and ends the position
- A position has failed only if no time point was segmented and extracted; a run of one time point is post-processed
- An error raised by one position is logged with its traceback and the other positions run (`run_one_position_safely`); `Pipeline.run` raises only if every position raised. Until November 2026 one error ended the run with one processor, and with several lost the report of the h5 files
- A position's image is opened once: `generate_h5file` takes the image's name. The logs are cached once for a folder in one process (`cache_logs_once`), by `PipelineParameters.default`, and `Pipeline.setup` caches them only for another folder, such as `store`. A local data set's logs are looked for in its folder and one folder down, never in a zarr store
- With `distributed` over 1, each spawned worker sets up the aliby logger for itself and appends to the one log file

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
  - The tiling itself lives in the `tiler` package (git.ecdf.ed.ac.uk/swain-lab/aliby/tiler), shared with wela, bairn and the curation GUI: `tiler.detect` finds traps (`aliby.tile.process_traps` re-exports it), `tiler.drift` measures drift and `tiler.crop` cuts and pads tiles (`aliby.tile.tiles` re-exports `tile_in_image` and `too_far_outside`). `Tiler` is the pipeline step around them; change tiling in tiler, not here
  - Where a tile sits is `sooth.tiles`, the definition of record; `aliby.tile.tiles.Tile` calls it, and `tests/test_tile_golden.py` pins it against real layouts
  - Drift registers each image to the first processed image (`drift_reference="first"`), stored as the change since the previous time point so drifts still sum to a displacement; a second registration to the previous image logs a warning when the two differ by more than `drift_check_px` (3). Registering to the previous image, the old method, accumulated 2-7 px of error on 192- and 288-frame movies against about 0.5 px
  - The h5 records `trap_info/attrs/image_size`, the field the traps were
    found in, because `max_size` is alibylite's own default of 1200 and not
    a measurement: a reader that sized a position by it drew a map of the
    wrong shape for any other camera
  - Traps are found and drift is measured in the middle z section, `ref_z`, which `PipelineParameters.default` takes from the metadata's number of brightfield sections unless `tiler={"ref_z": k}` gives one. Until October 2026 the tiler always used z 0: its default was read when aliby was imported, before any metadata. On `test26643` the middle section finds 94 traps where z 0 found 82, all 82 among them within 5 px, with the same drifts. With no metadata of z sections (a metadata dict, or TIFFs with no log) `ref_z` stays 0
  - The Tiler keeps the last z stack read for the reference channel and the images it registers for drift (`load_image`, `get_ref_plane`), so brightfield is read once at a time point, not once each for BABY and the extractor: 6 planes for a 5-section stack where 13 were read. Do not change in place an array `load_image` returns
  - Tiles are cut with numpy unless `lazy=True`. Wrapping a numpy image in dask once for each tile hashed every pixel each time, which was 40% of a run on zarr data
  - One lazy tile (`get_lazy_tile_view`, which wela's torch dataset calls for each tile, and `get_tile_data`) is read from a zarr array as its own region, in one request for all z sections, through `StackView` and `cut_tile`: 5 ms a tile where reading and hashing the whole stack took 10. One request for each z section is slower than either, 13 ms, with zarr 3
  - Time points are always the images' own indices: `initial_processing_tp` only says where to start, and the images before it get zero drift. It replaced `initial_tp`, which renumbered time points, and passing `initial_tp` raises
- `aliby.baby_sitter.BabyRunner`: Interfaces with Baby-seg to return cell masks, mother-bud pairs, and tracking data
- `extraction.core.extractor.Extractor`: Extracts areas, volumes, and fluorescence data using cell masks (writes directly to HDF5)
- `postprocessor.core.postprocessing.PostProcessor`: Applies cell picking, track merging, and runs processes like budding analysis

**Data Access Classes**
- `agora.cells`: Accesses cell information and masks from HDF5 files (lazy loading). `Cells.at_time` returns **one mask per cell the file records**, an empty edge mask included: its masks are matched to traps by position, and the extractor (`get_outlines`) matches them to `labels_at_time` by position again, so dropping a cell from one list and keeping it in the other — as it did until `tests/test_cells.py` was written — gave every later cell in that time point the trap and the label of the cell before it
  - A mask is its outline filled within the outline's bounding box
    (`fill_outline`): the same mask as filling the whole tile, in a quarter
    of the time, which was 122 ms a time point for 557 cells
- `agora.signal`: Gets extracted properties for all cells/timepoints from HDF5 (used in postprocessing)
  - `available` and `ntimepoints` are found again when the file's
    modification time changes (`find_once_for_file`), and `lineage` reads
    the file at each call. Until November 2026 all three were cached for
    the life of the Signal and went stale when a file was postprocessed
    again. `lineage(location)` raises `KeyError` if the location is missing;
    only with no location does it fall back to `postprocessing/lineage`
  - Columns in minutes are the time point times the interval over 60: integers when every one is whole, as for 300 s, and floats otherwise (0, 2.5, 5 for 150 s). `Grouper.tinterval_minutes`, which wela's DataLoader takes as `dt`, is not rounded either. The interval used to be rounded to whole minutes first, making 150 s two minutes and 30 s none
  - A time point with no cells has no rows in a Signal's table but is still a column, of NaN, so neighbouring columns are neighbouring time points
- `agora.bridge`: Interface layer for HDF5 file operations
- `aliby.io.image` and `aliby.io.omero` read pixels through tiler's sources
  (`ZarrSource`, `TiffFolderSource`, `TiffSource`, `OmeroSource`), shared
  with wela, bairn and the curation GUI; aliby adds only the microscope's
  log files. `dispatch_image` and the Image classes keep their API. Change
  how a format is read in tiler, not here. `tests/test_image_golden.py` pins
  the readers. `golden_df.pkl` is from the zarr of `test26643` with traps
  found in the middle z section, which its log gives. Running the regression
  pipeline on `test26643_tiff` reproduced the golden file exactly while both
  used z 0; a run from TIFFs with a metadata dict still uses z 0, because
  the dict gives no number of z sections, and so finds other traps unless
  it is given `tiler={"ref_z": 2}`, which has not been checked against the
  new golden file
  - `ImageDir` passes the log's channels to `TiffFolderSource`, which reads
    each channel from the files that name it. It used to stack the files in
    sorted order and name them in the log's, swapping channels whose order
    is not alphabetical (Brightfield, GFP, mCherry, cy5 in experiments 2801
    and 2809, which ran from OMERO and are unaffected)
  - The OMERO `Image` holds one tiler `OmeroSource` for its with block,
    which retries a request whose connection fails. `Image.tiles` (used by
    wela's `GrabTiles.get_tiles`) asks OMERO for only each tile's region: a
    slice of a lazy plane still downloads the whole plane, over 100 times
    the data for a 117-pixel tile of a 1200-pixel image. `tiler.read_tile`
    pads it by the rule `cut_tile` uses, so a tile is the same whichever way
    it is loaded
  - A local image's with block closes nothing, because wela builds a Tiler
    inside one and reads from it afterwards
  - OMERO and TIFF images reach the Tiler through `tiler.as_dask`, whose
    graph has two layers whatever the movie's length. Until November 2026
    it stacked a delayed read for each plane, and every read walked the
    whole graph: 0.3 s for one z stack of a 300-frame, 4-channel,
    5-section movie and 1.4 s for 1000 frames, now a few milliseconds.
    `image.data` builds the array anew at each access, so take it once
- `find_channel_index` returns a channel the name matches whole, ignoring
  case, before one it only begins: GFP finds GFP although GFPFast is listed
  first. Until November 2026 it returned the first channel the name began,
  so the extractor could store GFPFast's pixels as GFP
  - omero is imported only when OMERO data is opened, so local data and the
    pipeline need no omero-py
- `postprocessor.grouper`: Concatenates signals across positions to generate experiment-wide dataframes
  - Its properties are found anew at each access. Until November 2026 they
    were cached, so those that print printed once and `groups` and
    `channels` kept positions wela had since filtered out

### Microscope logs

- `logfile_parser.swainlab_parser` reads a current log (`.log`) up to the
  start of the experiment; its tables begin with a header and end with a
  blank line. A channel is named by its image config, such as `GFP_Z`,
  the name the images and OMERO use. Until November 2026 a config was
  kept only if its name and its channel were both exactly in
  `possible_imaging_channels`, so `GFP_Z`, `mCherry_Z` and `NADH10ms` were
  dropped: 17 of 19 stored logs lost a channel. OMERO runs took their
  channels from OMERO and were unaffected; a local run did not extract
  the dropped channels, or failed to open a TIFF position
- `channels_by_position` comes from each group's `field: config` block,
  whose hexadecimal number has a bit for each time point: a channel with
  none set is not the group's. A group given twice counts as given last.
  `MetaData` puts each position's channels in OMERO's order. `ImageDir`
  reads a TIFF position with its own channels, and the `Tiler` uses them
  only for an image with that many channels (`has_channels_of_its_own`)
- A line of general information is split at its first colon only, so
  details and tags keep their own colons
- The legacy parser (`Parser`, for `log.txt` and `Acq.txt`) has triggers
  `startswith` and `re` only, reads a flag of 0 as False (`bool("0")` is
  True), and has a section type `ignore`, which ends the section before
  it and keeps nothing. wela's `get_exp_conditions` uses it too

### Dependencies and Integration
- **Baby-seg**: Neural network segmentation (tensorflow backend)
- **OMERO**: Optional integration for microscopy server access
- **HDF5/Zarr**: Data storage formats
- **Pandas/NumPy**: Data manipulation and analysis
- Uses Poetry for dependency management with optional groups (baby, omero)
- **`sooth` and `tiler` float, and the lock does not pin them by intent.** Both are declared without a `rev`, so a commit in `poetry.lock` is wherever this repo last resolved, not a claim that they agree. Do not re-lock because a sibling changed; run `poetry update sooth tiler` when provisioning an environment. The rule, and the guard that replaces the pin (`scripts/check_api_use.py`), are in **sooth's `CLAUDE.md`**.

### Configuration Patterns
- All processing steps inherit from `agora.abc.ProcessABC` and use corresponding `ParametersABC` classes
- `ParametersABC.default` copies `_defaults` in depth, so a list among them, such as the picker's sequence, is not shared between instances
- Configuration uses nested dictionaries that can be recursively merged
- Global settings are defined in `aliby.global_settings` and include imaging specifications and default functions

### Data Storage
- Primary output format is HDF5 with structured datasets (one file per position)
- Metadata from microscopy logs is preserved
- Signal extraction results are organized by channel and processing function
- Writers in `agora.io.writers` handle structured data output
  - A write that fails puts the datasets it had added to back to their
    lengths and raises, ending the position. Until November 2026 it logged
    a warning and went on, leaving `trap`, `cell_label` and `timepoint` of
    unequal lengths, which no reader can take
  - Datasets whose rows are a few numbers have chunks of 1024 rows
    (`chunk_rows`). Left to h5py a chunk was the number of cells at the
    first time point: for a position beginning with three cells, reading
    three lists took 111 ms, not 1.4 ms, and they are read at every time
    point. Edge masks have chunks of 16 whole tiles (`image_chunk_rows`).
    With h5py's chunks, of part of a tile, a position beginning with three
    cells had masks taking 8.0 MB where 16 tiles take 3.0 MB; they are
    also written in 13 ms a time point, not 18, and one cell's masks
    through 80 time points are read in 44 ms, not 80
  - Compression is at level 4. Level 9 took 66 ms a time point to write
    400 cells' edge masks where 4 takes 19 ms, and 377 ms for 100 signals
    where 4 takes 299 ms, for files 17% and 2% smaller
  - The pipeline's `ExtractorWriter` keeps the signals of 16 time points
    (`signals_written_every` in `pipeline.py`) and writes them together,
    and what is waiting is written before postprocessing and when an error
    ends the position. Each append costs PyTables about 2 ms whatever its
    size: for 100 signals and 400 cells, one time point at a time took
    299 ms a time point at level 4, and 16 together take 19 ms. A killed
    process loses the signals of up to 15 time points, but a position is
    always run from its start, so nothing more is lost than before. The
    tables hold the same rows; pandas stores their columns in another
    order of blocks, which it reads and appends to all the same. An
    `ExtractorWriter` made with no `flush_every` writes each time point
    as it is given
  - `tidy` makes a signal's long table from numpy arrays; resetting the
    index and melting was 80 ms of a time point for 100 signals
- Cell properties stored as nested dictionaries: `{'general': {'None': ['area', 'volume', 'eccentricity']}}`
- Picker and merger choices are written to HDF5 for reproducibility
- Signals are pandas tables written with no PyTables index (`index=False` in `CoreWriter.add_df`), so a signal's group holds only `table`. A Signal is read whole and nothing looks rows up by the index, which took three quarters of the time to write and half the file. Files written before October 2026 have an `_i_table` group beside each `table`; `Signal` reads both

### Extraction Functions

**Cell Functions** (`extraction.core.functions.cell_functions`)
- Standard cell measurements: area, median fluorescence, eccentricity
- `centroid_x` and `centroid_y` count pixels from zero, as sooth and the
  curation GUI do, so that a centroid plus its tile's origin is the cell's
  pixel in the image. They counted from one until October 2026: in an h5
  extracted before then, `centroid_x`, `centroid_y`, `image_x`, `image_y`,
  `absolute_x` and `absolute_y` are each one pixel too large
- Multi-channel fluorescence functions
- A function is a cell function only if marked `@cell_function`; the helpers
  beside them (`centroid`, `crop_to_cell`, `identify_vacuole`) cannot be
  named in a tree. Until November 2026 every function in the module could
- A function that needs only a cell's pixels (mean, median, std, total,
  total_squared, max2p5pc, max5px_median) is also in `PIXEL_FUNCTIONS` as a
  function of the pixels. `Extractor.apply_pixel_functions` picks a cell's
  pixels out of its tile once for all of them

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
  `/trap_info/pdms_mask` by `ExtractorWriter.write_pdms_mask`, and reloaded on a
  resumed run so that it stays consistent. It lives in `trap_info`, with the other
  per-position data, rather than in `extraction`, which holds only signals; it is
  kept for the same reason as the picker's and merger's choices, being what
  determines every background value in the file
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
from itself would produce near-zero signals. Until November 2026 they did not skip it:
the background channels were looked for in `tree_dict`, which never held them, so an h5
from before then has a `cy5_bgsub` group. Cy5's image less background is now made only
if a multichannel function names Cy5.

`reduce_extract` reduces all tiles over z at once and, given `reduced_as`, keeps the
result until the next time point, so a channel is reduced once for a cell, its vacuole
and its cytoplasm. A function that returns a dict (`membrane_fluorescence`) becomes one
signal for each of its values, and keeps those names at a time point with no cells.

Measured on `test26643` (94 traps, about 108 cells a time point, two fluorescence
channels, no vacuoles): extraction is 0.28 s of a 2.45 s time point, BABY 1.6 s, the
tiler 0.29 s and the extractor's writer 0.22 s. Within extraction the medians for
background subtraction are the largest part, 0.06 s; finding them for all z sections
or all tiles at once is no faster.

A channel is extracted only if its name begins with one in
`global_settings.possible_imaging_channels`; `build_extraction_tree_from_meta` warns of
any other. In `extraction={...}`, dicts merge and sets join but a list replaces the
default list (`recursive_merge_extractor`).

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

- `PipelineParameters.default(postprocessing={...})` takes `picker`,
  `merger`, `bud_processes`, `targets` and `outpaths`, the keywords of
  `PostProcessorParameters.default`, which took none until November 2026
- A position with no cell at any time point has no Signals; the
  postprocessor writes empty merges, lineage and picks for it and runs no
  bud process

**Picker** (`postprocessor.core.reshapers.picker`)
- Selects cells with lineage information and minimum track length (default: seen at more than 3 time points, so 4 or more)
- Identifies mother-bud relationships using Baby's lineage data
- A picker that picks no cell gives a position with no cells and logs which
  pick removed the last of them. `Signal` tells this from a file with no
  `modifiers/picks`, which has not been postprocessed and gives every cell:
  `read_picks` returns an empty set for the first and `None` for the second.
  Until October 2026 both gave every cell, so a short movie with no lineage
  came back unpicked. `picker_sequence: []` is how to ask for every cell

**Merger** (`postprocessor.core.reshapers.merger`)
- Combines fragmented tracks that should represent the same cell

### The merger looks only at adjacent tracks, and by area alone

**Not changed, by decision (October 2026):** bairn is to replace the tracking,
and the curation GUI is being used to make its training data, so the merger
is left as it is. What follows is what was measured, for whoever returns to it.

`tracks.get_merges` pairs a track that ends at time point t with one that
starts at t + 1 in the same trap (`get_contiguous_pairs`), scores the pair by
area alone, and accepts a relative difference of up to `tolerance` (0.2).

Measured on the 18 stored positions with centroids (experiments 2813 and
4002, and the two test files), on tracks seen more than twice, comparing a
left track's last area and centroid with a right track's first.

*Lost cells mostly come back four or more time points later.* Pairs in one
trap whose right track starts within 8 px of where the left ended and within
30% of its area, against the rate for cells that were there when the left
track ended and so cannot be it:

| time points from end to start | pairs | near and alike | expected by chance |
| --- | --- | --- | --- |
| 1 (adjacent) | 9932 | 542 | 18 |
| 2 | 3607 | 61 | 9 |
| 3 | 3315 | 43 | 13 |
| 4 | 3698 | 558 | 22 |
| 5 | 3678 | 409 | 36 |
| 6 | 3302 | 275 | 48 |
| 7 | 3348 | 222 | 65 |
| 8 | 3277 | 218 | 78 |
| 10 | 3317 | 187 | 103 |
| 15 | 3038 | 157 | 115 |

The jump at 4 is BABY's tracker, which looks back three time points
(`nstepsback = 3` in `baby/tracker/core.py`) and rejoins shorter losses
itself. The excess over chance is about 520 pairs at 1 and about 1600 summed
over 4 to 10, so the merger can reach roughly a quarter of the lost cells that
place and size identify. Beyond about 10, under half of what passes is more
than chance.

*Position discriminates and area hardly does.* One cell, one time point on,
has an area within 20% in 94% of cases, and so do 16% of pairs of different
cells in a trap. One cell moves under 2.4 px in 95% of cases (4.1 px over two
time points, 5.5 over three, 6.6 over four, 7.5 over five, 8.4 over six), and
two different cells are within 2.4 px 0.2% of the time (median 25 px apart).

*The stored merges are doubtful.* All 2032 in these files are adjacent, and
the left track's end and the right track's start are a median 15.6 px apart
(quartiles 10 and 23); 15% are within 8 px, and a median cell's radius is
11.7 px. The trap has not shifted: cells that carry on through the same time
point move a median 0.9 px. For 81% of the merges over 8 px, another track in
the trap ends or starts at that moment, against 59% of the nearer ones.
Whether these are cells that were shoved or neighbours of a similar size
cannot be told from the numbers: it needs a look at the images.

*What a new merger should do*, if one is written: look at a sample of the
distant merges first; score on position as well as area, letting the distance
allowed grow with the gap; allow gaps of up to about 10 time points, keeping
the one-to-one pairing and the lineage check; and fall back to area and
adjacent tracks for a file with no centroids (44 of the 62 stored). The
distributions above for one cell and for two are what a posterior odds of a
pair being one cell would be built from.

The merges found with the default parameters are unchanged by the faster
`get_contiguous_pairs` (November 2026). `smooth=True` had never run: the
check for an odd window was the wrong way round. It now smooths each
track's valid values, still only for growing tracks longer than the
window (`clean_tracks`), and has not been tried on real data.

Also left as it is: `get_predicted_edge_values` fits a line to the last
`window` values of the left track and evaluates it at `len(y) + 1`, two time
points past the track's end, not one.

**Process Functions**
- `buddings`: Analyzes cell division events
- `bud_metric`: Calculates bud-specific measurements
- Applied to signals like volume to generate derived measurements
- A bud with no value at any time point, as in a fluorescence Signal, is
  no budding and adds nothing to its mother's buds. `buddings` gives its
  mothers in the Signal's order; they were in the order of a set
- Measured on a position of 7093 cells and 192 time points, under the
  profiler: postprocessing takes 0.77 s where it took 2.5 s, the merger
  0.26 s, the two bud metrics 0.26 s and buddings under 0.1 s

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
