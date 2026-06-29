# aliby examples

Notebook-style scripts that exercise each of aliby's main pipeline shapes.
Each script flows top-to-bottom and defaults to the bundled Zenodo test
dataset (fetched on first use via pooch, cached under
`~/.cache/pooch/aliby_tests/`). Swap in your own data by editing the
`DATA_PATH` / `REGEX` / `CAPTURE_ORDER` (or zarr root) constants at the top.

## Catalogue

| Example | Input modality | Segmenter | Extractor | Bundled fixture | Downstream app it derives from |
|---|---|---|---|---|---|
| [`01_cell_painting_tiff.py`](./01_cell_painting_tiff.py) | TIFF directory, multi-channel epifluorescence | `cellpose` (local) or `nahual_cellpose` (remote) | `cp_measure` intensity + sizeshape + colocalisation, joblib per-position | `crop_cellpainting_256` (516 KB) | uoe-kenneth-amiodarone (and the internal GSK pipeline it mirrors) |
| [`02_zarr_deep_embeddings.py`](./02_zarr_deep_embeddings.py) | Monozarr (CYX groups) | -- (tile only) | `nahual_embed_*` -- DINOv2 / OpenPhenom / MorphEM / SubCell | `crop_cellpainting_256.zarr` (540 KB) | JUMP_lite |
| [`03_yeast_timelapse_baby.py`](./03_yeast_timelapse_baby.py) | TCZYX zarr (time-lapse) | `nahual_baby` via baby-phone | `cp_measure` intensity + sizeshape; BABY tracking/lineage as a post-step | `crop_timeseries_alcatras_square_same_channels_293.zarr` (5.7 MB) | aliby_baby_processing |

`02_zarr_deep_embeddings.py` is the example that exercises the PR #20 fix
(`get_profiles_from_state`'s ndarray branch, `pipe_core.py:474-480`).
Without that patch every embedder pipeline raises
`ValueError: zip() argument 2 is shorter than argument 1` from
`format_extraction`'s strict zip. The script stubs an embedding ndarray
through `get_profiles_from_state` when no live Nahual server is configured,
so running the script alone is enough to demonstrate the fix.

## Running

```bash
uv run python examples/01_cell_painting_tiff.py
uv run python examples/02_zarr_deep_embeddings.py
uv run python examples/03_yeast_timelapse_baby.py
```

Examples 02 and 03 need a Nahual model server to actually invoke the
remote inference path -- the launch hint lives in each script's Section 4.
Without a server, 02 stubs the embedder output and 03 stops after building
the pipeline dict, so the script always runs to completion (and prints
expected outputs as comments).

## Bundled test dataset

The fixture is the same Zenodo deposit (`19411429`) the test suite uses.
Five sub-datasets totalling ~18 MB cover every modality aliby supports:

| Sub-dataset | Modality | Layout | Used by |
|---|---|---|---|
| `crop_cellpainting_256` | Cell Painting TIFF | `DatasetDir`, regex `WFC` | Example 01 + `tests/test_cellpose_cpmeasure_minimal.py` |
| `crop_cellpainting_256.zarr` | Cell Painting monozarr | `DatasetZarr`, CYX groups | Example 02 |
| `crop_timeseries_alcatras_round_diff_dims_293` | Yeast time-series TIFF | `DatasetDir`, regex `FTCZ` | `tests/test_cellpose_cpmeasure_minimal.py` |
| `crop_timeseries_alcatras_square_same_channels_293` | Yeast time-series TIFF | `DatasetDir`, regex `FTCZ` | `tests/test_cellpose_cpmeasure_minimal.py` |
| `crop_timeseries_alcatras_square_same_channels_293.zarr` | Yeast time-series zarr | `DatasetZarr`, TCZYX | Example 03 |

The catalogue lives at `src/aliby/test_data.py` and exposes:

```python
from aliby.test_data import DATASETS, get_dataset, get_dataset_path, get_data_root
```

Use `get_dataset_path("crop_cellpainting_256")` to get the absolute path
to a sub-dataset (downloads it on first call), and `get_dataset(name)` to
look up the regex / capture order / channel-name → index mapping. Tests
share the same module via the `data_dir` fixture in `tests/conftest.py`.

## Downstream cross-check

The four downstream applications below consume aliby in production. They
form the regression target for API-surface changes: any aliby release
should preserve the symbols and signatures they import.

| App | Repo (visibility) | aliby symbols imported | Module split | Tested against current `main`? |
|---|---|---|---|---|
| Cell Painting profiling (UoE/Kenneth) | `afermg/uoe-kenneth-amiodarone` (public) | `aliby.io.dataset.DatasetDir`, `aliby.pipe.run_pipeline_and_post`, `aliby.pipe_builder.build_pipeline_steps`, `aliby.pipe_core.configure_logging` | `pipe` (cellpose flavour) | Yes -- unaffected by PR #20 |
| Cell Painting profiling (GSK) | `afermg/gsk-ia` (private/404 from public gh) | same as uoe-kenneth (per uoe-kenneth's `CLAUDE.md`: "Mirrors the GSK standardize pipeline") | `pipe` | Inferred green via uoe-kenneth |
| Deep-learning featurisation | `afermg/JUMP_lite` `main` (public) | `aliby.io.dataset.dispatch_dataset`, `aliby.pipe.run_pipeline_and_post` | `pipe` + `nahual_embed_*` | Sensitive to PR #20 (broken without it). **Additional API drift**: JUMP_lite still calls `run_pipeline_and_post(img_source=..., fov=...)` which current aliby does not accept -- the signature is `(pipeline, pipeline_name, output_path, overwrite)`. Needs a downstream-side update beyond PR #20. |
| Yeast time-lapse | `afermg/aliby_baby_processing` (public) | `aliby.pipe_builder_baby.build_pipeline_steps`, `aliby.pipe_baby.run_pipeline_and_post` | `pipe_baby` (BABY flavour) | Yes -- unaffected by PR #20 |

The `JUMP_lite#3` PR pivots away from aliby featurisation entirely (it
uses saturation YAML pipelines) and is out of scope for this validation.

### How to re-run the cross-check

For each downstream repo, after pulling the latest aliby:

```bash
# 1. The aliby symbols the app imports must all still exist:
grep -RhE '^(from aliby|import aliby)' <downstream-repo>/ | sort -u
# 2. The entry-point signature each app uses must match current aliby:
grep -RnE '(build_pipeline_steps|run_pipeline_and_post)\(' <downstream-repo>/
```

Diff against the symbols / signatures in this table and the per-example
imports. Any drift here is a downstream-update task for the app's owner.

## Anonymisation

These examples were derived from production scripts that carried proprietary
identifiers (paths under `/datastore/alan/...`, ELN IDs, plate codes,
internal QC blacklists). The example versions use placeholder constants
only and document the expected layout in each script's header docstring.
The bundled Zenodo fixture (record `19411429`) was published by the aliby
maintainers and is already a publicly-cleared subset (the Cell Painting
crops are derived from a public JUMP Consortium QC plate, not from any
private GSK or UoE dataset).

If you add a new example, verify before committing:

```bash
grep -rE '(ELN[0-9]+|H00[A-Z0-9_]+|/datastore/alan/)' examples/*.py
```

The above must return no matches.
