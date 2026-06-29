"""
Regression coverage for the `nahual_embed_*` (arbitrary embedder) terminal
pathway in `get_profiles_from_state`.

A 2-month-old regression (introduced in commit 005622a, the two-pipelines
refactor) wrapped the embedder ndarray output as
`(itertools.cycle((("__", "__"),)), (ext_output,))`. Combined with
`format_extraction`'s `zip(*instructions_result, strict=True)`, the infinite
`cycle` iterator made the strict zip raise `ValueError: zip() argument 2 is
shorter than argument 1`, breaking *every* DL-embedding pipeline (DINOv2,
SubCell, MorphEM, OpenPhenom, ViT, ...).

The fix wraps the ndarray as a length-1 instructions tuple plus a length-1
metrics tuple so the strict zip yields exactly one (inst, metrics) pair —
the ndarray branch in `format_extraction` then consumes the entire array
via `np.ndenumerate` in that single iteration.

These tests pin both halves of the contract:

1. `format_extraction` accepts a length-1 instructions / length-1 metrics
   payload where the single metrics entry is a 2-D ndarray, and produces
   one row per ndarray row with one column per ndarray column.
2. `get_profiles_from_state` wraps a raw ndarray emitted by a `nahual_embed`
   step into that same shape and yields a non-empty profiles table.
"""

import numpy as np
import pyarrow as pa
import pytest

from aliby.pipe_core import get_profiles_from_state
from extraction.extract import format_extraction


def test_format_extraction_accepts_single_ndarray_payload():
    """The ndarray branch consumes the whole array in one zip iteration."""
    embedding = np.arange(12, dtype=np.float32).reshape(3, 4)
    # Shape produced by the fixed `get_profiles_from_state`: length-1 on
    # both sides so strict-zip yields exactly one pair.
    payload = ((("__", "__"),), (embedding,))

    table = format_extraction(payload)

    assert isinstance(table, pa.Table)
    # `np.ndenumerate` over a 3x4 array yields 12 entries, all rolled up
    # into the same (tile=0, label=0) row via pivoting.
    assert table.num_rows == 3
    metric_columns = [c for c in table.column_names if c.startswith("X_")]
    assert len(metric_columns) == 4
    assert set(table.column_names) >= {"tile", "label"} | set(metric_columns)


def test_format_extraction_rejects_cycle_payload():
    """An infinite `cycle` on the instructions side must NOT be accepted.

    This is the exact shape the buggy code produced. We pin the failure so
    a regression that re-introduces the cycle wrapper is caught here.
    """
    from itertools import cycle

    embedding = np.arange(6, dtype=np.float32).reshape(2, 3)
    bad_payload = (cycle((("__", "__"),)), (embedding,))

    with pytest.raises(ValueError, match="zip"):
        format_extraction(bad_payload)


def test_get_profiles_from_state_wraps_ndarray_output():
    """End-to-end: raw ndarray on `state["data"][nahual_embed_*]` must
    produce a non-empty profiles table without raising."""
    embedding = np.arange(8, dtype=np.float32).reshape(2, 4)
    state = {"data": {"nahual_embed_cells": [embedding]}}
    pipeline = {"steps": {"nahual_embed_cells": {}}}

    profiles = get_profiles_from_state(state, pipeline)

    assert isinstance(profiles, pa.Table)
    assert profiles.num_rows > 0
    assert "metadata_object" in profiles.column_names
    assert "metadata_tp" in profiles.column_names
    # One row per ndarray row, all under tp=0 and object=cells.
    assert set(profiles.column("metadata_object").to_pylist()) == {"cells"}
    assert set(profiles.column("metadata_tp").to_pylist()) == {0}


def test_get_profiles_from_state_handles_multiple_timepoints():
    """Each timepoint produces its own table; metadata_tp is set correctly."""
    state = {
        "data": {
            "nahual_embed_cells": [
                np.arange(6, dtype=np.float32).reshape(2, 3),
                np.arange(6, dtype=np.float32).reshape(2, 3) + 100,
            ]
        }
    }
    pipeline = {"steps": {"nahual_embed_cells": {}}}

    profiles = get_profiles_from_state(state, pipeline)

    assert profiles.num_rows > 0
    assert set(profiles.column("metadata_tp").to_pylist()) == {0, 1}
