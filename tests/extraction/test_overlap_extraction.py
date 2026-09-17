from functools import partial

import numpy as np
import pytest

from aliby.pipe_core import get_profiles_from_state
from extraction.extract import (
    extract_tree,
    format_extraction_overlap,
    process_tree_masks_overlap,
)


def _layered_masks(*layers):
    return [np.stack(layers)]


def test_real_overlap_area_profiles_restore_sparse_persistent_labels():
    frame_0 = _layered_masks(
        np.array([[1, 1], [0, 0]], dtype=np.uint16),
        np.array([[0, 2], [0, 2]], dtype=np.uint16),
    )
    frame_1 = _layered_masks(
        np.array([[2, 0], [0, 0]], dtype=np.uint16),
        np.array([[0, 305], [305, 305]], dtype=np.uint16),
    )
    tree = {"None": {"None": ("area",)}}
    pixels = np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)

    outputs = [
        process_tree_masks_overlap(
            tree,
            masks,
            pixels,
            partial(extract_tree, overlap=True),
        )
        for masks in (frame_0, frame_1)
    ]
    assert all(isinstance(area, np.integer) for output in outputs for area in output[1])
    state = {"data": {"extract_cells": outputs}}
    pipeline = {"steps": {"extract_cells": {}}}

    profiles = get_profiles_from_state(state, pipeline)
    area_column = next(
        column for column in profiles.column_names if column.endswith("/area")
    )
    observed = {
        (tp, label): area
        for tp, label, area in zip(
            profiles["metadata_tp"].to_pylist(),
            profiles["metadata_label"].to_pylist(),
            profiles[area_column].to_pylist(),
            strict=True,
        )
    }

    assert profiles.num_rows == 4
    assert observed == {(0, 1): 2, (0, 2): 2, (1, 2): 1, (1, 305): 3}


def test_overlap_extraction_measures_multiple_distinct_layers_with_joblib():
    masks = _layered_masks(
        np.array([[1, 1], [0, 0]], dtype=np.uint16),
        np.full((2, 2), 305, dtype=np.uint16),
    )
    pixels = np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)

    instructions, results, inverse_mappings = process_tree_masks_overlap(
        {"None": {"None": ("area",)}},
        masks,
        pixels,
        partial(extract_tree, overlap=True),
        ncores=1,
    )

    assert [instruction[0] for instruction in instructions] == [(0, 0, 1), (0, 1, 1)]
    assert results == [2, 4]
    assert inverse_mappings[(0, 0)][1] == 1
    assert inverse_mappings[(0, 1)][1] == 305


def test_overlap_extraction_rejects_duplicate_persistent_labels_across_layers():
    masks = _layered_masks(
        np.array([[7, 0], [0, 0]], dtype=np.uint16),
        np.array([[0, 7], [7, 7]], dtype=np.uint16),
    )
    pixels = np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)

    def should_not_measure(*args, **kwargs):
        pytest.fail("duplicate labels must be rejected before measurement")

    with pytest.raises(
        ValueError,
        match=r"Persistent label 7 appears in multiple overlap layers for tile 0",
    ):
        process_tree_masks_overlap(
            {"None": {"None": ("area",)}}, masks, pixels, should_not_measure
        )


def test_overlap_formatter_rejects_unsupported_metric_payload():
    masks = _layered_masks(np.array([[9, 0], [0, 0]], dtype=np.uint16))
    pixels = np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)

    output = process_tree_masks_overlap(
        {"None": {"None": ("area",)}},
        masks,
        pixels,
        lambda instructions, masks, pixels, **kwargs: [object()] * len(instructions),
    )

    with pytest.raises(TypeError, match="unsupported type object"):
        format_extraction_overlap(output)


def test_overlap_extraction_handles_empty_tiles_and_layers():
    masks = [
        np.zeros((0, 3, 3), dtype=np.uint16),
        np.zeros((1, 3, 3), dtype=np.uint16),
    ]
    pixels = np.zeros((2, 1, 1, 3, 3), dtype=np.uint8)
    measured_masks = []

    def measure(instructions, masks, pixels, **kwargs):
        del pixels, kwargs
        measured_masks.extend(masks)
        assert instructions == ()
        return []

    instructions, results, inverse_mappings = process_tree_masks_overlap(
        {"None": {"None": ("area",)}}, masks, pixels, measure
    )

    assert instructions == ()
    assert results == []
    assert [mask.shape for mask in measured_masks] == [(0, 3, 3), (1, 3, 3)]
    assert set(inverse_mappings) == {(1, 0)}
