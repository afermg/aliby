from functools import partial

import numpy as np

from aliby.pipe_core import get_profiles_from_state
from extraction.extract import extract_tree, process_tree_masks_overlap


def _layered_masks(*layers):
    return [np.stack(layers)]


def test_overlap_extraction_compacts_masks_and_restores_persistent_labels():
    frame_0 = _layered_masks(
        np.array([[1, 1], [0, 0]], dtype=np.uint16),
        np.array([[0, 2], [0, 2]], dtype=np.uint16),
    )
    frame_1 = _layered_masks(
        np.array([[2, 2], [0, 305]], dtype=np.uint16),
        np.array([[0, 305], [305, 0]], dtype=np.uint16),
    )
    tree = {"None": {"None": ("area",)}}
    pixels = np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)
    compact_maxima = []

    def measure(instructions, masks, pixels, **kwargs):
        del pixels, kwargs
        compact_maxima.extend(int(mask.max()) for mask in masks)
        return [1.0] * len(instructions)

    outputs = [
        process_tree_masks_overlap(tree, masks, pixels, measure)
        for masks in (frame_0, frame_1)
    ]
    state = {"data": {"extract_cells": outputs}}
    pipeline = {"steps": {"extract_cells": {}}}

    profiles = get_profiles_from_state(state, pipeline)
    labels_by_tp = {0: set(), 1: set()}
    for tp, label in zip(
        profiles["metadata_tp"].to_pylist(),
        profiles["metadata_label"].to_pylist(),
        strict=True,
    ):
        labels_by_tp[tp].add(label)

    assert compact_maxima == [1, 2]
    assert labels_by_tp == {0: {1, 2}, 1: {2, 305}}
    assert profiles.num_rows == 4


def test_overlap_extraction_measures_multiple_layers_with_parallel_path():
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
