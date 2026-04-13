"""
Parser for BABY segmentation output via nahual.

Only used when the segmentation method is 'nahual_baby'. BABY returns
layered masks (overlapping cells across layers), tracked cell labels
(consistent across timepoints), and lineage information (mother-bud
assignments).
"""

import pyarrow as pa


def parse_baby_segment_result(result: dict) -> dict:
    """Split a baby segment result dict into masks and metadata.

    Parameters
    ----------
    result : dict
        Dict returned by nahual.client.baby.process_data with
        return_metadata=True. Keys: 'masks' (list of nyx arrays),
        'metadata' (list of per-tile dicts with 'cell_label' and
        optionally 'mother_assign').

    Returns
    -------
    dict
        With keys 'masks' (list of nyx arrays for extraction) and
        'baby_meta' (per-tile tracking/lineage info).
    """
    return {
        "masks": result["masks"],
        "baby_meta": result["metadata"],
    }


def accumulate_tracking(
    baby_meta_history: list[list[dict]],
) -> dict[int, list[list[int]]]:
    """Build per-tile tracking from accumulated baby metadata.

    Parameters
    ----------
    baby_meta_history : list of list of dict
        Outer list is timepoints, inner list is tiles. Each dict has
        'cell_label' (list of int).

    Returns
    -------
    dict
        Mapping tile_id -> list of cell_label lists (one per timepoint).
    """
    if not baby_meta_history:
        return {}

    n_tiles = len(baby_meta_history[0])
    tracking = {tile_id: [] for tile_id in range(n_tiles)}

    for tp_meta in baby_meta_history:
        for tile_id, tile_meta in enumerate(tp_meta):
            tracking[tile_id].append(tile_meta.get("cell_label", []))

    return tracking


def accumulate_lineage(
    baby_meta_history: list[list[dict]],
) -> dict[int, list[list[int]]]:
    """Build per-tile lineage from accumulated baby metadata.

    Parameters
    ----------
    baby_meta_history : list of list of dict
        Outer list is timepoints, inner list is tiles. Each dict has
        'mother_assign' (list of int, 0 = no mother).

    Returns
    -------
    dict
        Mapping tile_id -> list of mother_assign lists (one per timepoint).
    """
    if not baby_meta_history:
        return {}

    n_tiles = len(baby_meta_history[0])
    lineage = {tile_id: [] for tile_id in range(n_tiles)}

    for tp_meta in baby_meta_history:
        for tile_id, tile_meta in enumerate(tp_meta):
            lineage[tile_id].append(tile_meta.get("mother_assign", []))

    return lineage


def baby_tracking_to_table(
    tracking: dict[int, list[list[int]]],
    lineage: dict[int, list[list[int]]],
) -> pa.Table:
    """Convert baby tracking and lineage dicts into a pyarrow Table.

    Parameters
    ----------
    tracking : dict
        From accumulate_tracking: tile_id -> list of cell_label lists.
    lineage : dict
        From accumulate_lineage: tile_id -> list of mother_assign lists.

    Returns
    -------
    pa.Table
        Table with columns: tile, tp, cell_label, mother_label.
    """
    rows = {"tile": [], "tp": [], "cell_label": [], "mother_label": []}

    for tile_id, tp_labels in tracking.items():
        tp_mothers = lineage.get(tile_id, [[] for _ in tp_labels])
        for tp, labels in enumerate(tp_labels):
            mothers = tp_mothers[tp] if tp < len(tp_mothers) else []
            for i, label in enumerate(labels):
                rows["tile"].append(tile_id)
                rows["tp"].append(tp)
                rows["cell_label"].append(label)
                # mother_assign is indexed by label-1; 0 means no mother
                mother = 0
                if mothers and label > 0 and label <= len(mothers):
                    mother = mothers[label - 1]
                rows["mother_label"].append(mother)

    return pa.Table.from_pydict(rows)
