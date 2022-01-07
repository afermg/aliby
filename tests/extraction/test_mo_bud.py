import pytest

import os
from pathlib import Path

# from copy import copy

import pickle

from extraction.core.tracks import get_joinable, get_joint_ids
from extraction.core.extractor import Extractor, ExtractorParameters
from extraction.core.lineage import reassign_mo_bud


DATA_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / Path("data")


def test_mobud_translation(tracks_pkl=None, mo_bud_pkl=None):

    if tracks_pkl is None:
        tracks_pkl = "tracks.pkl"

    if mo_bud_pkl is None:
        mo_bud_pkl = "mo_bud.pkl"

    mo_bud_pkl = Path(mo_bud_pkl)
    tracks_pkl = Path(tracks_pkl)

    with open(DATA_DIR / tracks_pkl, "rb") as f:
        tracks = pickle.load(f)
    with open(DATA_DIR / mo_bud_pkl, "rb") as f:
        mo_bud = pickle.load(f)

    params = Parameters(**get_params("batman_dual"))
    ext = Extractor(params)

    joinable = get_joinable(tracks, **ext.params.merge_tracks)
    trans = get_joint_ids(joinable)

    # Check that we have reassigned cell labels
    mo_bud2 = reassign_mo_bud(mo_bud, trans)

    assert mo_bud != mo_bud2
