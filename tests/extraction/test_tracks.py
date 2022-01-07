
from extraction.core.tracks import load_test_dset, clean_tracks, merge_tracks

def test_clean_tracks():
    tracks = load_test_dset()
    clean = clean_tracks(tracks, min_len=3)

    assert len(clean) < len(tracks)
    pass

def test_merge_tracks_drop():
    tracks = load_test_dset()

    joint_tracks,joint_ids = merge_tracks(tracks,window=3, degree=2, drop=True)

    assert len(joint_tracks)<len(tracks), 'Error when merging'

    assert len(joint_ids), 'No joint ids found'

    pass

def test_merge_tracks_nodrop():
    tracks = load_test_dset()

    joint_tracks,joint_ids = merge_tracks(tracks,window=3, degree=2, drop=False)

    assert len(joint_tracks)==len(tracks), 'Error when merging'

    assert len(joint_ids), 'No joint ids found'

    pass
