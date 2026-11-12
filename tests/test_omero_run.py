"""
Run the pipeline on an experiment held by the OMERO server.

The run needs the server and its password, so it is slow and is skipped
unless asked for and given the password in the environment:

    OMERO=password pytest tests/test_omero_run.py --run-slow
"""

import os

import h5py
import numpy as np
import pytest

from agora.io.signal import Signal
from aliby.global_settings import global_settings

HOST = "staffa.bio.ed.ac.uk"
USERNAME = "upload"
# the environment variable holding the password
PASSWORD_VARIABLE = "OMERO"
EXPT_ID = 2002
CHANNELS = ["Brightfield", "GFP_Z", "cy5", "mCherry_Z"]
NTIMEPOINTS = 8
SIGNALS = [
    "extraction/general/null/area",
    "extraction/general/null/volume",
    "extraction/general/null/image_x",
    "extraction/general/null/image_y",
    "extraction/GFP_Z/mean/mean",
    "extraction/GFP_Z_bgsub/mean/membrane_fluorescence_fl",
    "extraction/GFP_Z_bgsub/mean/membrane_fluorescence_remaining_fl",
    "postprocessing/buddings/extraction_general_null_volume",
    "postprocessing/bud_metric/extraction_general_null_volume",
]


@pytest.fixture
def position_h5(tmp_path, monkeypatch):
    """Run the first position of the experiment and return its h5 file."""
    from aliby.pipeline import Pipeline, PipelineParameters

    # a function not among the defaults, put back when the test ends
    monkeypatch.setattr(
        global_settings,
        "fluorescence_functions",
        [*global_settings.fluorescence_functions, "membrane_fluorescence"],
    )
    parameters = PipelineParameters.default(
        general={
            "expt_id": EXPT_ID,
            "distributed": 0,
            "directory": str(tmp_path),
            "host": HOST,
            "username": USERNAME,
            "password": os.environ[PASSWORD_VARIABLE],
            "filter": [0],
            "tps": NTIMEPOINTS,
            "channels": CHANNELS,
        }
    )
    Pipeline(parameters).run()
    h5_files = sorted(tmp_path.rglob("*.h5"))
    assert len(h5_files) == 1, f"expected one position, found {h5_files}"
    return h5_files[0]


@pytest.mark.slow
@pytest.mark.skipif(
    PASSWORD_VARIABLE not in os.environ,
    reason=f"set {PASSWORD_VARIABLE} to the OMERO password",
)
def test_a_position_on_omero_is_run_from_images_to_buds(position_h5):
    """Tile, segment, extract and postprocess one position from OMERO."""
    with h5py.File(position_h5, "r") as f:
        channels = [
            channel.decode() if isinstance(channel, bytes) else channel
            for channel in f.attrs["channels"]
        ]
        ntraps = len(f["trap_info/trap_locations"])
        timepoints = f["cell_info/timepoint"][()]
        for modifier in ("merges", "picks", "lineage_merged"):
            assert f"modifiers/{modifier}" in f
    # the channels are those asked for, in OMERO's order
    assert channels == CHANNELS
    assert ntraps > 0
    # cells were found at every time point
    assert sorted(np.unique(timepoints)) == list(range(NTIMEPOINTS))
    signal = Signal(position_h5)
    missing = set(SIGNALS) - set(signal.available)
    assert not missing, f"signals not in the h5 file: {sorted(missing)}"
    volume = signal.get_raw("extraction/general/null/volume")
    assert volume.shape[1] == NTIMEPOINTS
    assert len(volume) > 0
    # a cell has no value at a time point it is not seen at
    seen = volume.notna()
    assert (volume.values[seen] > 0).all()
    mean = signal.get_raw("extraction/GFP_Z/mean/mean")
    # the same cells at the same time points as the volumes
    assert mean.index.equals(volume.index)
    assert mean.notna().equals(seen)
    assert (mean.values[seen] > 0).all()
    membrane = signal.get_raw(SIGNALS[5])
    assert membrane.index.equals(volume.index)
    # a cell too small to have a membrane and an interior has no value
    measured = membrane.notna()
    assert not (measured & ~seen).values.any()
    assert measured.values.sum() > 0.9 * seen.values.sum()
    # a cell's place in the image lies within the image
    for name in ("image_x", "image_y"):
        place = signal.get_raw(f"extraction/general/null/{name}")
        assert place.notna().equals(seen)
        assert (place.values[seen] >= 0).all()
