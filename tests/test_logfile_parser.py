"""Test the parsers of the microscopes' log files."""

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from agora.io.metadata import MetaData
from aliby.io.image import ImageDir
from aliby.tile.tiler import Tiler
from logfile_parser import Parser
from logfile_parser.logfile_parser_legacy import ParseError
from logfile_parser.swainlab_parser import parse_swainlab_logs

# a log whose image configs are not named as their channels are, with a
# group given twice and a group that is not imaged in every channel
LOG = """\
2025-10-10 16:21:00,649 - INFO
Swain Lab microscope experiment log file
Microscope name: Batgirl
Omero tags: 10-Oct-2025,397 : htb2-GFP, Batgirl
Experiment details: Aim: a ramp after 3h (TP:36) : slope 0.1%/h.
-----Acquisition settings-----
2025-10-10 16:21:00,649 - INFO Image Configs:
Image config,Channel,Description,Exposure (ms),Number of Z sections,\
Z spacing (um),Sectioning method
Brightfield,Brightfield,Default brightfield. 5 Z sections,10,5,0.6,ZStage
GFP_Z,GFP,Default GFP, with Z sectioning,30,5,0.6,ZStage
NADH10ms,NAD(P)H,Default NAD(P)H imaging,10,1,0.6,ZStage
cy5,cy5,Default cy5. No Z sectioning,100,1,0.6,ZStage

Device properties:
Image config,device,property,value
Brightfield,MCC-DAC-4,Volts,0.1
GFP_Z,CairnIO,State,On


2025-10-10 16:21:00,649 - INFO
group: 1352 field: position
Name,X,Y,Z,Autofocus offset

group: 1352 field: time
start: 0
interval: 300
frames: 192

group: 1352 field: config
Brightfield: 0xffff

2025-10-10 16:21:00,649 - INFO
group: 1352 field: position
Name,X,Y,Z,Autofocus offset
1352_001,-1036,313.8,5276.85,172.875
1352_002,-740.8,-31.3,5274.7,172.875

group: 1352 field: time
start: 0
interval: 300
frames: 192

group: 1352 field: config
Brightfield: 0xffff
GFP_Z: 0xffff
NADH10ms: 0x5555
cy5: 0xffff

2025-10-10 16:21:00,649 - INFO
group: 1704 field: position
Name,X,Y,Z,Autofocus offset
1704_001,-1015.3,-1461.4,5265.25,172.875

group: 1704 field: time
start: 0
interval: 300
frames: 192

group: 1704 field: config
Brightfield: 0xffff
GFP_Z: 0x0
cy5: 0xffff

2025-10-10 16:21:00,649 - INFO
-----Experiment started-----
interval: 7
"""
CHANNELS = ["Brightfield", "GFP_Z", "NADH10ms", "cy5"]


@pytest.fixture
def log(tmp_path):
    """Write the log to a folder and return the file."""
    path = tmp_path / "expt.log"
    path.write_text(LOG)
    return path


def test_a_channel_is_named_by_its_image_config(log):
    """Keep every channel, whatever its name and its description."""
    # GFP_Z, and any config not named exactly as a known channel, was
    # left out, so that a local run did not extract it
    meta = parse_swainlab_logs(log)
    assert meta["channels"] == CHANNELS
    assert meta["number_z_sections"] == dict(zip(CHANNELS, [5, 5, 1, 1]))
    assert meta["exposure"]["GFP_Z"] == 30
    assert meta["sectioning_method"]["NADH10ms"] == "ZStage"
    assert meta["device"] == [
        ("Brightfield", "MCC-DAC-4", "Volts", 0.1),
        ("GFP_Z", "CairnIO", "State", "On"),
    ]
    assert meta["time_settings/timeinterval"] == 300
    assert meta["time_settings/ntimepoints"] == 192


def test_text_with_colons_is_kept_as_written(log):
    """Split a line of general information at its first colon only."""
    meta = parse_swainlab_logs(log)
    assert meta["experiment_details"] == [
        "Aim: a ramp after 3h (TP:36) : slope 0.1%/h."
    ]
    assert meta["omero_tags"] == ["10-Oct-2025", "397 : htb2-GFP", "Batgirl"]


def test_each_position_has_the_channels_of_its_group(log):
    """Give a position the channels its group is imaged in."""
    meta = parse_swainlab_logs(log)
    assert meta["channels_by_position"] == {
        "1352_001": CHANNELS,
        "1352_002": CHANNELS,
        "1704_001": ["Brightfield", "cy5"],
    }
    assert meta["spatial_locations"]["1704_001"] == (-1015.3, -1461.4)


def test_a_log_with_no_configs_has_no_channels_by_position(tmp_path):
    """Leave the key out, for a reader takes it to cover each position."""
    path = tmp_path / "expt.log"
    path.write_text(LOG.replace("field: config", "field: other"))
    assert "channels_by_position" not in parse_swainlab_logs(path)


def test_channels_by_position_follow_the_order_of_omero(log):
    """Order a position's channels as the images' channels are."""
    omero = ["cy5", "Brightfield", "GFP_Z", "NADH10ms"]
    meta = MetaData(log.parent, omero_meta={"channels": omero})
    assert meta.full["channels_by_position"]["1352_001"] == omero
    assert meta.full["channels_by_position"]["1704_001"] == [
        "cy5",
        "Brightfield",
    ]


def test_a_tiff_position_has_the_channels_of_its_own_group(log):
    """Read a position imaged in two channels of the experiment's four."""
    folder = log.parent / "1704_001"
    folder.mkdir()
    for channel in ("Brightfield", "cy5"):
        tifffile.imwrite(
            folder / f"1704_001_t0000_{channel}_z00.tiff",
            np.zeros((4, 5), dtype=np.uint16),
        )
    image = ImageDir(folder)
    assert image.metadata["channels"] == ["Brightfield", "cy5"]
    assert image.data.shape == (1, 2, 1, 4, 5)


ACQ = """\
Z_sectioning:
Sections,Spacing,PFSon?,AnyZ?,Drift,Method
       5,   0.60,     0,    1,    0,     2
Pump states at beginning of experiment:
Pump port, Diameter, Current rate, Direction, Running, Contents
     COM5,    15.47,         4.00,       INF,       1, glucose
     COM6,    15.47,         0.00,       INF,       0, NA
"""


def test_a_flag_of_zero_in_a_legacy_log_is_false():
    """Read 0 as False, which bool makes True of any text."""
    parsed = Parser("multiDGUI_acq_format").parse(ACQ.splitlines())
    assert parsed["zsectioning"]["pfson"] == [False]
    assert parsed["zsectioning"]["anyz"] == [True]
    assert parsed["pumpinit"]["isrunning"] == [True, False]


def test_what_follows_the_tags_of_a_legacy_log_is_not_kept():
    """End the tags at the line that follows them, and keep nothing."""
    lines = [
        "Omero tags:",
        "19-May-2022,Batgirl",
        "PFS is locked",
        "Time to next time point:23.9679",
    ]
    parsed = Parser("multiDGUI_log_format").parse(lines)
    assert parsed == {"omero_tags": ["19-May-2022", "Batgirl"]}


def test_two_sections_triggered_by_one_line_raise_a_parse_error(tmp_path):
    """Raise ParseError, not the AssertionError of an assert."""
    grammar = tmp_path / "grammar.json"
    grammar.write_text(
        '{"a": {"trigger_startswith": "x"}, "b": {"trigger_re": "^x"}}'
    )
    with pytest.raises(ParseError):
        Parser(str(grammar)).parse(["x"])


def test_the_tiler_takes_a_position_s_channels_only_if_they_fit():
    """Use the log's channels for a position if its image has as many."""
    tiler = SimpleNamespace(position_name="1704_001")
    full = {"channels_by_position": {"1704_001": ["Brightfield", "cy5"]}}
    meta = {"full": full}
    fits = Tiler.has_channels_of_its_own
    assert fits(tiler, np.zeros((3, 2, 5, 4, 4)), meta)
    # an image with every channel of the experiment is indexed by all
    assert not fits(tiler, np.zeros((3, 4, 5, 4, 4)), meta)
    assert not fits(tiler, np.zeros((3, 2, 5, 4, 4)), None)
    other = SimpleNamespace(position_name="unknown_001")
    assert not fits(other, np.zeros((3, 2, 5, 4, 4)), meta)
