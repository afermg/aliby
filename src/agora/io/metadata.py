"""
Anthology of interfaces for different parsers and lack of them.

ALIBY decides on using different metadata parsers based on two elements:

1. The parameter given by PipelineParameters (Either True/False, or a string pointing to the metadata file)
2. The available files in the root folder where images are found (remote or locally)

If parameters is a string pointing to a metadata file, ALIBY picks a parser based on the file format.
If parameters is True (as a boolean), ALIBY searches for any available file and uses the first valid one.
If there are no metadata files, ALIBY requires indicating indices for tiler, segmentation and extraction.

"""
import glob
import os
import typing as t
from datetime import datetime
from pathlib import Path, PosixPath

import pandas as pd
from pytz import timezone

from agora.io.writer import Writer
from logfile_parser import Parser
from logfile_parser.swainlab_parser import parse_from_swainlab_grammar


class MetaData:
    """Small metadata Process that loads log."""

    def __init__(self, log_dir, store):
        self.log_dir = log_dir
        self.store = store
        self.metadata_writer = Writer(self.store)

    def __getitem__(self, item):
        return self.load_logs()[item]

    def load_logs(self):
        # parsed_flattened = parse_logfiles(self.log_dir)
        parsed_flattened = dispatch_metadata_parser(self.log_dir)
        return parsed_flattened

    def run(self, overwrite=False):
        metadata_dict = self.load_logs()
        self.metadata_writer.write(
            path="/", meta=metadata_dict, overwrite=overwrite
        )

    def add_field(self, field_name, field_value, **kwargs):
        self.metadata_writer.write(
            path="/",
            meta={field_name: field_value},
            **kwargs,
        )

    def add_fields(self, fields_values: dict, **kwargs):
        for field, value in fields_values.items():
            self.add_field(field, value)


# Paradigm: able to do something with all datatypes present in log files,
# then pare down on what specific information is really useful later.

# Needed because HDF5 attributes do not support dictionaries
def flatten_dict(nested_dict, separator="/"):
    """
    Flattens nested dictionary. If empty return as-is.
    """
    flattened = {}
    if nested_dict:
        df = pd.json_normalize(nested_dict, sep=separator)
        flattened = df.to_dict(orient="records")[0] or {}

    return flattened


# Needed because HDF5 attributes do not support datetime objects
# Takes care of time zones & daylight saving
def datetime_to_timestamp(time, locale="Europe/London"):
    """
    Convert datetime object to UNIX timestamp
    """
    return timezone(locale).localize(time).timestamp()


def find_file(root_dir, regex):
    file = [
        f
        for f in glob.glob(os.path.join(str(root_dir), regex))
        if Path(f).name != "aliby.log"  # Skip filename reserved for aliby
    ]

    if len(file) > 1:
        print(
            "Warning:Metadata: More than one logfile found. Defaulting to first option."
        )
        file = [sorted(file)[0]]
    if len(file) == 0:
        print("Warning:Metadata: No valid swainlab .log found.")
    else:
        return file[0]
    return None


# TODO: re-write this as a class if appropriate
# WARNING: grammars depend on the directory structure of a locally installed
# logfile_parser repo
def parse_logfiles(
    root_dir,
    acq_grammar="multiDGUI_acq_format.json",
    log_grammar="multiDGUI_log_format.json",
):
    """
    Parse acq and log files depending on the grammar specified, then merge into
    single dict.
    """
    # Both acq and log files contain useful information.
    # ACQ_FILE = 'flavin_htb2_glucose_long_ramp_DelftAcq.txt'
    # LOG_FILE = 'flavin_htb2_glucose_long_ramp_Delftlog.txt'
    log_parser = Parser(log_grammar)
    acq_parser = Parser(acq_grammar)

    log_file = find_file(root_dir, "*log.txt")
    acq_file = find_file(root_dir, "*[Aa]cq.txt")

    parsed = {}
    if log_file and acq_file:
        with open(log_file, "r") as f:
            log_parsed = log_parser.parse(f)

        with open(acq_file, "r") as f:
            acq_parsed = acq_parser.parse(f)

        parsed = {**acq_parsed, **log_parsed}

    for key, value in parsed.items():
        if isinstance(value, datetime):
            parsed[key] = datetime_to_timestamp(value)

    parsed_flattened = flatten_dict(parsed)
    for k, v in parsed_flattened.items():
        if isinstance(v, list):
            parsed_flattened[k] = [0 if el is None else el for el in v]

    return parsed_flattened


def get_meta_swainlab(parsed_metadata: dict):
    """
    Convert raw parsing of Swainlab logfile to the metadata interface.

    Input:
    --------
    parsed_metadata: Dict[str, str or int or DataFrame or Dict]
    default['general', 'image_config', 'device_properties', 'group_position', 'group_time', 'group_config']

    Returns:
    --------
    Dictionary with metadata following the standard

    """
    channels = parsed_metadata["image_config"]["Image config"].values.tolist()
    # nframes = int(parsed_metadata["group_time"]["frames"].max())

    # return {"channels": channels, "nframes": nframes}
    return {"channels": channels}


def get_meta_from_legacy(parsed_metadata: dict):
    result = parsed_metadata
    result["channels"] = result["channels/channel"]
    return result


def parse_swainlab_metadata(filedir: t.Union[str, PosixPath]):
    """
    Dispatcher function that determines which parser to use based on the file ending.

    Input:
    --------
    filedir: Directory where the logfile is located.

    Returns:
    --------
    Dictionary with minimal metadata
    """
    filedir = Path(filedir)

    filepath = find_file(filedir, "*.log")
    if filepath:
        raw_parse = parse_from_swainlab_grammar(filepath)
        minimal_meta = get_meta_swainlab(raw_parse)
    else:
        if filedir.is_file():
            filedir = filedir.parent
        legacy_parse = parse_logfiles(filedir)
        minimal_meta = (
            get_meta_from_legacy(legacy_parse) if legacy_parse else {}
        )

    return minimal_meta


def dispatch_metadata_parser(filepath: t.Union[str, PosixPath]):
    """
    Function to dispatch different metadata parsers that convert logfiles into a
    basic metadata dictionary. Currently only contains the swainlab log parsers.

    Input:
    --------
    filepath: str existing file containing metadata, or folder containing naming conventions
    """
    parsed_meta = parse_swainlab_metadata(filepath)

    if parsed_meta is None:
        parsed_meta = dir_to_meta

    return parsed_meta


def dir_to_meta(path: PosixPath, suffix="tiff"):
    filenames = list(path.glob(f"*.{suffix}"))

    try:
        # Deduct order from filenames
        dimorder = "".join(
            map(lambda x: x[0], filenames[0].stem.split("_")[1:])
        )

        dim_value = list(
            map(
                lambda f: filename_to_dict_indices(f.stem),
                path.glob("*.tiff"),
            )
        )
        maxes = [max(map(lambda x: x[dim], dim_value)) for dim in dimorder]
        mins = [min(map(lambda x: x[dim], dim_value)) for dim in dimorder]
        _dim_shapes = [
            max_val - min_val + 1 for max_val, min_val in zip(maxes, mins)
        ]

        meta = {
            "size_" + dim: shape for dim, shape in zip(dimorder, _dim_shapes)
        }
    except Exception as e:
        print(
            f"Warning:Metadata: Cannot extract dimensions from filenames. Empty meta set {e}"
        )
        meta = {}

    return meta


def filename_to_dict_indices(stem: str):
    return {
        dim_number[0]: int(dim_number[1:])
        for dim_number in stem.split("_")[1:]
    }
