"""
Aliby decides on using different metadata parsers based on two elements:
1. The parameter given by PipelineParameters (either True/False or a string
pointing to the metadata file)
2. The available files in the root folder where images are found (either
remote or locally).

If parameters is a string pointing to a metadata file, Aliby picks a parser
based on the file format.
If parameters is True, Aliby searches for any available file and uses the
first valid one.
If there are no metadata files, Aliby requires indices in the tiff file names
for tiler, segmentation, and extraction.

WARNING: grammars depend on the directory structure of a local log-file_parser
repository.
"""
import glob
import logging
import os
import typing as t
from datetime import datetime
from pathlib import Path

import pandas as pd
from pytz import timezone

from agora.io.writer import Writer
from logfile_parser import Parser
from logfile_parser.swainlab_parser import parse_from_swainlab_grammar


class MetaData:
    """Metadata process that loads and parses log files."""

    def __init__(self, log_dir, store):
        """Initialise with log-file directory and h5 location to write."""
        self.log_dir = log_dir
        self.store = store
        self.metadata_writer = Writer(self.store)

    def __getitem__(self, item):
        """Load log and access item in resulting meta data dictionary."""
        return self.load_logs()[item]

    def load_logs(self):
        """Load log using a hierarchy of parsers."""
        parsed_flattened = dispatch_metadata_parser(self.log_dir)
        return parsed_flattened

    def run(self, overwrite=False):
        """Load and parse logs and write to h5 file."""
        metadata_dict = self.load_logs()
        self.metadata_writer.write(
            path="/", meta=metadata_dict, overwrite=overwrite
        )

    def add_field(self, field_name, field_value, **kwargs):
        """Write a field and its values to the h5 file."""
        self.metadata_writer.write(
            path="/",
            meta={field_name: field_value},
            **kwargs,
        )

    def add_fields(self, fields_values: dict, **kwargs):
        """Write a dict of fields and values to the h5 file."""
        for field, value in fields_values.items():
            self.add_field(field, value)


def flatten_dict(nested_dict, separator="/"):
    """
    Flatten nested dictionary because h5 attributes cannot be dicts.

    If empty return as-is.
    """
    flattened = {}
    if nested_dict:
        df = pd.json_normalize(nested_dict, sep=separator)
        flattened = df.to_dict(orient="records")[0] or {}
    return flattened


def datetime_to_timestamp(time, locale="Europe/London"):
    """Convert datetime object to UNIX timestamp."""
    # h5 attributes do not support datetime objects
    return timezone(locale).localize(time).timestamp()


def find_file(root_dir, regex):
    """Find files in a directory using regex."""
    # ignore aliby.log files
    file = [
        f
        for f in glob.glob(os.path.join(str(root_dir), regex))
        if Path(f).name != "aliby.log"
    ]
    if len(file) == 0:
        logging.getLogger("aliby").log(
            logging.WARNING, "Metadata: No valid swainlab .log found."
        )
        return None
    elif len(file) > 1:
        print(
            "Warning:Metadata: More than one log file found."
            " Defaulting to first option."
        )
        return sorted(file)[0]
    else:
        return file[0]


def parse_logfiles(
    root_dir,
    acq_grammar="multiDGUI_acq_format.json",
    log_grammar="multiDGUI_log_format.json",
):
    """
    Parse acq and log files using the grammar specified.

    Merge results into a single dict.
    """
    log_parser = Parser(log_grammar)
    acq_parser = Parser(acq_grammar)
    # an example acq file is 'flavin_htb2_glucose_long_ramp_DelftAcq.txt'
    log_file = find_file(root_dir, "*log.txt")
    # an example log file is 'flavin_htb2_glucose_long_ramp_Delftlog.txt'
    acq_file = find_file(root_dir, "*[Aa]cq.txt")
    # parse into a single dict
    parsed = {}
    if log_file and acq_file:
        with open(log_file, "r") as f:
            log_parsed = log_parser.parse(f)
        with open(acq_file, "r") as f:
            acq_parsed = acq_parser.parse(f)
        parsed = {**acq_parsed, **log_parsed}
    # convert data to having time stamps
    for key, value in parsed.items():
        if isinstance(value, datetime):
            parsed[key] = datetime_to_timestamp(value)
    # flatten dict
    parsed_flattened = flatten_dict(parsed)
    for k, v in parsed_flattened.items():
        if isinstance(v, list):
            # replace None with 0
            parsed_flattened[k] = [0 if el is None else el for el in v]
    return parsed_flattened


def get_meta_swainlab(parsed_metadata: dict):
    """
    Convert raw parsing of Swainlab logfile to the metadata interface.

    Parameters
    --------
    parsed_metadata: dict[str, str or int or DataFrame or Dict]
        default['general', 'image_config', 'device_properties',
                'group_position', 'group_time', 'group_config']

    Returns
    --------
    Dict with channels metadata
    """
    channels = parsed_metadata["image_config"]["Image config"].values.tolist()
    return {"channels": channels}


def get_meta_from_legacy(parsed_metadata: dict):
    """Fix naming convention for channels in legacy .txt log files."""
    result = parsed_metadata
    result["channels"] = result["channels/channel"]
    return result


def parse_swainlab_metadata(filedir: t.Union[str, Path]):
    """Parse new, .log, and old, .txt, files in a directory into a dict."""
    filedir = Path(filedir)
    filepath = find_file(filedir, "*.log")
    if filepath:
        # new log files ending in .log
        raw_parse = parse_from_swainlab_grammar(filepath)
        minimal_meta = get_meta_swainlab(raw_parse)
    else:
        # old log files ending in .txt
        if filedir.is_file() or str(filedir).endswith(".zarr"):
            # log file is in parent directory
            filedir = filedir.parent
        legacy_parse = parse_logfiles(filedir)
        minimal_meta = (
            get_meta_from_legacy(legacy_parse) if legacy_parse else {}
        )
    return minimal_meta


def dispatch_metadata_parser(filepath: t.Union[str, Path]):
    """
    Dispatch different metadata parsers that convert logfiles into a dictionary.

    Currently only contains the swainlab log parsers.

    Parameters
    --------
    filepath: str
        File containing metadata or folder containing naming conventions.
    """
    parsed_meta = parse_swainlab_metadata(filepath)
    if parsed_meta is None:
        # try to deduce metadata
        parsed_meta = dir_to_meta
    return parsed_meta


def dir_to_meta(path: Path, suffix="tiff"):
    """Deduce meta data from the naming convention of tiff files."""
    filenames = list(path.glob(f"*.{suffix}"))
    try:
        # deduce order from filenames
        dim_order = "".join(
            map(lambda x: x[0], filenames[0].stem.split("_")[1:])
        )
        dim_value = list(
            map(
                lambda f: filename_to_dict_indices(f.stem),
                path.glob("*.tiff"),
            )
        )
        maxs = [max(map(lambda x: x[dim], dim_value)) for dim in dim_order]
        mins = [min(map(lambda x: x[dim], dim_value)) for dim in dim_order]
        dim_shapes = [
            max_val - min_val + 1 for max_val, min_val in zip(maxs, mins)
        ]
        meta = {
            "size_" + dim: shape for dim, shape in zip(dim_order, dim_shapes)
        }
    except Exception as e:
        print(
            "Warning:Metadata: Cannot extract dimensions from filenames."
            f" Empty meta set {e}"
        )
        meta = {}
    return meta


def filename_to_dict_indices(stem: str):
    """Convert a file name into a dict by splitting."""
    return {
        dim_number[0]: int(dim_number[1:])
        for dim_number in stem.split("_")[1:]
    }
