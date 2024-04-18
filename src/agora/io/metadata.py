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
import numpy as np
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
        parsed_flattened = parse_metadata(self.log_dir)
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


def parse_metadata(filedir: t.Union[str, Path]):
    """
    Dispatch different metadata parsers that convert logfiles into a dictionary.

    Currently only contains the swainlab log parsers.

    Parameters
    --------
    filepath: str
        File containing metadata or folder containing naming conventions.
    """
    filedir = Path(filedir)
    if filedir.is_file() or str(filedir).endswith(".zarr"):
        # log file is in parent directory
        filedir = filedir.parent
    filepath = find_file(filedir, "*.log")
    if filepath:
        # new log files ending in .log
        raw_parse = parse_from_swainlab_grammar(filepath)
        minimal_meta = get_minimal_meta_swainlab(raw_parse)
    else:
        # legacy log files ending in .txt
        legacy_parse = parse_legacy_logfiles(filedir)
        minimal_meta = (
            get_meta_from_legacy(legacy_parse) if legacy_parse else {}
        )
    if minimal_meta is None:
        raise Exception("No metadata found.")
    else:
        return minimal_meta


def find_file(root_dir, regex):
    """Find files in a directory using regex."""
    # ignore aliby.log files
    file = [
        f
        for f in glob.glob(os.path.join(str(root_dir), regex))
        if Path(f).name != "aliby.log"
    ]
    if len(file) == 0:
        return None
    elif len(file) > 1:
        print(
            "Warning:Metadata: More than one log file found."
            " Defaulting to first option."
        )
        return sorted(file)[0]
    else:
        return file[0]


def get_minimal_meta_swainlab(parsed_metadata: dict):
    """
    Extract channels from parsed metadata.

    Parameters
    --------
    parsed_metadata: dict[str, str or int or DataFrame or Dict]
        default['general', 'image_config', 'device_properties',
                'group_position', 'group_time', 'group_config']

    Returns
    --------
    Dict with channels metadata
    """
    channels_dict = find_channels_by_position(parsed_metadata["group_config"])
    channels = parsed_metadata["image_config"]["Image config"].values.tolist()
    ntps = parsed_metadata["group_time"]["frames"].max()
    timeinterval = parsed_metadata["group_time"]["interval"].min()
    minimal_meta = {
        "channels_by_group": channels_dict,
        "channels": channels,
        "time_settings/ntimepoints": int(ntps),
        "time_settings/timeinterval": int(timeinterval),
    }
    return minimal_meta


def find_channels_by_position(meta):
    """
    Parse metadata to find the imaging channels for each group.

    Return a dict with groups as keys and channels as values.
    """
    if isinstance(meta, pd.DataFrame):
        imaging_channels = list(meta.columns)
        channels_dict = {group: [] for group in meta.index}
        for group in channels_dict:
            for channel in imaging_channels:
                if meta.loc[group, channel] is not None:
                    channels_dict[group].append(channel)
    elif isinstance(meta, dict) and "positions/posname" in meta:
        channels_dict = {
            position_name: [] for position_name in meta["positions/posname"]
        }
        imaging_channels = meta["channels"]
        for i, position_name in enumerate(meta["positions/posname"]):
            for imaging_channel in imaging_channels:
                if (
                    "positions/" + imaging_channel in meta
                    and meta["positions/" + imaging_channel][i]
                ):
                    channels_dict[position_name].append(imaging_channel)
    else:
        channels_dict = {}
    return channels_dict


### legacy code for acq and log files ###


def parse_legacy_logfiles(
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
    log_file = find_file(root_dir, "*log.txt")
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


def get_meta_from_legacy(parsed_metadata: dict):
    """Fix naming convention for channels in legacy .txt log files."""
    result = parsed_metadata
    result["channels"] = result["channels/channel"]
    return result


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
