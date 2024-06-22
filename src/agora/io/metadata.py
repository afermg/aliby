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
import os
import typing as t
from pathlib import Path

import pandas as pd
from agora.io import metadata_legacy
from agora.io.writer import Writer
from logfile_parser.swainlab_parser_new import parse_from_swainlab_grammar


class MetaData:
    """Metadata process that loads and parses log files."""

    def __init__(self, log_dir, store):
        """Initialise with log-file directory and h5 location to write."""
        self.log_dir = log_dir
        self.store = store
        self.metadata_writer = Writer(self.store)

    def load_logs(self):
        """Load log using a hierarchy of parsers."""
        logs_metadata = parse_logs(self.log_dir)
        return logs_metadata

    def run(self, overwrite=False):
        """
        Load and parse logs and write to h5 file.

        Used by pipline.py.
        """
        metadata_dict = self.load_logs()
        self.metadata_writer.write(
            path="/", meta=metadata_dict, overwrite=overwrite
        )


def parse_logs(filedir: t.Union[str, Path]):
    """
    Dispatch metadata parsers to parse logfiles into a dict.

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
        # log files ending in .log
        raw_parse = parse_from_swainlab_grammar(filepath)
        minimal_meta = get_minimal_meta_swainlab(raw_parse)
    else:
        # legacy log files ending in .txt
        legacy_parse = metadata_legacy.parse_legacy_logs(filedir)
        minimal_meta = (
            metadata_legacy.get_meta_from_legacy(legacy_parse)
            if legacy_parse
            else {}
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
        if "aliby" not in Path(f).name
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
    parsed_ntps = parsed_metadata["group_time"]["frames"]
    if type(parsed_ntps) is int:
        ntps = parsed_ntps
    else:
        ntps = parsed_ntps.max()
    parsed_tinterval = parsed_metadata["group_time"]["interval"]
    if type(parsed_tinterval) is int:
        timeinterval = parsed_tinterval
    else:
        timeinterval = parsed_tinterval.min()
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
