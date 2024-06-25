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
from logfile_parser.swainlab_parser import parse_swainlab_logs


class MetaData:
    """Metadata process that loads and parses log files."""

    def __init__(self, log_dir):
        """Initialise by loading and parsing microscopy logs."""
        self.log_dir = log_dir
        self.full = parse_microscopy_logs(log_dir)

    @property
    def minimal(self) -> t.Dict:
        """Get minimal microscopy metadata to write to h5 file."""
        if not hasattr(self, "_minimal_meta"):
            if hasattr(self, "full"):
                if "legacy" in self.full:
                    self._minimal_meta = metadata_legacy.get_meta_from_legacy(
                        self.full
                    )
                else:
                    self._minimal_meta = get_minimal_meta_swainlab(self.full)
            else:
                self._minimal_meta = {}
        return self._minimal_meta


def parse_microscopy_logs(filedir: t.Union[str, Path]) -> t.Dict:
    """
    Dispatch metadata parsers to parse microscopy logfiles.

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
        full_meta = parse_swainlab_logs(filepath)
    else:
        # legacy log files ending in .txt
        full_meta = metadata_legacy.parse_legacy_logs(filedir)
    if full_meta is None:
        raise Exception("No microscopy metadata found.")
    else:
        return full_meta


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


def get_minimal_meta_swainlab(full_metadata: t.Dict) -> t.Dict:
    """Extract channels and time settings from microscopy metadata."""
    # TODO "channels_by_group": channels_dict,
    minimal_meta = {
        key: full_metadata[key]
        for key in [
            "channels",
            "time_settings/ntimepoints",
            "time_settings/timeinterval",
        ]
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
