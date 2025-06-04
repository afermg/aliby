"""Parse metadata from microscopy logs."""

import glob
import os
import typing as t
from pathlib import Path

from agora.io import metadata_legacy
from logfile_parser.swainlab_parser import parse_swainlab_logs


class MetaData:
    """Metadata process to load and parse log files."""

    def __init__(self, log_dir, OMERO_channels: t.List[str] = None):
        """Initialise by loading and parsing microscopy logs."""
        self.log_dir = log_dir
        self.full = parse_microscopy_logs(log_dir)
        if OMERO_channels is not None:
            # OMERO overrules metadata from logs
            self.full["channels"] = OMERO_channels
        # add channels per position
        if "legacy" in self.full:
            self.full["channels_by_position"] = (
                metadata_legacy.find_channels_by_position_legacy(self.full)
            )

    @property
    def minimal(self) -> t.Dict:
        """Get minimal microscopy metadata to write to h5 file."""
        if not hasattr(self, "_minimal_meta"):
            self._minimal_meta = get_minimal_meta_swainlab(self.full)
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
    minimal_meta = {
        key: full_metadata[key]
        for key in [
            "channels",
            "time_settings/ntimepoints",
            "time_settings/timeinterval",
        ]
    }
    return minimal_meta
