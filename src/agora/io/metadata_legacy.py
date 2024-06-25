"""Legacy code for acq and log files."""

import glob
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from logfile_parser import Parser
from pytz import timezone


def parse_legacy_logs(
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
    log_file = find_file_legacy(root_dir, "*log.txt")
    acq_file = find_file_legacy(root_dir, "*[Aa]cq.txt")
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
    # add spatial locations as a dict
    parsed_flattened["spatial_locations"] = {
        position: (
            parsed_flattened["positions/xpos"][i],
            parsed_flattened["positions/ypos"][i],
        )
        for i, position in enumerate(parsed_flattened["positions/posname"])
    }
    # update naming of channels field from the legacy convention
    parsed_flattened["channels"] = parsed_flattened["channels/channel"]
    # add watermark
    parsed_flattened["legacy"] = True
    return parsed_flattened


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


def find_file_legacy(root_dir, regex):
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
