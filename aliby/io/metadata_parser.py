"""
Parse microscopy log files according to specified JSON grammars.
Produces dictionary to include in HDF5
"""
import glob
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pytz import timezone

from logfile_parser import Parser

# Paradigm: able to do something with all datatypes present in log files,
# then pare down on what specific information is really useful later.

# Needed because HDF5 attributes do not support dictionaries
def flatten_dict(nested_dict, separator="/"):
    """
    Flattens nested dictionary
    """
    df = pd.json_normalize(nested_dict, sep=separator)
    return df.to_dict(orient="records")[0]


# Needed because HDF5 attributes do not support datetime objects
# Takes care of time zones & daylight saving
def datetime_to_timestamp(time, locale="Europe/London"):
    """
    Convert datetime object to UNIX timestamp
    """
    return timezone(locale).localize(time).timestamp()


def find_file(root_dir, regex):
    file = glob.glob(os.path.join(str(root_dir), regex))
    if len(file) != 1:
        return None
    else:
        return file[0]


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
    try:
        log_file = find_file(root_dir, "*log.txt")
    except:
        raise ValueError("Experiment log file not found.")
    with open(log_file, "r") as f:
        log_parsed = log_parser.parse(f)

    acq_parser = Parser(acq_grammar)
    try:
        acq_file = find_file(root_dir, "*[Aa]cq.txt")
    except:
        raise ValueError("Experiment acq file not found.")
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
