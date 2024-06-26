"""Parse microscopy log files generated in the Swain lab."""

import re
import typing as t
from pathlib import PosixPath

from aliby.global_settings import possible_imaging_channels


def parse_swainlab_logs(filepath: t.Union[str, PosixPath]) -> t.Dict:
    """Parse and process a Swain lab microscopy log file."""
    raw_meta = first_parse(filepath)
    meta = raw_meta.copy()
    # convert raw_meta values into dicts
    for meta_key in [
        "exposure",
        "number_z_sections",
        "z_spacing",
        "sectioning_method",
    ]:
        meta[meta_key] = {
            channel: raw_meta[meta_key][i]
            for i, channel in enumerate(raw_meta["channels"])
        }
    meta["spatial_locations"] = {
        position: raw_meta["spatial_locations"][i]
        for i, position in enumerate(raw_meta["group"])
    }
    return meta


def first_parse(filepath: t.Union[str, PosixPath]) -> t.Dict:
    """Parse a Swain lab microscopy log file into a dict of lists."""
    meta: t.Dict[str, t.Union[t.List]] = {
        "channels": [],
        "exposure": [],
        "number_z_sections": [],
        "z_spacing": [],
        "sectioning_method": [],
        "group": [],
        "spatial_locations": [],
        "device": [],
    }
    general_setting = True
    acquisition_setting = False
    group_setting = False
    devices_setting = False
    with open(filepath, "r", encoding="UTF-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()
            # general information
            if general_setting and ":" in line:
                parse_general(line, meta)
            # acquisition settings
            if line == "-----Acquisition settings-----":
                acquisition_setting = True
                general_setting = False
                continue
            line_bits = [bit.strip() for bit in line.split(",")]
            if acquisition_setting:
                parse_acquisition(line_bits, meta)
            # information on devices
            if line == "Device properties:":
                devices_setting = True
                acquisition_setting = False
                continue
            if devices_setting:
                parse_devices(line_bits, meta)
            # information on groups
            if line == "Name,X,Y,Z,Autofocus offset":
                group_setting = True
                devices_setting = False
                continue
            if group_setting and not line:
                # blank line
                group_setting = False
                continue
            if group_setting:
                parse_group(line_bits, meta)
            # time settings
            add_to_meta("interval", line, meta, "time_settings/timeinterval")
            add_to_meta("frames", line, meta, "time_settings/ntimepoints")
            if line == "-----Experiment started-----":
                break
    return meta


def add_to_meta(search_word: str, line: str, meta: t.Dict, key: str) -> None:
    """
    Parse line for search_word and add following number to meta dict.

    E.g. interval: 300
    """
    values = re.findall(rf"{search_word}:\s*(\d+)", line)
    if values:
        value = int(values[0])
        if key in meta and meta[key] != value:
            print("Warning - metadata: {key} has different values.")
        else:
            meta[key] = value


def parse_general(line: str, meta: t.Dict) -> None:
    """Parse general information on the experiment."""
    bits = [bit.strip() for bit in line.split(":")]
    if re.search("[a-zA-Z+]", bits[0]):
        meta_key = bits[0].lower().replace(" ", "_")
        if meta_key == "omero_tags":
            meta[meta_key] = bits[1].split(",")
        else:
            meta[meta_key] = [":".join(bits[1:])]


def parse_acquisition(bits: t.List[str], meta: t.Dict) -> None:
    """Parse information on the imaging channels."""
    if (
        bits[0] in possible_imaging_channels
        and bits[1] in possible_imaging_channels
    ):
        meta["channels"].append(bits[0])
        meta["exposure"].append(float(bits[3]))
        meta["number_z_sections"].append(int(bits[4]))
        meta["z_spacing"].append(float(bits[5]))
        meta["sectioning_method"].append(bits[6])


def parse_group(bits: t.List[str], meta: t.Dict) -> None:
    """Parse information on the imaging groups."""
    meta["group"].append(bits[0])
    meta["spatial_locations"].append((float(bits[1]), float(bits[2])))


def parse_devices(bits: t.List[str], meta: t.Dict) -> None:
    """Parse information on the devices used in the experiment."""
    if bits[0] in possible_imaging_channels:
        meta["device"].append((bits[0], bits[1], bits[2], float(bits[3])))
