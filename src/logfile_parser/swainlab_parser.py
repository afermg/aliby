"""Parse microscopy log files generated in the Swain lab."""

import logging
import re
from pathlib import Path

acquisition_header = "Image config,Channel,"
devices_header = "Image config,device,"
positions_header = "Name,X,Y,Z,Autofocus offset"
config_header = re.compile(r"group:.*field:\s*config")
by_channel = (
    "exposure",
    "number_z_sections",
    "z_spacing",
    "sectioning_method",
)


def parse_swainlab_logs(filepath: str | Path) -> dict:
    """
    Parse and process a Swain lab microscopy log file.

    Channels are named by their image configs, such as GFP_Z, which
    are the names the images have.
    """
    raw_meta = first_parse(filepath)
    meta = raw_meta.copy()
    channels = raw_meta["channels"]
    # convert raw_meta values into dicts
    for meta_key in by_channel:
        meta[meta_key] = dict(zip(channels, raw_meta[meta_key]))
    meta["spatial_locations"] = dict(
        zip(raw_meta["group"], raw_meta["spatial_locations"])
    )
    if raw_meta["channels_by_position"]:
        # in the order of the channels, as the images have them
        meta["channels_by_position"] = {
            position: [channel for channel in channels if channel in imaged]
            for position, imaged in raw_meta["channels_by_position"].items()
        }
    else:
        # a log that does not say which channels each group has
        del meta["channels_by_position"]
    return meta


def first_parse(filepath: str | Path) -> dict:
    """
    Parse a Swain lab microscopy log file into a dict of lists.

    The log is read up to the start of the experiment. Its tables each
    begin with a header and end with a blank line.
    """
    meta = {
        "channels": [],
        "exposure": [],
        "number_z_sections": [],
        "z_spacing": [],
        "sectioning_method": [],
        "group": [],
        "spatial_locations": [],
        "device": [],
        "channels_by_position": {},
    }
    section = "general"
    # the positions of the group being read
    positions = []
    with open(filepath, "r", encoding="UTF-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()
            if line == "-----Experiment started-----":
                break
            if line == "-----Acquisition settings-----":
                section = None
            elif section == "general":
                if ":" in line:
                    parse_general(line, meta)
            elif not line:
                # a blank line ends a table
                section = None
            elif line.startswith(acquisition_header):
                section = "acquisition"
            elif line.startswith(devices_header):
                section = "devices"
            elif line == positions_header:
                section = "positions"
                positions = []
            elif config_header.match(line):
                section = "config"
                # a group given again replaces what it gave before
                for position in positions:
                    meta["channels_by_position"][position] = set()
            elif section is None:
                # time settings
                add_to_meta(
                    "interval", line, meta, "time_settings/timeinterval"
                )
                add_to_meta("frames", line, meta, "time_settings/ntimepoints")
            else:
                bits = [bit.strip() for bit in line.split(",")]
                if section == "acquisition":
                    parse_acquisition(bits, meta)
                elif section == "devices":
                    parse_devices(bits, meta)
                elif section == "positions":
                    parse_group(bits, meta)
                    positions.append(bits[0])
                elif section == "config":
                    parse_config(line, positions, meta)
    return meta


def add_to_meta(search_word: str, line: str, meta: dict, key: str) -> None:
    """
    Parse line for search_word and add following number to meta dict.

    E.g. interval: 300
    """
    values = re.findall(rf"{search_word}:\s*(\d+)", line)
    if values:
        value = int(values[0])
        if key in meta and meta[key] != value:
            logging.getLogger("aliby").warning(
                f"Metadata: {key} has different values; using {meta[key]}."
            )
        else:
            meta[key] = value


def parse_general(line: str, meta: dict) -> None:
    """
    Parse general information on the experiment.

    Split a line at its first colon only, for the value may have its own.
    """
    key, value = line.split(":", 1)
    if re.search("[a-zA-Z+]", key):
        meta_key = key.strip().lower().replace(" ", "_")
        if meta_key == "omero_tags":
            meta[meta_key] = [tag.strip() for tag in value.split(",")]
        else:
            meta[meta_key] = [value.strip()]


def parse_acquisition(bits: list[str], meta: dict) -> None:
    """
    Parse a row of the table of image configs, one for each channel.

    The numbers are counted from the row's end because a description
    may have commas.
    """
    if len(bits) < 7:
        return
    meta["channels"].append(bits[0])
    meta["exposure"].append(float(bits[-4]))
    meta["number_z_sections"].append(int(bits[-3]))
    meta["z_spacing"].append(float(bits[-2]))
    meta["sectioning_method"].append(bits[-1])


def parse_group(bits: list[str], meta: dict) -> None:
    """Parse a position's name and where it is."""
    meta["group"].append(bits[0])
    meta["spatial_locations"].append((float(bits[1]), float(bits[2])))


def parse_devices(bits: list[str], meta: dict) -> None:
    """Parse a row of the devices used for a channel."""
    if len(bits) < 4:
        return
    try:
        value = float(bits[3])
    except ValueError:
        value = bits[3]
    meta["device"].append((bits[0], bits[1], bits[2], value))


def parse_config(line: str, positions: list[str], meta: dict) -> None:
    """
    Parse a channel of a group and the time points it is imaged at.

    The time points are the bits of a hexadecimal number, such as
    0xffff, and a channel imaged at none is not a channel of the group.
    """
    channel, _, time_points = line.rpartition(":")
    try:
        imaged = int(time_points, 16) != 0
    except ValueError:
        return
    if imaged:
        for position in positions:
            meta["channels_by_position"][position].add(channel.strip())
