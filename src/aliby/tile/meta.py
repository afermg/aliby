#!/usr/bin/env jupyter
"""
Parsing and heuristics to select channels
"""
from agora.io.metadata import find_channels_by_position


def find_channel_swainlab(meta: dict[str], position_name: str):
    """
    Apply a series of heuristics to find the correct metadata channel.
    Specific to the Swain Lab metadata structure.
    """
    channel_dict = {}
    # get channels for this position
    if "channels_by_group" in meta:
        channel_dict = meta["channels_by_group"]
    elif "positions/posname" in meta:
        # old meta data from image
        channel_dict = find_channels_by_position(meta["positions/posname"])
    channels = []
    if channel_dict:
        # TODO maybe we should use image.shape and image.dimoder instead?
        channels = channel_dict.get(
            self.position_name,
            list(range(meta.get("size_c", 0))),
        )

    if not channels:
        # image meta data contains channels for that image
        channels = meta.get("channels", list(range(meta.get("size_c", 0))))

    # sort channels based on OMERO's channel order
    if "OMERO_channels" in kwargs:
        channels = [
            ch for och in OMERO_channels for ch in channels if ch == och
        ]
