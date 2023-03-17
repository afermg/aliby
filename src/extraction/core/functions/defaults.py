# File with defaults for ease of use
import re
import typing as t
from pathlib import Path
import h5py

# should we move these functions here?
from aliby.tile.tiler import find_channel_name


def exparams_from_meta(
    meta: t.Union[dict, Path, str], extras: t.Collection[str] = ["ph"]
):
    """
    Obtain parameters from metadata of the h5 file.

    Compares a list of candidate channels using case-insensitive
    REGEX to identify valid channels.
    """
    meta = meta if isinstance(meta, dict) else load_metadata(meta)
    base = {
        "tree": {"general": {"None": ["area", "volume", "eccentricity"]}},
        "multichannel_ops": {},
    }
    candidate_channels = {
        "Citrine",
        "GFP",
        "GFPFast",
        "mCherry",
        "pHluorin405",
        "pHluorin488",
        "Flavin",
        "Cy5",
        "mKO2",
    }
    default_reductions = {"max"}
    default_metrics = {
        "mean",
        "median",
        "std",
        "imBackground",
        "max5px",
        # "nuc_est_conv",
    }
    # define ratiometric combinations
    # key is numerator and value is denominator
    # add more to support additional channel names
    ratiometric_combinations = {"phluorin405": ("phluorin488", "gfpfast")}
    default_reduction_metrics = {
        r: default_metrics for r in default_reductions
    }
    # default_rm["None"] = ["nuc_conv_3d"] # Uncomment this to add nuc_conv_3d (slow)
    extant_fluorescence_ch = []
    for av_channel in candidate_channels:
        # find matching channels in metadata
        found_channel = find_channel_name(meta.get("channels", []), av_channel)
        if found_channel is not None:
            extant_fluorescence_ch.append(found_channel)
    for ch in extant_fluorescence_ch:
        base["tree"][ch] = default_reduction_metrics
    base["sub_bg"] = extant_fluorescence_ch
    # additional extraction defaults if the channels are available
    if "ph" in extras:
        # SWAINLAB specific names
        # find first valid combination of ratiometric fluorescence channels
        numerator_channel, denominator_channel = (None, None)
        for ch1, chs2 in ratiometric_combinations.items():
            found_channel1 = find_channel_name(extant_fluorescence_ch, ch1)
            if found_channel1 is not None:
                numerator_channel = found_channel1
                for ch2 in chs2:
                    found_channel2 = find_channel_name(
                        extant_fluorescence_ch, ch2
                    )
                    if found_channel2:
                        denominator_channel = found_channel2
                        break
        # if two compatible ratiometric channels are available
        if numerator_channel is not None and denominator_channel is not None:
            sets = {
                b + a: (x, y)
                for a, x in zip(
                    ["", "_bgsub"],
                    (
                        [numerator_channel, denominator_channel],
                        [
                            f"{numerator_channel}_bgsub",
                            f"{denominator_channel}_bgsub",
                        ],
                    ),
                )
                for b, y in zip(["em_ratio", "gsum"], ["div0", "add"])
            }
            for i, v in sets.items():
                base["multichannel_ops"][i] = [
                    *v,
                    default_reduction_metrics,
                ]
    return base


def load_metadata(file: t.Union[str, Path], group="/"):
    """Get meta data from an h5 file."""
    with h5py.File(file, "r") as f:
        meta = dict(f[group].attrs.items())
    return meta
