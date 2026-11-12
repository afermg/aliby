"""
Say what to extract and hold what was extracted.

An extraction tree names channels, then how to reduce their z sections,
then the functions to apply: {"GFP": {"max": ["mean", "median"]}}.
"""

import copy
import logging
import re
import typing as t
from pathlib import Path

import h5py
import pandas as pd
from agora.abc import ParametersABC
from aliby.global_settings import global_settings

# define types
reduction_method = t.Union[t.Callable, str]
extraction_tree = t.Dict[
    str, t.Dict[reduction_method, t.Dict[str, t.Collection]]
]
extraction_result = t.Dict[
    str, t.Dict[reduction_method, t.Dict[str, t.Dict[str, pd.Series]]]
]


def build_extraction_tree_from_meta(meta: t.Union[dict, Path, str]):
    """Build extraction-tree dict from microscopy metadata."""
    if not isinstance(meta, dict):
        # load meta data
        with h5py.File(meta, "r") as f:
            meta = dict(f["/"].attrs.items())
    tree_dict = {
        "tree": {"general": {"null": global_settings.outline_functions}},
        "multichannel_funs": {},
    }
    candidate_channels = global_settings.possible_imaging_channels
    default_reductions = {"max", "mean"}
    default_fluorescence_metrics = global_settings.fluorescence_functions
    default_reduction_and_fluorescence_metrics = {
        r: default_fluorescence_metrics for r in default_reductions
    }
    extant_fluorescence_ch = []
    for av_channel in candidate_channels:
        # find every matching channel in the metadata, not only the first:
        # GFP matches GFPFast too, and stopping there left a GFP channel
        # listed after it unextracted
        for channel in meta.get("channels", []):
            if (
                re.match(av_channel, channel, re.IGNORECASE)
                and channel not in extant_fluorescence_ch
            ):
                extant_fluorescence_ch.append(channel)
    not_extracted = [
        channel
        for channel in meta.get("channels", [])
        if channel not in extant_fluorescence_ch
        and str(channel).lower() != "brightfield"
    ]
    if not_extracted:
        logging.getLogger("aliby").warning(
            f"Extractor: no fluorescence will be extracted from "
            f"{', '.join(map(str, not_extracted))}, which begins with none "
            "of global_settings.possible_imaging_channels. Add the name "
            "there, or give the channel in the extraction tree."
        )
    background_channels = {
        ch for ch in extant_fluorescence_ch if ch.lower() == "cy5"
    }
    for ch in extant_fluorescence_ch:
        if ch in background_channels:
            # background channels get only outside-cell metrics
            tree_dict["tree"][ch] = {
                "mean": [
                    "median_background",
                    "mean_background",
                    "std_background",
                ]
            }
        else:
            tree_dict["tree"][ch] = copy.deepcopy(
                default_reduction_and_fluorescence_metrics
            )
    # subtract background from all fluorescence channels
    tree_dict["subtract_background"] = set(extant_fluorescence_ch)
    # extract intracellular masks for non-background channels only
    tree_dict["intracellular_masks"] = {
        ch for ch in extant_fluorescence_ch if ch not in background_channels
    }
    tree_dict["identify_vacuoles"] = True
    tree_dict["background_channels"] = background_channels
    tree_dict["mask_pdms"] = True
    tree_dict["exclude_obscured"] = True
    return tree_dict


def recursive_merge_extractor(dict1, dict2):
    """
    Merge the second extractor tree into the first and return the first.

    Dicts are merged and sets are joined, but a list in the second
    replaces the list in the first: to give the functions for a channel
    and a reduction is to give all of them. To add a function to every
    channel, append it to global_settings.fluorescence_functions
    instead, as examples/run_custom_function.py does.
    """
    for key, value in dict2.items():
        if (
            key in dict1
            and isinstance(dict1[key], dict)
            and isinstance(value, dict)
        ):
            dict1[key] = recursive_merge_extractor(dict1[key], value)
        elif isinstance(dict1.get(key), set) and isinstance(value, set):
            # merge sets at leaf node
            dict1[key] = dict1[key].union(value)
        else:
            # overwrite scalars and new keys
            dict1[key] = value
    return dict1


class ExtractorParameters(ParametersABC):
    """Base class to define parameters for extraction."""

    def __init__(
        self,
        tree: extraction_tree,
        subtract_background: set = set(),
        multichannel_funs: t.Dict = {},
        intracellular_masks: set = set(),
        identify_vacuoles: bool = True,
        background_channels: set = set(),
        mask_pdms: bool = True,
        exclude_obscured: bool = True,
    ):
        """
        Initialise.

        Parameters
        ----------
        tree: dict
            Nested dictionary indicating channels, reduction functions
            and metrics to be used.
            str channel -> str reduction -> str metric
            The "null" reduction indicates no z-reduction is needed.
            If not of depth three, tree will be filled with "null".
        subtract_background: set
            A set of strings of the channels to correct for
            background.
        multichannel_funs: dict
            Dict of multichannel functions.
        intracellular_masks: set
            A set of strings of the channels for which to extract
            vacuole and cytoplasm metrics.
        identify_vacuoles: bool
            If True (default), compute vacuole and cytoplasm sub-masks
            when maby is installed. Set to False to skip vacuole
            detection regardless of package availability.
        background_channels: set
            A set of strings of the channels for which to extract
            the median background signal (pixels outside all cells).
            Defaults to Cy5 channels when built from metadata.
        mask_pdms: bool
            If True (default), find the PDMS trap in the brightfield
            images and exclude it from all background estimates. The
            PDMS is autofluorescent and so is not background.
        exclude_obscured: bool
            If True (default), measure no cell that reaches the edge of
            its tile, or the edge of the image where a tile hangs off
            it: such a cell is cut off, and its area, volume and
            fluorescence are those of the visible part. It is still
            segmented, tracked and stored, and its pixels are still
            kept out of every background estimate.
        """
        self.tree = tree
        self.subtract_background = subtract_background
        self.multichannel_funs = multichannel_funs
        self.intracellular_masks = intracellular_masks
        self.identify_vacuoles = identify_vacuoles
        self.background_channels = background_channels
        self.mask_pdms = mask_pdms
        self.exclude_obscured = exclude_obscured

    @classmethod
    def default(cls):
        """Override ParametersABC default class method."""
        return cls({})

    @classmethod
    def from_meta(cls, meta):
        """Instantiate from the meta data; used by Pipeline."""
        return cls(**build_extraction_tree_from_meta(meta))


def flatten_nesteddict(
    nest: dict, to="series", tp: int = None
) -> t.Dict[str, pd.Series]:
    """
    Convert a nested extraction dict into a dict of pd.Series.

    Parameters
    ----------
    nest: dict of dicts
        Contains the nested results of extraction.
    to: str (optional)
        Specifies the format of the output, either pd.Series
        (default) or a list.
    tp: int
        Time point used to name the pd.Series.

    Returns
    -------
    d: dict
        A dict with a concatenated string of channel, reduction
        metric, and cell metric as keys and either a pd.Series or
        a list of the corresponding extracted data as values.
    """
    d = {}
    for k0, v0 in nest.items():
        for k1, v1 in v0.items():
            for k2, v2 in v1.items():
                d["/".join((k0, k1, k2))] = (
                    pd.Series(*v2, name=tp) if to == "series" else v2
                )
    return d
