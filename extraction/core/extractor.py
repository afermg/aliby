import os
from pathlib import Path, PosixPath
import pkg_resources
from collections.abc import Iterable

# from copy import copy
from typing import Union, List, Dict, Callable
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix, vstack, issparse
from tqdm import tqdm


from extraction.core.functions.loaders import (
    load_funs,
    load_custom_args,
    load_redfuns,
    load_mergefuns,
)
from extraction.core.functions.defaults import exparams_from_meta
from extraction.core.functions.distributors import trap_apply, reduce_z
from extraction.core.functions.utils import depth

from agora.abc import ProcessABC, ParametersABC
from agora.io.writer import Writer, load_attributes
from agora.io.cells import Cells
from aliby.tile.tiler import Tiler

CELL_FUNS, TRAPFUNS, FUNS = load_funs()
CUSTOM_FUNS, CUSTOM_ARGS = load_custom_args()
RED_FUNS = load_redfuns()
MERGE_FUNS = load_mergefuns()

# Assign datatype depending on the metric used
m2type = {"mean": np.float32, "median": np.ubyte, "imBackground": np.ubyte}


class ExtractorParameters(ParametersABC):
    """
    Base class to define parameters for extraction
    :tree: dict of depth n. If not of depth three tree will be filled with Nones
        str channel -> U(function,None) reduction -> str metric

    """

    def __init__(
        self,
        tree: Dict[Union[str, None], Dict[Union[Callable, None], List[str]]] = None,
        sub_bg: set = set(),
        multichannel_ops: Dict = {},
    ):

        self.tree = fill_tree(tree)

        self.sub_bg = sub_bg
        self.multichannel_ops = multichannel_ops

    @staticmethod
    def guess_from_meta(store_name: str, suffix="fast"):
        """
        Make a guess on the parameters using the hdf5 metadata

        Add anything as a suffix, default "fast"


        Parameters:
            store_name : str or Path indicating the results' storage.
            suffix : str to add at the end of the predicted parameter set
        """

        with h5py.open(store_name, "r") as f:
            microscope = f["/"].attrs.get("microscope")  # TODO Check this with Arin
        assert microscope, "No metadata found"

        return "_".join((microscope, suffix))

    @classmethod
    def default(cls):
        return cls({})

    @classmethod
    def from_meta(cls, meta):
        return cls(**exparams_from_meta(meta))


class Extractor(ProcessABC):
    """
        Base class to perform feature extraction.

        Parameters
        ----------
        parameters: core.extractor Parameters
            Parameters that include with channels, reduction and
    b            extraction functions to use.
        store: str
            Path to hdf5 storage file. Must contain cell outlines.
        tiler: pipeline-core.core.segmentation tiler
            Class that contains or fetches the image to be used for segmentation.
    """

    default_meta = {"pixel_size": 0.236, "z_size": 0.6, "spacing": 0.6}

    def __init__(
        self, parameters: ExtractorParameters, store: str = None, tiler: Tiler = None
    ):
        self.params = parameters
        if store:
            self.local = store
            self.load_meta()
        else:  # In case no h5 file is used, just use the parameters straight ahead
            self.meta = {"channel": parameters.to_dict()["tree"].keys()}
        if tiler:
            self.tiler = tiler
        self.load_funs()

    @classmethod
    def from_tiler(cls, parameters: ExtractorParameters, store: str, tiler: Tiler):
        return cls(parameters, store=store, tiler=tiler)

    @classmethod
    def from_img(cls, parameters: ExtractorParameters, store: str, img_meta: tuple):
        return cls(parameters, store=store, tiler=Tiler(*img_meta))

    # @classmethod
    # def from_store(cls, parameters: ExtractorParameters, store: str, img_meta: tuple):
    #     return cls(parameters, store=store, tiler=Tiler(*img_meta))

    @property
    def channels(self):
        if not hasattr(self, "_channels"):
            if type(self.params.tree) is dict:
                self._channels = tuple(self.params.tree.keys())

        return self._channels

    @property
    def current_position(self):
        return self.local.split("/")[-1][:-3]

    @property
    def group(self):  # Path within hdf5
        if not hasattr(self, "_out_path"):
            self._group = "/extraction/"
        return self._group

    @property
    def pos_file(self, store_name="store.h5"):
        if not hasattr(self, "_pos_file"):
            return self.local

    def load_custom_funs(self):
        """
        Load parameters of functions that require them from expt.
        These must be loaded within the Extractor instance because their parameters
        depend on their experiment's metadata.
        """
        funs = set(
            [
                fun
                for ch in self.params.tree.values()
                for red in ch.values()
                for fun in red
            ]
        )
        funs = funs.intersection(CUSTOM_FUNS.keys())
        ARG_VALS = {
            k: {k2: self.get_meta(k2) for k2 in v} for k, v in CUSTOM_ARGS.items()
        }
        # self._custom_funs = {trap_apply(CUSTOM_FUNS[fun],])
        self._custom_funs = {}
        for k, f in CUSTOM_FUNS.items():

            def tmp(f):
                return lambda m, img: trap_apply(f, m, img, **ARG_VALS.get(k, {}))

            self._custom_funs[k] = tmp(f)

    def load_funs(self):
        self.load_custom_funs()
        self._all_cell_funs = set(self._custom_funs.keys()).union(CELL_FUNS)
        self._all_funs = {**self._custom_funs, **FUNS}

    def load_meta(self):
        self.meta = load_attributes(self.local)

    def get_traps(
        self, tp: int, channels: list = None, z: list = None, **kwargs
    ) -> tuple:
        if channels is None:
            channel_ids = list(range(len(self.tiler.channels)))
        elif len(channels):
            channel_ids = [self.tiler.get_channel_index(ch) for ch in channels]
        else:
            channel_ids = None

        if z is None:
            z = list(range(self.tiler.shape[-1]))

        traps = (
            self.tiler.get_traps_timepoint(tp, channels=channel_ids, z=z, **kwargs)
            if channel_ids
            else None
        )

        return traps

    def extract_traps(
        self,
        traps: List[np.array],
        masks: List[np.array],
        metric: str,
        labels: List[int] = None,
    ) -> dict:
        """
        Apply a function for a whole position.

        :traps: List[np.array] list of images
        :masks: List[np.array] list of masks
        :metric:str metric to extract
        :labels: List[int] cell Labels to use as indices for output DataFrame
        :pos_info: bool Whether to add the position as index or not

        returns
        :d: Dictionary of dataframe
        """

        if labels is None:
            raise Warning("No labels given. Sorting cells using index.")

        cell_fun = True if metric in self._all_cell_funs else False

        idx = []
        results = []

        for trap_id, (mask_set, trap, lbl_set) in enumerate(
            zip(masks, traps, labels.values())
        ):
            if len(mask_set):  # ignore empty traps
                result = self._all_funs[metric](mask_set, trap)
                if cell_fun:
                    for lbl, val in zip(lbl_set, result):
                        results.append(val)
                        idx.append((trap_id, lbl))
                else:
                    results.append(result)
                    idx.append(trap_id)

        return (tuple(results), tuple(idx))

    def extract_funs(
        self, traps: List[np.array], masks: List[np.array], metrics: List[str], **kwargs
    ) -> dict:
        """
        Extract multiple metrics from a timepoint
        """
        d = {
            metric: self.extract_traps(
                traps=traps, masks=masks, metric=metric, **kwargs
            )
            for metric in metrics
        }

        return d

    def reduce_extract(
        self, traps: Union[np.array, None], masks: list, red_metrics: dict, **kwargs
    ) -> dict:
        """
        :param red_metrics: dict in which keys are reduction funcions and
        values are strings indicating the metric function
        :**kwargs: All other arguments, must include masks and traps.
        """

        reduced_traps = {}
        if traps is not None:
            for red_fun in red_metrics.keys():
                reduced_traps[red_fun] = [
                    self.reduce_dims(trap, method=RED_FUNS[red_fun]) for trap in traps
                ]

        d = {
            red_fun: self.extract_funs(
                metrics=metrics,
                traps=reduced_traps.get(red_fun, [None for _ in masks]),
                masks=masks,
                **kwargs,
            )
            for red_fun, metrics in red_metrics.items()
        }
        return d

    def reduce_dims(self, img: np.array, method=None) -> np.array:
        # assert len(img.shape) == 3, "Incorrect number of dimensions"

        if method is None:
            return img

        return reduce_z(img, method)

    def extract_tp(
        self, tp: int, tree: dict = None, tile_size: int = 117, **kwargs
    ) -> dict:
        """
        :param tp: int timepoint from which to extract results
        :param tree: dict of dict {channel : {reduction_function : metrics}}
        :**kwargs: Must include masks and preferably labels.
        """

        if tree is None:
            tree = self.params.tree

        ch_tree = {ch: v for ch, v in tree.items() if ch != "general"}
        tree_chs = (*ch_tree,)

        cells = Cells.hdf(self.local)

        # labels
        raw_labels = cells.labels_at_time(tp)
        labels = {
            trap_id: raw_labels.get(trap_id, []) for trap_id in range(cells.ntraps)
        }

        # masks
        raw_masks = cells.at_time(tp, kind="mask")

        masks = {trap_id: [] for trap_id in range(cells.ntraps)}
        for trap_id, cells in raw_masks.items():
            if len(cells):
                masks[trap_id] = np.dstack(np.array(cells)).astype(bool)

        masks = [np.array(v) for v in masks.values()]

        # traps
        traps = self.get_traps(tp, tile_size=tile_size, channels=tree_chs)

        self.img_bgsub = {}
        if self.params.sub_bg:
            bg = [
                ~np.sum(m, axis=2).astype(bool)
                if np.any(m)
                else np.zeros((tile_size, tile_size))
                for m in masks
            ]

        d = {}

        for ch, red_metrics in tree.items():
            img = None
            # ch != is necessary for threading
            if ch != "general" and traps is not None and len(traps):
                img = traps[:, tree_chs.index(ch), 0]

            d[ch] = self.reduce_extract(
                red_metrics=red_metrics, traps=img, masks=masks, labels=labels, **kwargs
            )

            if (
                ch in self.params.sub_bg and img is not None
            ):  # Calculate metrics with subtracted bg
                ch_bs = ch + "_bgsub"

                self.img_bgsub[ch_bs] = []
                for trap, maskset in zip(img, bg):

                    cells_fl = np.zeros_like(trap)

                    is_cell = np.where(maskset)
                    if len(is_cell[0]):  # skip calculation for empty traps
                        cells_fl = np.median(trap[is_cell], axis=0)

                    self.img_bgsub[ch_bs].append(trap - cells_fl)

                d[ch_bs] = self.reduce_extract(
                    red_metrics=ch_tree[ch],
                    traps=self.img_bgsub[ch_bs],
                    masks=masks,
                    labels=labels,
                    **kwargs,
                )

        # Additional operations between multiple channels (e.g. pH calculations)
        for name, (chs, merge_fun, red_metrics) in self.params.multichannel_ops.items():
            if len(
                set(chs).intersection(set(self.img_bgsub.keys()).union(tree_chs))
            ) == len(chs):
                imgs = [self.get_imgs(ch, traps, tree_chs) for ch in chs]
                merged = MERGE_FUNS[merge_fun](*imgs)
                d[name] = self.reduce_extract(
                    red_metrics=red_metrics,
                    traps=merged,
                    masks=masks,
                    labels=labels,
                    **kwargs,
                )

        del traps, masks
        return d

    def get_imgs(self, channel, traps, channels=None):
        """
        Returns the image from a correct source, either raw or bgsub


        :channel: str name of channel to get
        :img: ndarray (trap_id, channel, tp, tile_size, tile_size, n_zstacks) of standard channels
        :channels: List of channels
        """

        if channels is None:
            channels = (*self.params.tree,)

        if channel in channels:
            return traps[:, channels.index(channel), 0]
        elif channel in self.img_bgsub:
            return self.img_bgsub[channel]

    def run(self, tree=None, tps: List[int] = None, save=True, **kwargs) -> dict:

        if tree is None:
            tree = self.params.tree

        if tps is None:
            tps = list(range(self.meta["time_settings/ntimepoints"]))

        d = {}
        for tp in tps:
            new = flatten_nest(
                self.extract_tp(tp=tp, tree=tree, **kwargs),
                to="series",
                tp=tp,
            )

            for k in new.keys():
                n = new[k]
                d[k] = pd.concat((d.get(k, None), n), axis=1)

        for k in d.keys():
            indices = ["experiment", "position", "trap", "cell_label"]
            idx = (
                indices[-d[k].index.nlevels :]
                if d[k].index.nlevels > 1
                else [indices[-2]]
            )
            d[k].index.names = idx

            toreturn = d

        if save:
            self.save_to_hdf(toreturn)

        return toreturn

    def extract_pos(
        self, tree=None, tps: List[int] = None, save=True, **kwargs
    ) -> dict:

        if tree is None:
            tree = self.params.tree

        if tps is None:
            tps = list(range(self.meta["time_settings/ntimepoints"]))

        d = {}
        for tp in tps:
            new = flatten_nest(
                self.extract_tp(tp=tp, tree=tree, **kwargs),
                to="series",
                tp=tp,
            )

            for k in new.keys():
                n = new[k]
                d[k] = pd.concat((d.get(k, None), n), axis=1)

        for k in d.keys():
            indices = ["experiment", "position", "trap", "cell_label"]
            idx = (
                indices[-d[k].index.nlevels :]
                if d[k].index.nlevels > 1
                else [indices[-2]]
            )
            d[k].index.names = idx

            toreturn = d

        if save:
            self.save_to_hdf(toreturn)

        return toreturn

    def save_to_hdf(self, group_df, path=None):
        if path is None:
            path = self.local

        self.writer = Writer(path)
        for path, df in group_df.items():
            dset_path = "/extraction/" + path
            self.writer.write(dset_path, df)
        self.writer.id_cache.clear()

    def get_meta(self, flds):
        if not hasattr(flds, "__iter__"):
            flds = [flds]
        meta_short = {k.split("/")[-1]: v for k, v in self.meta.items()}
        return {f: meta_short.get(f, self.default_meta.get(f, None)) for f in flds}


### Helpers
def flatten_nest(nest: dict, to="series", tp: int = None) -> dict:
    """
    Convert a nested extraction dict into a dict of series
    :param nest: dict contained the nested results of extraction
    :param to: str = 'series' Determine output format, either list or  pd.Series
    :param tp: int timepoint used to name the series
    """

    d = {}
    for k0, v0 in nest.items():
        for k1, v1 in v0.items():
            for k2, v2 in v1.items():
                d["/".join((k0, k1, k2))] = (
                    pd.Series(*v2, name=tp) if to == "series" else v2
                )

    return d


def fill_tree(tree):
    if tree is None:
        return None
    tree_depth = depth(tree)
    if depth(tree) < 3:
        d = {None: {None: {None: []}}}
        for _ in range(2 - tree_depth):
            d = d[None]
        d[None] = tree
        tree = d
    return tree


class hollowExtractor(Extractor):
    """Extractor that only cares about receiving image and masks,
    used for testing.
    """

    def __init__(self, parameters):
        self.params = parameters
