"""Extract areas, volumes and fluorescence for the cells in one position."""

import copy
import logging
import typing as t
from pathlib import Path

import bottleneck as bn
import dask.array as da
import h5py
import numpy as np
import pandas as pd
from agora.abc import StepABC
from agora.io.cells import Cells
from agora.io.writers import pdms_mask_path
from aliby.tile.tiler import Tiler
from sooth import half_size, is_obscured
from tiler import shown_in_tile
from extraction.functions.cell_functions import (
    _MODEL_TO_PARAM,
    is_model_available,
)
from extraction.functions.loaders import (
    load_all_functions,
    load_reduction_functions,
)
from extraction.masks import (
    compute_intracellular_masks,
    get_background_masks,
    get_cell_labels,
    get_imgs_background_subtract,
    get_masks,
    get_outlines,
)
from extraction.pdms import compute_pdms_mask
from extraction.tree import (
    ExtractorParameters,
    extraction_tree,
    flatten_nesteddict,
    reduction_method,
)

# global reduction functions
REDUCTION_FUNS = load_reduction_functions()


def reduce_z(trap_image: np.ndarray, fun: t.Callable, axis: int = 0):
    """
    Reduce the trap_image to 2d.

    Parameters
    ----------
    trap_image: array
        Images for all the channels associated with a trap
    fun: function
        Function to execute the reduction
    axis: int (default 0)
        Axis in which we apply the reduction operation.
    """
    if hasattr(fun, "__module__") and fun.__module__[:10] == "bottleneck":
        # bottleneck type
        return getattr(bn.reduce, fun.__name__)(trap_image, axis=axis)
    if isinstance(fun, np.ufunc):
        # optimise the reduction function if possible
        return fun.reduce(trap_image, axis=axis)
    # WARNING: Very slow, only use when no alternatives exist
    return np.apply_along_axis(fun, axis, trap_image)


class Extractor(StepABC):
    """
    Apply a metric to cells identified in the tiles.

    Using the cell masks, the Extractor applies a metric, such as
    area or median, to cells identified in the image tiles.

    Its methods require both tile images and masks.

    Usually the metric is applied to only a tile's masked area, but
    some metrics depend on the whole tile.

    Extraction follows a three-level tree structure. Channels, such
    as GFP, are the root level; the reduction algorithm, such as
    maximum projection, is the second level; the specific metric,
    or operation, to apply to the masks, such as mean, is the third
    or leaf level.
    """

    def __init__(
        self,
        parameters: ExtractorParameters,
        store: str,
        tiler: t.Optional[Tiler] = None,
    ):
        """
        Initialise Extractor.

        Parameters
        ----------
        parameters: core.extractor Parameters
            Parameters that include the channels, reduction and
            extraction functions.
        store: str
            Path to the h5 file containing the cell masks.
            Used by extract_tp.
        tiler: pipeline-core.core.segmentation tiler
            Class that contains or fetches the images used for
            segmentation.
        """
        self.params = parameters
        # what StepABC's parameters property reads
        self._parameters = parameters
        if tiler:
            self.tiler = tiler
            available_channels = set((*tiler.channels, "general"))
            # only extract for channels available
            self.params.tree = {
                k: v
                for k, v in self.params.tree.items()
                if k in available_channels
            }
            self.params.subtract_background = available_channels.intersection(
                self.params.subtract_background
            )
            self.params.intracellular_masks = available_channels.intersection(
                self.params.intracellular_masks
            )
            self.params.background_channels = available_channels.intersection(
                self.params.background_channels
            )
        self.params.multichannel_funs = self.find_usable_multichannel_funs()
        # user-controlled flag
        if not self.params.identify_vacuoles:
            self.params.intracellular_masks = set()
        else:
            # dependency-controlled flags (generic, covers future models)
            for model_name, param_attr in _MODEL_TO_PARAM.items():
                if getattr(
                    self.params, param_attr, None
                ) and not is_model_available(model_name):
                    setattr(self.params, param_attr, set())
        self.store = store
        self.cell_fun_names, self.all_funs = load_all_functions()
        # found once per position from the brightfield images
        self.pdms_mask = self.load_pdms_mask()
        self.sought_pdms_mask = self.pdms_mask is not None
        # labels, per trap, of the cells not measured at this time point
        self.obscured = {}
        # the z reductions of this time point's images, by image and
        # reduction, so that each is made once for a cell, its vacuole
        # and its cytoplasm
        self.reduced_tiles = {}
        # the functions seen to return a dict, with the names of their
        # values
        self.dict_fun_keys = {}

    def find_usable_multichannel_funs(self) -> dict:
        """
        Find the multichannel functions whose channels are all extracted.

        A multichannel function is given the images of the channels in
        the extraction tree and of brightfield. Warn of any that asks for
        another channel, which would otherwise be passed over in silence
        at every time point.

        Returns
        -------
        usable: dict
            The multichannel functions that can be extracted.
        """
        extracted = set(self.params.tree) - {"general"} | {"Brightfield"}
        usable = {}
        for label, spec in self.params.multichannel_funs.items():
            missing = [ch for ch in spec[0] if ch not in extracted]
            if missing:
                logging.getLogger("aliby").warning(
                    f"Extractor: {label} will not be extracted because "
                    f"{', '.join(missing)} is not among the channels "
                    f"extracted: {', '.join(sorted(extracted))}."
                )
            else:
                usable[label] = spec
        return usable

    @classmethod
    def from_tiler(
        cls,
        parameters: ExtractorParameters,
        store: str,
        tiler: Tiler,
    ):
        """
        Initiate from a tiler instance.

        Used by pipeline.py.
        """
        return cls(parameters, store=store, tiler=tiler)

    @classmethod
    def from_img(
        cls,
        parameters: ExtractorParameters,
        store: str,
        img_meta: tuple,
    ):
        """Initiate from images."""
        return cls(parameters, store=store, tiler=Tiler(*img_meta))

    def get_tiles(
        self,
        tp: int,
        channels: t.Optional[t.List[t.Union[str, int]]] = None,
        z: t.Optional[t.List[str]] = None,
        lazy: bool = True,
    ) -> t.Optional[t.Union[np.ndarray, da.Array]]:
        """
        Find tiles for a given time point and given channels.

        Use memory-efficient lazy loading.

        Parameters
        ----------
        tp: int
            Time point of interest.
        channels: list of strings (optional)
            Channels of interest.
        z: list of integers (optional)
            Indices for the z-stacks of interest.
        lazy: bool, optional
            If True, return dask array for memory efficiency.
            Default is True.

        Returns
        -------
        tiles: dask array or numpy array
            Tiles with dimensions (tiles, channels, 1, z, y, x): the
            axis of length one is the Tiler's relic. Return dask array
            if lazy=True, numpy array if lazy=False.
        """
        if channels is None:
            # find channels from tiler
            channel_ids = list(range(len(self.tiler.channels)))
        elif len(channels):
            # a subset of channels was specified
            channel_ids = [self.tiler.get_channel_index(ch) for ch in channels]
        else:
            channel_ids = None
        if z is None:
            # include all z channels
            z = list(range(self.tiler.shape[-3]))
        # use lazy loading by default for memory efficiency
        if lazy:
            tiles = (
                self.tiler.get_tiles_timepoint_lazy(
                    tp, channels=channel_ids, z=z
                )
                if channel_ids
                else None
            )
        else:
            # use numpy arrays
            tiles = (
                self.tiler.get_tiles_timepoint(
                    tp, channels=channel_ids, z=z, lazy=lazy
                )
                if channel_ids
                else None
            )
        return tiles

    def apply_extraction_function(
        self,
        traps: t.List[np.ndarray],
        masks: t.List[np.ndarray],
        fun_name: str,
        cell_labels: t.Dict[int, t.List[int]],
        channels: t.List[str],
    ) -> t.Tuple[t.Union[t.Tuple[float], t.Tuple[t.Tuple[int]]]]:
        """
        Apply an extraction function to all cells or traps.

        Process one time point.

        Parameters
        ----------
        traps: list of arrays
            List of images.
        masks: list of arrays
            List of masks.
        fun_name: str
            Name of the function to apply.
        cell_labels: dict
            A dict with trap_ids as keys and a list of cell labels
            as values.
        channels: list of str
            A list of the channels corresponding to the data in
            traps.

        Returns
        -------
        res_idx: a tuple of tuples
            A two-tuple comprising a tuple of results and a tuple
            of the tile_id and cell labels.
        """
        is_cell_fun = fun_name in self.cell_fun_names
        idx = []
        results = []
        for trap_id, (mask_set, trap, local_cell_labels) in enumerate(
            zip(masks, traps, cell_labels.values())
        ):
            # cell functions need cells, but background functions
            # need only an image because a trap with no cells is
            # entirely background
            has_data = len(mask_set) if is_cell_fun else trap is not None
            if has_data:
                if is_cell_fun:
                    # find property from the tile
                    result = self.all_funs[fun_name](mask_set, trap, channels)
                    # store results for each cell separately, except for
                    # a cell the tile shows only part of
                    hidden = self.obscured.get(trap_id, set())
                    for cell_label, val in zip(local_cell_labels, result):
                        if cell_label in hidden:
                            continue
                        results.append(val)
                        idx.append((trap_id, cell_label))
                else:
                    # background function for cy5-like signals
                    results.append(
                        self.all_funs[fun_name](
                            mask_set,
                            trap,
                            channels,
                            exclude_mask=self.pdms_mask,
                        )
                    )
                    idx.append(trap_id)
        return (tuple(results), tuple(idx))

    def apply_extraction_functions(
        self,
        tiles: t.List[np.array],
        masks: t.List[np.array],
        funs: t.List[str],
        channels: t.List[str],
        cell_labels: t.List[int],
    ) -> dict[str, tuple]:
        """
        Return dict with extraction function names as keys and their results.

        Use data from one time point.

        The functions that need only a cell's pixels are applied
        together, so that the pixels are picked out of the tile once for
        all of them. A function that returns a dict is replaced by one
        entry for each of the dict's values, named for the function and
        the value's key.

        Returns
        -------
        d: dict
            Dict mapping function names to (results, indices) tuples.
        """
        of_pixels = {
            fun: self.all_funs[fun].of_pixels
            for fun in funs
            if hasattr(self.all_funs.get(fun), "of_pixels")
        }
        from_pixels = (
            self.apply_pixel_functions(tiles, masks, of_pixels, cell_labels)
            if of_pixels
            else {}
        )
        d = {
            fun: (
                from_pixels[fun]
                if fun in from_pixels
                else self.apply_extraction_function(
                    traps=tiles,
                    masks=masks,
                    fun_name=fun,
                    channels=channels,
                    cell_labels=cell_labels,
                )
            )
            for fun in funs
        }
        # check for functions returning a dict rather than a value
        # a time point whose every cell is obscured has no results
        dict_fns = [
            fun for fun in d if d[fun][0] and isinstance(d[fun][0][0], dict)
        ]
        for fn in list(d):
            if fn in self.dict_fun_keys and not d[fn][0]:
                # no cells: name its values as when there are cells, not
                # leave one empty entry under the function's own name
                for subkey in self.dict_fun_keys[fn]:
                    d[subkey] = ((), ())
                del d[fn]
        for fn in dict_fns:
            # add to d for each key in returned dict
            mini_d = {}
            for res in d[fn][0]:
                for key in res:
                    subkey = f"{fn}_{key}"
                    if subkey in mini_d:
                        mini_d[subkey].append(res[key])
                    else:
                        mini_d[subkey] = [res[key]]
            idx = d[fn][1]
            for subkey, value in mini_d.items():
                d[subkey] = (tuple(value), idx)
            del d[fn]
            self.dict_fun_keys[fn] = list(mini_d)
        return d

    def apply_pixel_functions(
        self,
        tiles: t.List[np.ndarray],
        masks: t.List[np.ndarray],
        of_pixels: dict[str, t.Callable],
        cell_labels: t.Dict[int, t.List[int]],
    ) -> dict[str, tuple]:
        """
        Apply functions of a cell's pixels to all cells at one time point.

        Pick each cell's pixels out of its tile once and give them to
        every function: picking them out for each function took as long
        as the functions.

        Parameters
        ----------
        tiles: list of arrays
            An image for each tile.
        masks: list of arrays
            The cells' masks for each tile.
        of_pixels: dict
            Functions of a cell's pixels, by name.
        cell_labels: dict
            A dict with trap_ids as keys and a list of cell labels
            as values.

        Returns
        -------
        d: dict
            Dict mapping function names to (results, indices) tuples, as
            apply_extraction_function gives for one function.
        """
        results = {fun: [] for fun in of_pixels}
        idx = []
        for trap_id, (mask_set, tile, labels) in enumerate(
            zip(masks, tiles, cell_labels.values())
        ):
            if not len(mask_set):
                continue
            # a cell the tile shows only part of is not measured
            hidden = self.obscured.get(trap_id, set())
            for mask, cell_label in zip(mask_set, labels):
                if cell_label in hidden:
                    continue
                pixels = tile[mask]
                for fun, fun_of_pixels in of_pixels.items():
                    results[fun].append(fun_of_pixels(pixels))
                idx.append((trap_id, cell_label))
        idx = tuple(idx)
        return {fun: (tuple(results[fun]), idx) for fun in of_pixels}

    def reduce_extract(
        self,
        tiles: np.ndarray,
        masks: t.List[np.ndarray],
        reduction_funs: t.Dict[reduction_method, t.Collection[str]],
        channels: t.List[str],
        cell_labels: t.List[int],
        reduced_as: str | None = None,
    ) -> dict[str, dict[reduction_method, dict[str, tuple]]]:
        """
        Reduce to a 2D image and then extract.

        Parameters
        ----------
        tiles: array
            An array of image data arranged as (tiles, Z, Y, X).
        masks: list of arrays
            An array of masks for each trap: one per cell at the
            trap.
        reduction_funs: dict
            An upper branch of the extraction tree: a dict for which
            keys are reduction functions and values are either a list
            or a set of strings giving the extraction functions to apply.
            For example: {'max': {'max5px_median', 'mean', 'median'}}
        channels: list of str
            A list comprising the channel corresponding to the data
            in tiles.
        cell_labels: list of int
            Cell labels per trap.
        reduced_as: str, optional
            A name for the images, such as "GFP_bgsub". Their reductions
            are kept under it until the next time point, and used again
            if the same name is given, as it is for a cell's vacuole and
            cytoplasm. If None, nothing is kept.

        Returns
        ------
        d: dict
            Dict of results with reductions and metrics nested.
        """
        # create dict with keys naming the reduction in the
        # z-direction and the reduced data as values
        reduced_tiles = {}
        if tiles is not None:
            for reduction in reduction_funs:
                red_fun = REDUCTION_FUNS[reduction]
                if red_fun is None:
                    continue
                kept_as = (reduced_as, reduction)
                if reduced_as is not None and kept_as in self.reduced_tiles:
                    reduced_tiles[reduction] = self.reduced_tiles[kept_as]
                    continue
                # all the tiles at once, over their z axis
                reduced_tiles[reduction] = reduce_z(
                    np.asarray(tiles), red_fun, axis=1
                )
                if reduced_as is not None:
                    self.reduced_tiles[kept_as] = reduced_tiles[reduction]
        # calculate cell and tile properties
        d = {}
        for reduction, funs in reduction_funs.items():
            d[reduction] = self.apply_extraction_functions(
                tiles=reduced_tiles.get(reduction, [None for _ in masks]),
                masks=masks,
                funs=funs,
                channels=channels,
                cell_labels=cell_labels,
            )
        return d

    def load_pdms_mask(self):
        """
        Read any PDMS mask already found for this position.

        Reuse the mask so that a resumed run gives the same background
        estimates as the run it continues.

        Returns
        -------
        mask: array or None
            A boolean array, shape (Y, X), or None if no mask is
            stored or if masking the PDMS is switched off.
        """
        if not self.params.mask_pdms or not Path(self.store).exists():
            return None
        try:
            with h5py.File(self.store, "r") as f:
                if pdms_mask_path in f:
                    return f[pdms_mask_path][()].astype(bool)
        except OSError:
            # the h5 file is not yet readable
            pass
        return None

    def set_pdms_mask(self, brightfield, masks):
        """
        Find the PDMS trap unless it has been sought already.

        Look only once per position: the trap does not move and a
        failure to find it will recur at every time point.

        Parameters
        ----------
        brightfield: array
            Brightfield images for all tiles, shape
            (no_tiles, Z, Y, X).
        masks: list of arrays
            Segmentation masks per tile, each (no_cells, Y, X).
        """
        if self.params.mask_pdms and not self.sought_pdms_mask:
            self.sought_pdms_mask = True
            if brightfield is None or not len(brightfield):
                return
            # project over z, as the vacuole CNN also does, because
            # the trap is in focus at no single z
            self.pdms_mask = compute_pdms_mask(brightfield.mean(axis=1), masks)

    def _extract_channel(
        self,
        ch,
        reduction_funs,
        img,
        img_bgsub,
        masks,
        cell_labels,
        background_chs,
    ):
        """
        Extract raw and background-subtracted metrics for one channel.

        Returns
        -------
        d: dict
            Results keyed by channel name and channel + '_bgsub'.
        """
        d = {
            ch: self.reduce_extract(
                # use null for "general" - no fluorescence image
                tiles=img.get(ch, None),
                masks=masks,
                reduction_funs=reduction_funs,
                cell_labels=cell_labels,
                channels=[ch],
                reduced_as=ch,
            )
        }
        if (
            ch != "general"
            and ch not in background_chs
            and ch in self.params.subtract_background
        ):
            d[ch + "_bgsub"] = self.reduce_extract(
                tiles=img_bgsub[ch + "_bgsub"],
                masks=masks,
                reduction_funs=reduction_funs,
                cell_labels=cell_labels,
                channels=[ch],
                reduced_as=ch + "_bgsub",
            )
        return d

    def _extract_intracellular(
        self,
        ch,
        reduction_funs,
        img,
        img_bgsub,
        masks,
        cell_labels,
        vac_masks,
        cyt_masks,
    ):
        """
        Extract vacuole and cytoplasm metrics for one channel.

        Returns
        -------
        d: dict
            Results keyed by channel + '_vacuole' / '_cytoplasm' suffixes.
        """
        d = {}
        for suffix, sub_masks in [
            ("_vacuole", vac_masks),
            ("_cytoplasm", cyt_masks),
        ]:
            d[ch + suffix] = self.reduce_extract(
                tiles=img.get(ch, None),
                masks=sub_masks,
                reduction_funs=reduction_funs,
                cell_labels=cell_labels,
                channels=[ch],
                reduced_as=ch,
            )
            if ch in self.params.subtract_background:
                d[ch + suffix + "_bgsub"] = self.reduce_extract(
                    tiles=img_bgsub[ch + "_bgsub"],
                    masks=sub_masks,
                    reduction_funs=reduction_funs,
                    cell_labels=cell_labels,
                    channels=[ch],
                    reduced_as=ch + "_bgsub",
                )
        return d

    def extract_single_channel_functions(
        self,
        tree_dict,
        cell_labels,
        img,
        img_bgsub,
        masks,
        vac_masks=None,
        cyt_masks=None,
    ):
        """
        Extract all metrics requiring a single fluorescence channel.

        Returns
        -------
        d: dict
            Nested extraction results.
        """
        d = {}
        # the Extractor's own background channels: tree_dict holds none
        background_chs = self.params.background_channels
        for ch, reduction_funs in tree_dict["tree"].items():
            d.update(
                self._extract_channel(
                    ch,
                    reduction_funs,
                    img,
                    img_bgsub,
                    masks,
                    cell_labels,
                    background_chs,
                )
            )
            if ch in self.params.intracellular_masks and vac_masks is not None:
                d.update(
                    self._extract_intracellular(
                        ch,
                        reduction_funs,
                        img,
                        img_bgsub,
                        masks,
                        cell_labels,
                        vac_masks,
                        cyt_masks,
                    )
                )
        return d

    def extract_multichannel_functions(
        self, cell_labels, img, img_bgsub, masks
    ):
        """
        Extract all metrics requiring multiple fluorescence channels.

        Include 'Brightfield'.

        Multichannel functions do not use tree_dict.
        Instead in extraction, parameters include,

            {"multichannel"} : [channels, reduction function,
                                multichannel function name]

        For example, for the ratio_1_over_2 multichannel function

            {"multichannel": [["CFP", "YFP"], "max", "ratio_1_over_2"]}

        Pass these to PipelineParameters.default as

            extraction={"multichannel_funs": {
                "multichannel": [["CFP", "YFP"], "max", "ratio_1_over_2"]
            }}

        which will create Signals called

            '/extraction/multichannel/max/ratio_1_over_2'
            '/extraction/multichannel/max/ratio_1_over_2_bgsub'

        See examples/run_custom_function.py.
        """
        available_channels = set(list(img.keys()) + list(img_bgsub.keys()))
        d = {}
        # multichannel_label below is the "multichannel" string
        for multichannel_label, (
            channels,
            reduction,
            multichannel_function,
        ) in self.params.multichannel_funs.items():
            common_channels = set(channels).intersection(available_channels)
            # all required channels should be available
            if len(common_channels) == len(channels):
                for images, suffix in zip([img, img_bgsub], ["", "_bgsub"]):
                    if suffix and "Brightfield" in channels:
                        # brightfield has no background to subtract, and
                        # the function needs an image for every channel
                        continue
                    # channels
                    channel_images = [
                        images.get(ch + suffix) for ch in channels
                    ]
                    if any(image is None for image in channel_images):
                        # a channel whose background is not subtracted
                        continue
                    channels_stack = np.stack(channel_images, axis=-1)
                    # reduce in Z
                    tiles = REDUCTION_FUNS[reduction](channels_stack, axis=1)
                    # set up dict
                    if multichannel_label not in d:
                        d[multichannel_label] = {}
                    if reduction not in d[multichannel_label]:
                        d[multichannel_label][reduction] = {}
                    # apply multichannel function
                    d[multichannel_label][reduction][
                        multichannel_function + suffix
                    ] = self.apply_extraction_function(
                        tiles,
                        masks,
                        multichannel_function,
                        cell_labels,
                        channels,
                    )
        return d

    def make_tree_dict(self, tree: extraction_tree):
        """Put extraction tree into a dict."""
        if tree is None:
            # use default
            tree = copy.deepcopy(self.params.tree)
        tree_dict = {
            # the whole extraction tree
            "tree": tree,
            # the extraction tree for fluorescence channels
            "channels_tree": {
                ch: v for ch, v in tree.items() if ch != "general"
            },
        }
        # tuple of the fluorescence channels
        tree_dict["channels"] = (*tree_dict["channels_tree"],)
        return tree_dict

    def load_tp_data(
        self,
        tp: int,
        tile_size: int,
        tree_dict: dict,
        masks: t.Optional[t.List[np.ndarray]] = None,
        cell_labels: t.Optional[dict] = None,
    ) -> tuple[dict, list, list, dict, np.ndarray]:
        """
        Load all data needed for extraction at one time point.

        Parameters
        ----------
        tp: int
            Time point being analysed.
        tile_size: int
            Size of each tile.
        tree_dict: dict
            Processed extraction tree from make_tree_dict.
        masks: list of arrays (optional)
            Pre-computed masks.
        cell_labels: dict (optional)
            Pre-computed cell labels.

        Returns
        -------
        cell_labels: dict
            Cell labels per trap.
        masks: list
            List of mask arrays, one per trap.
        outlines: list or None
            List of labelled-outline arrays, one (Y, X) image
            per trap with integer cell labels; None unless vacuoles are
            to be found, which is all the outlines are for.
        img: dict
            Fluorescence images keyed by channel name.
        img_bgsub: dict
            Background-subtracted images keyed by channel name.
        """
        # create a fresh Cells object each time because the h5 file
        # grows between time points and Cells has cached_property
        cells = Cells(self.store)
        cell_labels = get_cell_labels(tp, cell_labels, cells)
        masks = get_masks(tp, masks, cells)
        outlines = None
        if self.params.intracellular_masks:
            outlines_dict = get_outlines(tp, cell_labels, cells)
            outlines = [outlines_dict[tid] for tid in range(cells.ntraps)]
        # fluorescence data for all traps at the time point
        # stored as an array arranged as (traps, channels, 1, Z, Y, X)
        tiles = self.get_tiles(tp, channels=tree_dict["channels"], lazy=False)
        # brightfield images as (traps, Z, Y, X), from the Tiler's
        # reference channel whatever it is called
        brightfield = self.tiler.get_tiles_timepoint(
            tp,
            channels=[self.tiler.ref_channel_index],
            z=list(range(self.tiler.shape[-3])),
            lazy=False,
        )[:, 0, 0, ...]
        # find the PDMS trap before any background is estimated
        self.set_pdms_mask(brightfield, masks)
        if self.params.subtract_background:
            bgs = get_background_masks(masks, tile_size, self.pdms_mask)
        else:
            bgs = np.array([])
        img, img_bgsub = get_imgs_background_subtract(
            self.params, tree_dict, tiles, bgs
        )
        img["Brightfield"] = brightfield
        return cell_labels, masks, outlines, img, img_bgsub

    def extract_tp(
        self,
        tp: int,
        tile_size: int,
        tree: t.Optional[extraction_tree] = None,
        masks: t.Optional[t.List[np.ndarray]] = None,
        cell_labels: t.Optional[t.List[int]] = None,
    ) -> t.Dict[str, t.Dict[str, t.Dict[str, tuple]]]:
        """
        Extract for an individual time point.

        Parameters
        ----------
        tp: int
            Time point being analysed.
        tree: dict
            Nested dictionary indicating channels, reduction
            functions and metrics to be used.
            For example:
            {'general': {'null': ['area', 'volume', 'eccentricity']}}
        tile_size: int
            Size of the tile to be extracted.
        masks: dict of arrays
            A dictionary with trap_ids as keys, in order, and as values
            the masks of each trap's cells, with dimensions
            (ncells, Y, X) and with one mask per cell.
        cell_labels: dict
            A dictionary with trap_ids as keys and cell_labels as
            values.

        Returns
        -------
        res: dict
            Dictionary of the results with three levels of
            dictionaries.
            The first level has channels as keys.
            The second level has reduction metrics as keys.
            The third level has cell or background metrics as keys
            and a two-tuple as values.
            The first tuple is the result of applying the metrics
            to a particular cell or trap; the second tuple is either
            (trap_id, cell_label) for a metric applied to a cell or
            a trap_id for a metric applied to a trap.

            An example is d["GFP"]["max"]["mean"][0], which gives
            a tuple of the calculated mean GFP fluorescence for all
            cells.
        """
        tree_dict = self.make_tree_dict(tree)
        # the reductions kept are of the last time point's images
        self.reduced_tiles = {}
        # load all data for this time point
        cell_labels, masks, outlines, img, img_bgsub = self.load_tp_data(
            tp, tile_size, tree_dict, masks, cell_labels
        )
        # every cell has already shaped the background; only now are the
        # cut-off ones set aside, so they are measured by nothing else
        self.obscured = self.find_obscured(tp, masks, cell_labels)
        # compute intracellular sub-masks if requested
        vac_masks, cyt_masks = None, None
        if self.params.intracellular_masks:
            vac_masks, cyt_masks = compute_intracellular_masks(
                outlines, masks, img
            )
        # extract for functions requiring one channel
        res_one = self.extract_single_channel_functions(
            tree_dict,
            cell_labels,
            img,
            img_bgsub,
            masks,
            vac_masks=vac_masks,
            cyt_masks=cyt_masks,
        )
        # extract for functions requiring multiple channels
        res_multiple = self.extract_multichannel_functions(
            cell_labels, img, img_bgsub, masks
        )
        return {**res_one, **res_multiple}

    def find_obscured(
        self, tp: int, masks: list[np.ndarray], cell_labels: dict
    ) -> dict[int, set[int]]:
        """
        Find the cells that their tiles show only part of.

        A cell is obscured if it reaches the tile's border or the pixels
        a tile hanging off the image is padded with; sooth's
        ``is_obscured`` is the rule and tiler's ``shown_in_tile`` says
        which pixels were imaged.

        Parameters
        ----------
        tp: int
            Time point, which places each tile.
        masks: list of arrays
            Cell masks per trap, each (ncells, Y, X).
        cell_labels: dict
            Cell labels per trap, in the order of the masks.

        Returns
        -------
        obscured: dict
            Trap ids as keys and sets of obscured cell labels as values;
            empty if exclude_obscured is False.
        """
        if not self.params.exclude_obscured:
            return {}
        shape_yx = self.tiler.shape[-2:]
        obscured = {}
        for trap_id, (mask_set, labels) in enumerate(
            zip(masks, cell_labels.values())
        ):
            if not len(mask_set):
                continue
            shown = shown_in_tile(
                self.tiler.tile_locs.tiles[trap_id].as_range(tp),
                shape_yx,
                self.tiler.tile_size,
            )
            hidden = {
                label
                for mask, label in zip(mask_set, labels)
                if is_obscured(mask, shown)
            }
            if hidden:
                obscured[trap_id] = hidden
        return obscured

    def _run_tp(self, tps: t.List[int] = None, tree=None) -> dict:
        """
        Run extraction for one position and specified time points.

        One time point is run at a time in pipeline.
        Save the results to a h5 file.

        Parameters
        ----------
        tps: list of int (optional)
            Time points to include.
        tree: dict (optional)
            Nested dictionary indicating channels, reduction
            functions and metrics to be used. For example:
            {'general': {'null': ['area', 'volume',
            'eccentricity']}}
            Without centroid_x and centroid_y among them, the cells'
            places in the image are not found.

        Returns
        -------
        d: dict
            A dict of the extracted data for one position with a
            concatenated string of channel, reduction metric, and
            cell metric as keys and pd.DataFrame of the extracted
            data for all time points as values.
        """
        if tree is None:
            tree = self.params.tree
        if isinstance(tps, int):
            tps = [tps]
        # store results in dict
        extract_dict = {}
        for tp in tps:
            # extract for each time point and convert to pd.Series
            res = self.extract_tp(
                tp=tp, tile_size=self.tiler.tile_size, tree=tree
            )
            new = flatten_nesteddict(res, to="series", tp=tp)
            # concatenate with earlier time points
            for key in new:
                extract_dict[key] = pd.concat(
                    (extract_dict.get(key, None), new[key]), axis=1
                )
        # add indices to pd.Series containing the extracted data
        for key in extract_dict:
            indices = [
                "experiment",
                "position",
                "trap",
                "cell_label",
            ]
            idx = (
                indices[-extract_dict[key].index.nlevels :]
                if extract_dict[key].index.nlevels > 1
                else [indices[-2]]
            )
            extract_dict[key].index.names = idx
        # add cells' spatial locations within the image
        self.add_spatial_locations_of_cells(extract_dict)
        return extract_dict

    def add_spatial_locations_of_cells(self, extract_dict):
        """
        Add spatial location within image of each cell.

        Add nothing if the centroids were not extracted, as with a tree
        that does not ask for them.
        """
        x_df = extract_dict.get("general/null/centroid_x")
        y_df = extract_dict.get("general/null/centroid_y")
        if x_df is None or y_df is None:
            return
        extract_dict["general/null/image_x"] = x_df.copy()
        extract_dict["general/null/image_y"] = y_df.copy()
        # a cell's image coordinate is its tile's origin plus its centroid,
        # and the origin is sooth's -- centre - size // 2 on each axis --
        # because that is where the tile was cut. The tile's *geometric*
        # centre, (size - 1) / 2, is half a pixel away from it for an even
        # size, which placed every cell of an even tile half a pixel down
        # and to the right of its own tile. x before y, because the columns
        # below are (x, y)
        offset_y, offset_x = half_size(self.tiler.tile_size)
        half_extent = np.asarray([offset_x, offset_y])
        traps = np.array(x_df.index.get_level_values("trap"))
        # how many traps there are, not whether any is named something
        # other than zero: a position holding one trap, named 0, had its
        # cells left at their tile-local centroids
        if len(traps):
            for tp in x_df.columns:
                tile_locs = self.tiler.tile_locs.centres_at_time(tp)
                centroid_coords = np.column_stack(
                    (x_df[tp].values, y_df[tp].values)
                )
                coords_in_image = (
                    centroid_coords + tile_locs[traps][:, ::-1] - half_extent
                )
                extract_dict["general/null/image_x"][tp] = coords_in_image[
                    :, 0
                ]
                extract_dict["general/null/image_y"][tp] = coords_in_image[
                    :, 1
                ]
        if self.tiler.spatial_location is not None:
            extract_dict["general/null/absolute_x"] = (
                extract_dict["general/null/image_x"].copy()
                + self.tiler.spatial_location[0]
            )
            extract_dict["general/null/absolute_y"] = (
                extract_dict["general/null/image_y"].copy()
                + self.tiler.spatial_location[1]
            )
