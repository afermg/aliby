"""Extract areas, volumes and fluorescence for the cells in one position."""

import copy
import typing as t
from pathlib import Path

import bottleneck as bn
import dask.array as da
import h5py
import numpy as np
import pandas as pd
from agora.abc import ParametersABC, StepABC
from agora.io.cells import Cells
from aliby.global_settings import global_settings
from aliby.tile.tiler import Tiler, find_channel_name
from extraction.core.functions.cell_functions import (
    _MODEL_TO_PARAM,
    identify_vacuole,
    is_model_available,
)
from extraction.core.functions.loaders import (
    load_all_functions,
    load_reduction_functions,
)

# define types
reduction_method = t.Union[t.Callable, str]
extraction_tree = t.Dict[
    str, t.Dict[reduction_method, t.Dict[str, t.Collection]]
]
extraction_result = t.Dict[
    str, t.Dict[reduction_method, t.Dict[str, t.Dict[str, pd.Series]]]
]

# global reduction functions
REDUCTION_FUNS = load_reduction_functions()


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
        # find matching channels in metadata
        found_channel = find_channel_name(meta.get("channels", []), av_channel)
        if found_channel is not None:
            extant_fluorescence_ch.append(found_channel)
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
    return tree_dict


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
        """
        self.tree = tree
        self.subtract_background = subtract_background
        self.multichannel_funs = multichannel_funs
        self.intracellular_masks = intracellular_masks
        self.identify_vacuoles = identify_vacuoles
        self.background_channels = background_channels

    @classmethod
    def default(cls):
        """Override ParametersABC default class method."""
        return cls({})

    @classmethod
    def from_meta(cls, meta):
        """Instantiate from the meta data; used by Pipeline."""
        return cls(**build_extraction_tree_from_meta(meta))


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

    # get pixel_size; z_size; z_spacing
    default_meta = global_settings.imaging_specifications

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
            Tiles with dimensions (tiles, channels, z, y, x).
            Return dask array if lazy=True, numpy array if
            lazy=False.
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
            # ignore empty traps
            if len(mask_set):
                # find property from the tile
                result = self.all_funs[fun_name](mask_set, trap, channels)
                if is_cell_fun:
                    # store results for each cell separately
                    for cell_label, val in zip(local_cell_labels, result):
                        results.append(val)
                        idx.append((trap_id, cell_label))
                else:
                    # background function for cy5-like signals
                    results.append(result)
                    idx.append(trap_id)
        return (tuple(results), tuple(idx))

    def apply_extraction_functions(
        self,
        tiles: t.List[np.array],
        masks: t.List[np.array],
        funs: t.List[str],
        channels: t.List[str],
        cell_labels: t.List[int],
    ) -> tuple[dict[str, pd.Series], dict[str, list[str]]]:
        """
        Return dict with extraction function names as keys and their results.

        Use data from one time point.

        Returns
        -------
        d: dict
            Dict mapping function names to (results, indices) tuples.
        replacements: dict
            Dict mapping original function names to lists of expanded
            sub-key names for functions that return dicts.
        """
        d = {
            fun: self.apply_extraction_function(
                traps=tiles,
                masks=masks,
                fun_name=fun,
                channels=channels,
                cell_labels=cell_labels,
            )
            for fun in funs
        }
        # check for functions returning a dict rather than a value
        dict_fns = [fun for fun in d if isinstance(d[fun][0][0], dict)]
        replacements = {}
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
            replacements[fn] = list(mini_d)
        return d, replacements

    def reduce_extract(
        self,
        tiles: np.ndarray,
        masks: t.List[np.ndarray],
        reduction_funs: t.Dict[reduction_method, t.Collection[str]],
        channels: t.List[str],
        cell_labels: t.List[int],
    ) -> tuple[
        dict[str, dict[reduction_method, dict[str, pd.Series]]],
        dict[str, list[str]],
    ]:
        """
        Reduce to a 2D image and then extract.

        Parameters
        ----------
        tiles: array
            An array of image data arranged as (tiles, X, Y, Z).
        masks: list of arrays
            An array of masks for each trap: one per cell at the
            trap.
        reduction_funs: dict
            An upper branch of the extraction tree: a dict for which
            keys are reduction functions and values are either a list
            or a set of strings giving the extraction functions to apply.
            For example: {'np_max': {'max5px', 'mean', 'median'}}
        channels: list of str
            A list comprising the channel corresponding to the data
            in tiles.
        cell_labels: list of int
            Cell labels per trap.

        Returns
        ------
        d: dict
            Dict of results with reductions and metrics nested.
        replacements: dict
            Accumulated replacements for multi-value functions.
        """
        # create dict with keys naming the reduction in the
        # z-direction and the reduced data as values
        reduced_tiles = {}
        if tiles is not None:
            for reduction in reduction_funs:
                red_fun = REDUCTION_FUNS[reduction]
                if red_fun is not None:
                    reduced_tiles[reduction] = [
                        reduce_z(tile_data, red_fun) for tile_data in tiles
                    ]
        # calculate cell and tile properties
        d = {}
        all_replacements = {}
        for reduction, funs in reduction_funs.items():
            results, replacements = self.apply_extraction_functions(
                tiles=reduced_tiles.get(reduction, [None for _ in masks]),
                masks=masks,
                funs=funs,
                channels=channels,
                cell_labels=cell_labels,
            )
            d[reduction] = results
            all_replacements.update(replacements)
        return d, all_replacements

    def get_masks(self, tp, masks, cells):
        """Get masks as a list with an array of masks per trap."""
        if masks is None:
            raw_masks = cells.at_time(tp, kind="mask")
            masks = {trap_id: [] for trap_id in range(cells.ntraps)}
            for trap_id, trap_cells in raw_masks.items():
                if len(trap_cells):
                    masks[trap_id] = np.stack(np.array(trap_cells)).astype(
                        bool
                    )
        # one array of shape (n_cells, tile_size, tile_size) per trap
        masks = [np.array(v) for v in masks.values()]
        return masks

    def get_cell_labels(self, tp, cell_labels, cells):
        """Get cell labels per trap as a dict with trap_ids as keys."""
        if cell_labels is None:
            raw_cell_labels = cells.labels_at_time(tp)
            cell_labels = {
                trap_id: raw_cell_labels.get(trap_id, [])
                for trap_id in range(cells.ntraps)
            }
        return cell_labels

    def get_outlines(self, tp, cell_labels, cells):
        """Get cell outlines with individual labels as a dict with trap_ids as keys."""
        cell_mask_dict = {
            trap_id: cells.at_time(tp, kind="edgemask").get(trap_id, [])
            for trap_id in range(cells.ntraps)
        }
        cell_labels = self.get_cell_labels(tp, cell_labels, cells)
        outlines_dict = {}
        for trap_id in range(cells.ntraps):
            edgemasks = cell_mask_dict[trap_id]
            labels = cell_labels[trap_id]
            if edgemasks and labels:
                outlines_dict[trap_id] = np.stack(
                    [
                        cell_mask * label
                        for cell_mask, label in zip(edgemasks, labels)
                    ]
                ).max(axis=0)
            else:
                outlines_dict[trap_id] = np.array([])
        return outlines_dict

    def get_background_masks(self, masks, tile_size):
        """
        Generate boolean background masks for pre-reduction subtraction.

        Combine cell masks per trap and take the logical inverse to
        identify background pixels. These masks are used in
        ``get_imgs_background_subtract`` to compute and subtract the
        median background independently for each z-slice, before the
        z-reduction step.

        Parameters
        ----------
        masks: list of arrays
            Segmentation masks per trap, each (N_traps, Y, X).
        tile_size: int
            Size of each tile in pixels.

        Returns
        -------
        bgs: array
            Boolean background masks with shape (N_traps, Y, X), or
            an empty array if background subtraction is disabled.
        """
        if self.params.subtract_background:
            combined = np.array(
                [
                    (
                        np.any(m, axis=0)
                        if len(m)
                        else np.zeros((tile_size, tile_size), dtype=bool)
                    )
                    for m in masks
                ]
            )
            bgs = ~combined
        else:
            bgs = np.array([])
        return bgs

    def compute_intracellular_masks(self, outlines, masks, img):
        """
        Compute vacuole and cytoplasm sub-masks from brightfield.

        Parameters
        ----------
        outlines: list of arrays
            Labelled-outline images per trap, each (Y, X) with
            integer cell labels.
        masks: list of arrays
            Cell masks per trap, each (ncells, Y, X).
        img: dict
            Image dict containing "Brightfield" key with shape
            (ntraps, Y, X).

        Returns
        -------
        vac_masks: list of arrays
            Vacuole masks per trap with same structure as masks.
        cyt_masks: list of arrays
            Cytoplasm masks per trap with same structure as masks.
        """
        bf = img["Brightfield"]
        vac_masks = []
        cyt_masks = []
        for trap_outline, trap_masks, bf_trap in zip(outlines, masks, bf):
            if len(trap_masks):
                # single binary vacuole image for the whole trap
                trap_vac = identify_vacuole(trap_outline, bf_trap)
                # split into per-cell masks
                vac_list, cyt_list = [], []
                for cell_mask in trap_masks:
                    cell_vac = cell_mask & trap_vac
                    if cell_vac.any():
                        vac_list.append(cell_vac)
                        cyt_list.append(cell_mask & ~trap_vac)
                    else:
                        # no vacuole found; cytoplasm is also undefined
                        vac_list.append(np.zeros_like(cell_mask))
                        cyt_list.append(np.zeros_like(cell_mask))
                vac_masks.append(np.stack(vac_list))
                cyt_masks.append(np.stack(cyt_list))
            else:
                vac_masks.append(np.array([]))
                cyt_masks.append(np.array([]))
        return vac_masks, cyt_masks

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
        replacements: dict
            Accumulated replacements for multi-value functions.
        """
        d, replacements = self.reduce_extract(
            # use null for "general" - no fluorescence image
            tiles=img.get(ch, None),
            masks=masks,
            reduction_funs=reduction_funs,
            cell_labels=cell_labels,
            channels=[ch],
        )
        d = {ch: d}
        if ch != "general" and ch not in background_chs:
            bgsub_results, bgsub_replacements = self.reduce_extract(
                tiles=img_bgsub[ch + "_bgsub"],
                masks=masks,
                reduction_funs=reduction_funs,
                cell_labels=cell_labels,
                channels=[ch],
            )
            d[ch + "_bgsub"] = bgsub_results
            replacements.update(bgsub_replacements)
        return d, replacements

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
        replacements: dict
            Accumulated replacements for multi-value functions.
        """
        d = {}
        replacements = {}
        for suffix, sub_masks in [
            ("_vacuole", vac_masks),
            ("_cytoplasm", cyt_masks),
        ]:
            results, rep = self.reduce_extract(
                tiles=img.get(ch, None),
                masks=sub_masks,
                reduction_funs=reduction_funs,
                cell_labels=cell_labels,
                channels=[ch],
            )
            d[ch + suffix] = results
            replacements.update(rep)
            if ch in self.params.subtract_background:
                bgsub_results, bgsub_rep = self.reduce_extract(
                    tiles=img_bgsub[ch + "_bgsub"],
                    masks=sub_masks,
                    reduction_funs=reduction_funs,
                    cell_labels=cell_labels,
                    channels=[ch],
                )
                d[ch + suffix + "_bgsub"] = bgsub_results
                replacements.update(bgsub_rep)
        return d, replacements

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
        replacements: dict
            Accumulated replacements for multi-value functions.
        """
        d = {}
        all_replacements = {}
        background_chs = tree_dict.get("background_channels", set())
        for ch, reduction_funs in tree_dict["tree"].items():
            ch_d, ch_rep = self._extract_channel(
                ch, reduction_funs, img, img_bgsub,
                masks, cell_labels, background_chs,
            )
            d.update(ch_d)
            all_replacements.update(ch_rep)
            if ch in self.params.intracellular_masks and vac_masks is not None:
                intra_d, intra_rep = self._extract_intracellular(
                    ch, reduction_funs, img, img_bgsub,
                    masks, cell_labels, vac_masks, cyt_masks,
                )
                d.update(intra_d)
                all_replacements.update(intra_rep)
        return d, all_replacements

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

        For example, for the ratio multichannel function

            {"multichannel": [["CFP", "YFP"], "max", "ratio"]}

        If params is an instance of PipelineParameters, use

            params.to_dict()["extraction"]["multichannel_funs"].update(
            {"multichannel": [["CFP", YFP"], "max", "ratio"]}
            )

        which will create a Signal called

            '/extraction/multichannel/max/ratio'
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
                    # channels
                    channels_stack = np.stack(
                        [
                            images[ch + suffix]
                            for ch in channels
                            if ch + suffix != "Brightfield_bgsub"
                        ],
                        axis=-1,
                    )
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
        outlines: list
            List of labelled-outline arrays, one (Y, X) image
            per trap with integer cell labels.
        img: dict
            Fluorescence images keyed by channel name.
        img_bgsub: dict
            Background-subtracted images keyed by channel name.
        """
        # create a fresh Cells object each time because the h5 file
        # grows between time points and Cells has cached_property
        cells = Cells(self.store)
        cell_labels = self.get_cell_labels(tp, cell_labels, cells)
        masks = self.get_masks(tp, masks, cells)
        outlines_dict = self.get_outlines(tp, cell_labels, cells)
        outlines = [outlines_dict[tid] for tid in range(cells.ntraps)]
        # fluorescence data for all traps at the time point
        # stored as an array arranged as (traps, channels, 1, Z, X, Y)
        tiles = self.get_tiles(tp, channels=tree_dict["channels"], lazy=False)
        bgs = self.get_background_masks(masks, tile_size)
        img, img_bgsub = self.get_imgs_background_subtract(
            tree_dict, tiles, bgs
        )
        # add brightfield images as (traps, channels, 1, Z, X, Y)
        img["Brightfield"] = self.get_tiles(
            tp, channels=["Brightfield"], lazy=False
        )[:, 0, 0, ...]
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
        masks: list of arrays
            A list of masks per trap with each mask having dimensions
            (ncells, tile_size, tile_size) and with one mask per cell.
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

            An example is d["GFP"]["np_max"]["mean"][0], which gives
            a tuple of the calculated mean GFP fluorescence for all
            cells.
        """
        tree_dict = self.make_tree_dict(tree)
        # load all data for this time point
        cell_labels, masks, outlines, img, img_bgsub = self.load_tp_data(
            tp, tile_size, tree_dict, masks, cell_labels
        )
        # compute intracellular sub-masks if requested
        vac_masks, cyt_masks = None, None
        if self.params.intracellular_masks:
            vac_masks, cyt_masks = self.compute_intracellular_masks(
                outlines, masks, img
            )
        # extract for functions requiring one channel
        res_one, replacements = self.extract_single_channel_functions(
            tree_dict,
            cell_labels,
            img,
            img_bgsub,
            masks,
            vac_masks=vac_masks,
            cyt_masks=cyt_masks,
        )
        # update tree dict for functions that returned multiple values
        for fn, replace_list in replacements.items():
            tree_dict = replace_in_nesteddict(tree_dict, fn, replace_list)
        # extract for functions requiring multiple channels
        res_multiple = self.extract_multichannel_functions(
            cell_labels, img, img_bgsub, masks
        )
        return {**res_one, **res_multiple}

    def get_imgs_background_subtract(self, tree_dict, tiles, bgs):
        """
        Get raw and background-subtracted fluorescence images.

        Subtract the median background per trap per z-slice before
        z-reduction, so each slice has its own background level
        removed. This differs from the ``imBackground`` trap function,
        which reports the background of an already z-reduced image.

        Parameters
        ----------
        tree_dict: dict
            Extraction tree specifying channels and functions.
        tiles: array
            Tile images with shape (traps, channels, Z, X, Y).
        bgs: array
            Boolean background masks with shape (traps, Y, X) from
            ``get_background_masks``.

        Returns
        -------
        img: dict
            Raw images keyed by channel name.
        img_bgsub: dict
            Background-subtracted images keyed by channel + '_bgsub',
            or None for channels without background subtraction.
        """
        img = {}
        img_bgsub = {}
        for ch in tree_dict["channels_tree"]:
            if tiles is not None and len(tiles):
                # image data for all traps for a particular channel
                # and time point arranged as (traps, Z, X, Y)
                ch_idx = tree_dict["channels"].index(ch)
                img[ch] = tiles[:, ch_idx, 0, ...]
                if (
                    bgs.any()
                    and ch in self.params.subtract_background
                    and img[ch] is not None
                ):
                    # subtract median background per trap per z
                    # img[ch] has shape (traps, Z, X, Y)
                    # bgs has shape (traps, X, Y)
                    ch_img = img[ch]
                    n_traps, n_z = ch_img.shape[:2]
                    bg_medians = np.empty((n_traps, n_z))
                    for i in range(n_traps):
                        for j in range(n_z):
                            bg_medians[i, j] = bn.nanmedian(
                                ch_img[i, j][bgs[i]]
                            )
                    # subtract: broadcast (traps, Z) over (X, Y)
                    img_bgsub[ch + "_bgsub"] = (
                        ch_img - bg_medians[:, :, np.newaxis, np.newaxis]
                    )
                else:
                    img_bgsub[ch + "_bgsub"] = None
            else:
                img[ch] = None
                img_bgsub[ch + "_bgsub"] = None
        return img, img_bgsub

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
        """Add spatial location within image of each cell."""
        x_df = extract_dict["general/null/centroid_x"]
        y_df = extract_dict["general/null/centroid_y"]
        extract_dict["general/null/image_x"] = x_df.copy()
        extract_dict["general/null/image_y"] = y_df.copy()
        half_width = (self.tiler.tile_size - 1) / 2
        traps = np.array(x_df.index.get_level_values("trap"))
        if np.any(traps):
            for tp in x_df.columns:
                tile_locs = self.tiler.tile_locs.centres_at_time(tp)
                centroid_coords = np.column_stack(
                    (x_df[tp].values, y_df[tp].values)
                )
                coords_in_image = (
                    centroid_coords + tile_locs[traps][:, ::-1] - half_width
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


def replace_in_nesteddict(tree_dict, original, replacement):
    """Replace a string with multiple strings in a nested dict."""
    if isinstance(tree_dict, dict):
        return {
            key: replace_in_nesteddict(value, original, replacement)
            for key, value in tree_dict.items()
        }
    elif isinstance(tree_dict, list):
        new_list = []
        for item in tree_dict:
            if item == original:
                new_list.extend(replacement)
            else:
                new_list.append(
                    replace_in_nesteddict(item, original, replacement)
                )
        return new_list
    elif isinstance(tree_dict, tuple):
        new_items = []
        for item in tree_dict:
            if item == original:
                new_items.extend(replacement)
            else:
                new_items.append(
                    replace_in_nesteddict(item, original, replacement)
                )
        return tuple(new_items)
    else:
        return tree_dict
