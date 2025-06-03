"""Extract areas, volumes and fluorescence for the cells in one position."""

import copy
import typing as t
from pathlib import Path

import aliby.global_settings as global_settings
import bottleneck as bn
import h5py
import numpy as np
import pandas as pd
from agora.abc import ParametersABC, StepABC
from agora.io.cells import Cells
from agora.io.dynamic_writer import load_meta
from aliby.tile.tiler import Tiler, find_channel_name
from extraction.core.functions.loaders import (
    load_all_functions,
    load_reduction_functions,
)

# define types
reduction_method = t.Union[t.Callable, str, None]
extraction_tree = t.Dict[
    str, t.Dict[reduction_method, t.Dict[str, t.Collection]]
]
extraction_result = t.Dict[
    str, t.Dict[reduction_method, t.Dict[str, t.Dict[str, pd.Series]]]
]

# global reduction functions
REDUCTION_FUNS = load_reduction_functions()


def extraction_params_from_meta(meta: t.Union[dict, Path, str]):
    """Obtain parameters for extraction from microscopy metadata."""
    if not isinstance(meta, dict):
        # load meta data
        with h5py.File(meta, "r") as f:
            meta = dict(f["/"].attrs.items())
    base = {
        "tree": {"general": {"null": global_settings.outline_functions}},
        "multichannel_funs": {},
    }
    candidate_channels = set(global_settings.possible_imaging_channels)
    default_reductions = {"max"}
    default_fluorescence_metrics = set(global_settings.fluorescence_functions)
    default_reduction_and_fluorescence_metrics = {
        r: default_fluorescence_metrics for r in default_reductions
    }
    extant_fluorescence_ch = []
    for av_channel in candidate_channels:
        # find matching channels in metadata
        found_channel = find_channel_name(meta.get("channels", []), av_channel)
        if found_channel is not None:
            extant_fluorescence_ch.append(found_channel)
    for ch in extant_fluorescence_ch:
        base["tree"][ch] = copy.deepcopy(
            default_reduction_and_fluorescence_metrics
        )
    base["sub_bg"] = extant_fluorescence_ch
    return base


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
    elif isinstance(fun, np.ufunc):
        # optimise the reduction function if possible
        return fun.reduce(trap_image, axis=axis)
    else:
        # WARNING: Very slow, only use when no alternatives exist
        return np.apply_along_axis(fun, axis, trap_image)


class ExtractorParameters(ParametersABC):
    """Base class to define parameters for extraction."""

    def __init__(
        self,
        tree: extraction_tree,
        sub_bg: set = set(),
        multichannel_funs: t.Dict = {},
    ):
        """
        Initialise.

        Parameters
        ----------
        tree: dict
            Nested dictionary indicating channels, reduction functions and
            metrics to be used.
            str channel -> U(function, None) reduction -> str metric
            If not of depth three, tree will be filled with None.
        sub_bg: set
        multichannel_funs: dict
            Dict of multichannel functions.
        """
        self.tree = tree
        self.sub_bg = sub_bg
        self.multichannel_funs = multichannel_funs

    @classmethod
    def default(cls):
        """Override ParametersABC default class method."""
        return cls({})

    @classmethod
    def from_meta(cls, meta):
        """Instantiate from the meta data; used by Pipeline."""
        return cls(**extraction_params_from_meta(meta))


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
        store: t.Optional[str] = None,
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
        tiler: pipeline-core.core.segmentation tiler
            Class that contains or fetches the images used for
            segmentation.
        """
        self.params = parameters
        if store:
            self.h5path = store
            self.meta = load_meta(self.h5path)
        else:
            # if no h5 file, use the parameters directly
            self.meta = {"channel": parameters.to_dict()["tree"].keys()}
        if tiler:
            self.tiler = tiler
            available_channels = set((*tiler.channels, "general"))
            # only extract for channels available
            self.params.tree = {
                k: v
                for k, v in self.params.tree.items()
                if k in available_channels
            }
            self.params.sub_bg = available_channels.intersection(
                self.params.sub_bg
            )
        self.get_functions()

    @classmethod
    def from_tiler(
        cls,
        parameters: ExtractorParameters,
        store: str,
        tiler: Tiler,
    ):
        """Initiate from a tiler instance."""
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

    @property
    def channels(self):
        """Get a tuple of the available channels."""
        if not hasattr(self, "_channels"):
            if type(self.params.tree) is dict:
                self._channels = tuple(self.params.tree.keys())
        return self._channels

    @property
    def current_position(self):
        """Return position being analysed."""
        return str(self.h5path).split("/")[-1][:-3]

    @property
    def group(self):
        """Return out path to write in the h5 file."""
        if not hasattr(self, "_out_path"):
            self._group = "/extraction/"
        return self._group

    def get_functions(self):
        """Define all functions."""
        self.all_cell_funs, self.all_funs = load_all_functions()

    def get_tiles(
        self,
        tp: int,
        channels: t.Optional[t.List[t.Union[str, int]]] = None,
        z: t.Optional[t.List[str]] = None,
    ) -> t.Optional[np.ndarray]:
        """
        Find tiles for a given time point, channels, and z-stacks.

        Any additional keyword arguments are passed to
        tiler.get_tiles_timepoint

        Parameters
        ----------
        tp: int
            Time point of interest.
        channels: list of strings (optional)
            Channels of interest.
        z: list of integers (optional)
            Indices for the z-stacks of interest.
        """
        if channels is None:
            # find channels from tiler
            channel_ids = list(range(len(self.tiler.channels)))
        elif len(channels):
            # a subset of channels was specified
            channel_ids = [self.tiler.get_channel_index(ch) for ch in channels]
        else:
            # a list of the indices of the z stacks
            channel_ids = None
        if z is None:
            # include all Z channels
            z = list(range(self.tiler.shape[-3]))
        # get the image data via tiler
        tiles = (
            self.tiler.get_tiles_timepoint(tp, channels=channel_ids, z=z)
            if channel_ids
            else None
        )
        # tiles has dimensions (tiles, channels, 1, Z, X, Y)
        return tiles

    def apply_cell_function(
        self,
        traps: t.List[np.ndarray],
        masks: t.List[np.ndarray],
        cell_function: str,
        cell_labels: t.Dict[int, t.List[int]],
        channels: t.List[str],
    ) -> t.Tuple[t.Union[t.Tuple[float], t.Tuple[t.Tuple[int]]]]:
        """
        Apply a cell function to all cells at all traps for one time point.

        Parameters
        ----------
        traps: list of arrays
            t.List of images.
        masks: list of arrays
            t.List of masks.
        cell_function: str
            Function to apply.
        cell_labels: dict
            A dict with trap_ids as keys and a list of cell labels as
            values.
        channels: list of str
            A list of the channels corresponding to the data in traps.

        Returns
        -------
        res_idx: a tuple of tuples
            A two-tuple comprising a tuple of results and a tuple of
            the tile_id and cell labels
        """
        if cell_labels is None:
            self.log("No cell labels given. Sorting cells using index.")
        cell_fun = True if cell_function in self.all_cell_funs else False
        idx = []
        results = []
        for trap_id, (mask_set, trap, local_cell_labels) in enumerate(
            zip(masks, traps, cell_labels.values())
        ):
            # ignore empty traps
            if len(mask_set):
                # find property from the tile
                result = self.all_funs[cell_function](mask_set, trap, channels)
                if cell_fun:
                    # store results for each cell separately
                    for cell_label, val in zip(local_cell_labels, result):
                        results.append(val)
                        idx.append((trap_id, cell_label))
                else:
                    # background (trap) function
                    results.append(result)
                    idx.append(trap_id)
        res_idx = (tuple(results), tuple(idx))
        return res_idx

    def apply_cell_functions(
        self,
        tiles: t.List[np.array],
        masks: t.List[np.array],
        cell_funs: t.List[str],
        channels: t.List[str],
        **kwargs,
    ) -> t.Dict[str, pd.Series]:
        """
        Return dict with cell_funs as keys and their results as values.

        Use data from one time point.
        """
        d = {
            cell_fun: self.apply_cell_function(
                traps=tiles,
                masks=masks,
                cell_function=cell_fun,
                channels=channels,
                **kwargs,
            )
            for cell_fun in cell_funs
        }
        return d

    def reduce_extract(
        self,
        tiles: np.ndarray,
        masks: t.List[np.ndarray],
        reduction_cell_funs: t.Dict[reduction_method, t.Collection[str]],
        channels: t.List[str],
        **kwargs,
    ) -> t.Dict[str, t.Dict[reduction_method, t.Dict[str, pd.Series]]]:
        """
        Reduce to a 2D image and then extract.

        Parameters
        ----------
        tiles: array
            An array of image data arranged as (tiles, X, Y, Z)
        masks: list of arrays
            An array of masks for each trap: one per cell at the trap
        reduction_cell_funs: dict
            An upper branch of the extraction tree: a dict for which
            keys are reduction functions and values are either a list
            or a set of strings giving the cell functions to apply.
            For example: {'np_max': {'max5px', 'mean', 'median'}}
        channels: list of str
            A list comprising the channel corresponding to the data in tiles.
        **kwargs: dict
            All other arguments passed to Extractor.apply_cell_functions.

        Returns
        ------
        Dict of dataframes with the corresponding reductions and metrics nested.
        """
        # create dict with keys naming the reduction in the z-direction
        # and the reduced data as values
        reduced_tiles = {}
        if tiles is not None:
            for reduction in reduction_cell_funs.keys():
                reduced_tiles[reduction] = [
                    reduce_z(tile_data, REDUCTION_FUNS[reduction])
                    for tile_data in tiles
                ]
        # calculate cell and tile properties
        d = {
            reduction: self.apply_cell_functions(
                tiles=reduced_tiles.get(reduction, [None for _ in masks]),
                masks=masks,
                cell_funs=cell_funs,
                channels=channels,
                **kwargs,
            )
            for reduction, cell_funs in reduction_cell_funs.items()
        }
        return d

    def make_tree_dict(self, tree: extraction_tree):
        """Put extraction tree into a dict."""
        if tree is None:
            # use default
            tree = self.params.tree
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

    def get_masks(self, tp, masks, cells):
        """Get the masks as a list with an array of masks for each trap."""
        # find the cell masks for a given trap as a dict with trap_ids as keys
        if masks is None:
            raw_masks = cells.at_time(tp, kind="mask")
            masks = {trap_id: [] for trap_id in range(cells.ntraps)}
            for trap_id, cells in raw_masks.items():
                if len(cells):
                    masks[trap_id] = np.stack(np.array(cells)).astype(bool)
        # convert to a list of masks
        # one array of size (no cells, tile_size, tile_size) per trap
        masks = [np.array(v) for v in masks.values()]
        return masks

    def get_cell_labels(self, tp, cell_labels, cells):
        """Get the cell labels per trap as a dict with trap_ids as keys."""
        if cell_labels is None:
            raw_cell_labels = cells.labels_at_time(tp)
            cell_labels = {
                trap_id: raw_cell_labels.get(trap_id, [])
                for trap_id in range(cells.ntraps)
            }
        return cell_labels

    def get_background_masks(self, masks, tile_size):
        """
        Generate boolean background masks.

        Combine masks per trap and then take the logical inverse.
        """
        if self.params.sub_bg:
            bgs = ~np.array(
                list(
                    map(
                        # sum over masks for each cell
                        lambda x: (
                            np.sum(x, axis=0)
                            if np.any(x)
                            else np.zeros((tile_size, tile_size))
                        ),
                        masks,
                    )
                )
            ).astype(bool)
        else:
            bgs = np.array([])
        return bgs

    def extract_one_channel(
        self, tree_dict, cell_labels, img, img_bgsub, masks, **kwargs
    ):
        """Extract as dict all metrics requiring only a single channel."""
        d = {}
        for ch, reduction_cell_funs in tree_dict["tree"].items():
            # extract from all images including bright field
            d[ch] = self.reduce_extract(
                # use None for "general" - no fluorescence image
                tiles=img.get(ch, None),
                masks=masks,
                reduction_cell_funs=reduction_cell_funs,
                cell_labels=cell_labels,
                channels=[ch],
                **kwargs,
            )
            if ch != "general":
                # extract from background-corrected fluorescence images
                d[ch + "_bgsub"] = self.reduce_extract(
                    tiles=img_bgsub[ch + "_bgsub"],
                    masks=masks,
                    reduction_cell_funs=reduction_cell_funs,
                    cell_labels=cell_labels,
                    channels=[ch],
                    **kwargs,
                )
        return d

    def extract_multiple_channels(self, cell_labels, img, img_bgsub, masks):
        """
        Extract as a dict all metrics requiring multiple channels.

        Include 'Brightfield'.

        Multichannel functions do not use tree_dict.
        Instead in extraction, parameters include,

            {"multichannel"} : [channels, reduction function,
                                multichannel function name]

        For example, for the ratio mulitchannel function

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
                        # create "multichannel" key
                        d[multichannel_label] = {}
                    if reduction not in d[multichannel_label]:
                        d[multichannel_label][reduction] = {}
                    # apply multichannel function
                    d[multichannel_label][reduction][
                        multichannel_function + suffix
                    ] = self.apply_cell_function(
                        tiles,
                        masks,
                        multichannel_function,
                        cell_labels,
                        channels,
                    )
        return d

    def extract_tp(
        self,
        tp: int,
        tile_size: int,
        tree: t.Optional[extraction_tree] = None,
        masks: t.Optional[t.List[np.ndarray]] = None,
        cell_labels: t.Optional[t.List[int]] = None,
        **kwargs,
    ) -> t.Dict[str, t.Dict[str, t.Dict[str, tuple]]]:
        """
        Extract for an individual time point.

        Parameters
        ----------
        tp : int
            Time point being analysed.
        tree : dict
            Nested dictionary indicating channels, reduction functions
            and metrics to be used.
            For example: {'general': {'null': ['area', 'volume', 'eccentricity']}}
        tile_size : int
            Size of the tile to be extracted.
        masks : list of arrays
            A list of masks per trap with each mask having dimensions
            (ncells, tile_size, tile_size) and with one mask per cell.
        cell_labels : dict
            A dictionary with trap_ids as keys and cell_labels as values.
        **kwargs : keyword arguments
            Passed to extractor.reduce_extract.

        Returns
        -------
        res: dict
            Dictionary of the results with three levels of dictionaries.
            The first level has channels as keys.
            The second level has reduction metrics as keys.
            The third level has cell or background metrics as keys and a
            two-tuple as values.
            The first tuple is the result of applying the metrics to a
            particular cell or trap; the second tuple is either
            (trap_id, cell_label) for a metric applied to a cell or a
            trap_id for a metric applied to a trap.

            An example is d["GFP"]["np_max"]["mean"][0], which gives a tuple
            of the calculated mean GFP fluorescence for all cells.
        """
        # dict of information from extraction tree
        tree_dict = self.make_tree_dict(tree)
        # create a Cells object to extract information from the h5 file
        cells = Cells(self.h5path)
        # find the cell labels as dict with trap_ids as keys
        cell_labels = self.get_cell_labels(tp, cell_labels, cells)
        # get masks one per cell per trap
        masks = self.get_masks(tp, masks, cells)
        # find fluoresence data for all traps at the time point
        # stored as an array arranged as (traps, channels, 1, Z, X, Y)
        tiles = self.get_tiles(tp, channels=tree_dict["channels"])
        # generate boolean masks for background for each trap
        bgs = self.get_background_masks(masks, tile_size)
        # get fluorescence images and background-corrected images as dicts
        # with fluorescence channels as keys
        img, img_bgsub = self.get_imgs_background_subtract(
            tree_dict, tiles, bgs
        )
        # brightfield images
        img["Brightfield"] = self.get_tiles(tp, channels=["Brightfield"])[
            :, 0, 0, ...
        ]
        # perform extraction
        res_one = self.extract_one_channel(
            tree_dict, cell_labels, img, img_bgsub, masks, **kwargs
        )
        res_multiple = self.extract_multiple_channels(
            cell_labels, img, img_bgsub, masks
        )
        res = {**res_one, **res_multiple}
        return res

    def get_imgs_background_subtract(self, tree_dict, tiles, bgs):
        """
        Get two dicts of fluorescence images.

        Return images and background subtracted image for all traps
        for one time point.
        """
        img = {}
        img_bgsub = {}
        for ch, _ in tree_dict["channels_tree"].items():
            # NB ch != is necessary for threading
            if tiles is not None and len(tiles):
                # image data for all traps for a particular channel and
                # time point arranged as (traps, Z, X, Y)
                # we use 0 here to access the single time point available
                img[ch] = tiles[:, tree_dict["channels"].index(ch), 0, ...]
                if (
                    bgs.any()
                    and ch in self.params.sub_bg
                    and img[ch] is not None
                ):
                    # subtract median background
                    bgsub_mapping = map(
                        # move Z to last column to allow subtraction
                        lambda img, bgs: np.moveaxis(img, 0, -1)
                        # median of background over all pixels for each Z section
                        - bn.median(img[:, bgs], axis=1),
                        img[ch],
                        bgs,
                    )
                    # apply map and convert to array
                    mapping_result = np.stack(list(bgsub_mapping))
                    # move Z axis back to the second column
                    img_bgsub[ch + "_bgsub"] = np.moveaxis(
                        mapping_result, -1, 1
                    )
            else:
                img[ch] = None
                img_bgsub[ch] = None
        return img, img_bgsub

    def _run_tp(
        self,
        tps: t.List[int] = None,
        tree=None,
        save=True,
        **kwargs,
    ) -> dict:
        """
        Run extraction for one position and for the specified time points.

        One time point is run at a time in pipeline.
        Save the results to a h5 file.

        Parameters
        ----------
        tps: list of int (optional)
            Time points to include.
        tree: dict (optional)
            Nested dictionary indicating channels, reduction functions and
            metrics to be used.
            For example: {'general': {'null': ['area', 'volume', 'eccentricity']}}
        save: boolean (optional)
            If True, save results to h5 file.
        kwargs: keyword arguments (optional)
            Passed to extract_tp.

        Returns
        -------
        d: dict
            A dict of the extracted data for one position with a concatenated
            string of channel, reduction metric, and cell metric as keys and
            pd.DataFrame of the extracted data for all time points as values.
        """
        if tree is None:
            tree = self.params.tree
        if tps is None:
            tps = list(range(self.meta["time_settings/ntimepoints"][0]))
        elif isinstance(tps, int):
            tps = [tps]
        # store results in dict
        extract_dict = {}
        for tp in tps:
            # extract for each time point and convert to dict of pd.Series
            new = flatten_nesteddict(
                self.extract_tp(
                    tp=tp, tile_size=self.tiler.tile_size, tree=tree, **kwargs
                ),
                to="series",
                tp=tp,
            )
            # concatenate with data extracted from earlier time points
            for key in new.keys():
                extract_dict[key] = pd.concat(
                    (extract_dict.get(key, None), new[key]), axis=1
                )
        # add indices to pd.Series containing the extracted data
        for k in extract_dict.keys():
            indices = ["experiment", "position", "trap", "cell_label"]
            idx = (
                indices[-extract_dict[k].index.nlevels :]
                if extract_dict[k].index.nlevels > 1
                else [indices[-2]]
            )
            extract_dict[k].index.names = idx
        # add cells' spatial locations within the image
        self.add_spatial_locations_of_cells(extract_dict)
        # save
        # if save:
        #   self.save_to_h5(extract_dict)
        return extract_dict

    def add_spatial_locations_of_cells(self, extract_dict):
        """Add spatial location within image of each cell to extract_dict."""
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

    def save_to_h5(self, extract_dict, path=None):
        """Save the extracted data for one position to the h5 file."""
        if path is None:
            path = self.h5path
        for extract_name, data in extract_dict.items():
            dset_path = "/extraction/" + extract_name
            add_df_to_h5(path, dset_path, data)

    def get_meta(self, flds: t.Union[str, t.Collection]):
        """Obtain metadata for one or multiple fields."""
        if isinstance(flds, str):
            flds = [flds]
        meta_short = {k.split("/")[-1]: v for k, v in self.meta.items()}
        return {
            f: meta_short.get(f, self.default_meta.get(f, None)) for f in flds
        }


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
        Specifies the format of the output, either pd.Series (default)
        or a list
    tp: int
        Time point used to name the pd.Series

    Returns
    -------
    d: dict
        A dict with a concatenated string of channel, reduction metric,
        and cell metric as keys and either a pd.Series or a list of the
        corresponding extracted data as values.
    """
    d = {}
    for k0, v0 in nest.items():
        for k1, v1 in v0.items():
            for k2, v2 in v1.items():
                d["/".join((k0, k1, k2))] = (
                    pd.Series(*v2, name=tp) if to == "series" else v2
                )
    return d
