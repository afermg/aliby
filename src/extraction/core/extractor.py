"""Extract areas, volumes and fluorescence for the cells in one position."""

import copy
import typing as t
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import aliby.global_settings as global_settings
from agora.abc import ParametersABC, StepABC
from agora.io.cells import Cells
from agora.io.dynamic_writer import load_meta
from agora.utils.masks import transform_2d_to_3d
from aliby.tile.tiler import Tiler, find_channel_name
from extraction.core.functions.distributors import reduce_z
from extraction.core.functions.loaders import (
    load_funs,
    load_redfuns,
)

# define types
reduction_method = t.Union[t.Callable, str, None]
extraction_tree = t.Dict[str, t.Dict[reduction_method, t.Dict[str, t.Collection]]]
extraction_result = t.Dict[
    str, t.Dict[reduction_method, t.Dict[str, t.Dict[str, pd.Series]]]
]

# Global variables used to load functions that either analyse cells
# or their background. These global variables both allow the functions
# to be stored in a dictionary for access only on demand and to be
# defined simply in extraction/core/functions.
CELL_FUNS, TRAP_FUNS, ALL_FUNS = load_funs()
REDUCTION_FUNS = load_redfuns()


def extraction_params_from_meta(meta: t.Union[dict, Path, str]):
    """Obtain parameters for extraction from microscopy metadata."""
    if not isinstance(meta, dict):
        # load meta data
        with h5py.File(meta, "r") as f:
            meta = dict(f["/"].attrs.items())
    base = {
        "tree": {"general": {"None": global_settings.outline_functions}},
        "multichannel_ops": {},
    }
    candidate_channels = set(global_settings.possible_imaging_channels)
    default_reductions = {"max"}
    default_fluorescence_metrics = set(global_settings.fluorescence_functions)
    default_reduction_and_fluorescence_metrics = {
        r: default_fluorescence_metrics for r in default_reductions
    }
    # uncomment to add nuc_conv_3d (slow)
    # default_reduction_metrics["None"] = ["nuc_conv_3d"]
    extant_fluorescence_ch = []
    for av_channel in candidate_channels:
        # find matching channels in metadata
        found_channel = find_channel_name(meta.get("channels", []), av_channel)
        if found_channel is not None:
            extant_fluorescence_ch.append(found_channel)
    for ch in extant_fluorescence_ch:
        base["tree"][ch] = copy.deepcopy(default_reduction_and_fluorescence_metrics)
    base["sub_bg"] = extant_fluorescence_ch
    return base


class ExtractorParameters(ParametersABC):
    """Base class to define parameters for extraction."""

    def __init__(
        self,
        tree: extraction_tree,
        sub_bg: set = set(),
        multichannel_ops: t.Dict = {},
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
        multichannel_ops: dict
        """
        self.tree = tree
        self.sub_bg = sub_bg
        self.multichannel_ops = multichannel_ops

    @classmethod
    def default(cls):
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
                k: v for k, v in self.params.tree.items() if k in available_channels
            }
            (
                len(set(self.params.tree).intersection(available_channels))
                == len(set(self.params.tree)),
                "At least one channel was dropped",
            )
            self.params.sub_bg = available_channels.intersection(self.params.sub_bg)
            # add background subtracted channels to those available
            available_channels_bgsub = available_channels.union([
                c + "_bgsub" for c in self.params.sub_bg
            ])
            # remove any multichannel operations requiring a missing channel
            for op, (input_ch, _, _) in self.params.multichannel_ops.items():
                if not set(input_ch).issubset(available_channels_bgsub):
                    self.params.multichannel_ops.pop(op)
        self.load_funs()

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

    def load_funs(self):
        """Define all functions, including custom ones."""
        self.all_cell_funs = CELL_FUNS
        # merge the two dicts
        self.all_funs = ALL_FUNS

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
        elif len(channels) and isinstance(channels[0], str):
            # a subset of channels was specified
            channel_ids = [self.tiler.get_channel_index(ch) for ch in channels]
        elif len(channels) and isinstance(channels[0], int):
            # a list of indices
            channel_ids = channels
        else:
            # a list of the indices of the z stacks
            channel_ids = None
        if z is None:
            # include all Z channels
            z = list(range(self.tiler.pixels.shape[-3]))
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
        tiles: t.List[np.ndarray],
        masks: t.List[np.ndarray],
        cell_function: str,
        cell_labels: t.Dict[int, t.List[int]] = None,
    ) -> t.Tuple[t.Tuple[float or int]]:
        """
        Apply a cell function to all cells at all tiles for one time point.

        Parameters
        ----------
        tiles: list of arrays
            t.List of images.
        masks: list of arrays
            t.List of masks.
        cell_function: str
            Function to apply.
        cell_labels: dict
            A dict with tile_ids as keys and a list of cell labels as
            values.

        Returns
        -------
        res_idx: a tuple of tuples
            A two-tuple comprising a tuple of results and a tuple of
            the tile_id and cell labels
        """

        # Unflatten flat masks and generate cell_labels
        if len(masks) and masks[0].ndim == 2:
            if cell_labels is None:
                cell_labels = {}
            fov_masks_2d = masks
            masks = []
            for i, (masks_2d) in enumerate(fov_masks_2d):
                local_cell_labels, masks_3d = transform_2d_to_3d(masks_2d)
                masks.append(masks_3d)
                if cell_labels is None:
                    cell_labels[i] = local_cell_labels
        elif cell_labels is None:
            self.log("No cell labels given. Sorting cells using index.")

        idx = []
        results = []
        for tile_id, (mask_set, tile, local_cell_labels) in enumerate(
            zip(masks, tiles, cell_labels.values())
        ):
            # ignore empty tiles
            if len(mask_set):
                # find property from the tile
                result = self.all_funs[cell_function](mask_set, tile)
                if cell_function in self.all_cell_funs:
                    # store results for each cell separately
                    for cell_label, val in zip(local_cell_labels, result):
                        results.append(val)
                        idx.append((tile_id, cell_label))
                else:
                    # background (tile) function
                    results.append(result)
                    idx.append(tile_id)
        res_idx = (tuple(results), tuple(idx))
        return res_idx

    def apply_cell_funs(
        self,
        tiles: t.List[np.array],
        masks: t.List[np.array],
        cell_funs: t.List[str],
        **kwargs,
    ) -> t.Dict[str, pd.Series]:
        """
        Return dict with cell_funs as keys and their results as values.

        Use data from one time point.
        """
        d = {
            cell_fun: self.apply_cell_function(
                tiles=tiles, masks=masks, cell_function=cell_fun, **kwargs
            )
            for cell_fun in cell_funs
        }
        return d

    def reduce_extract(
        self,
        tiles: np.ndarray,
        masks: t.List[np.ndarray],
        reduction_cell_funs: t.Dict[reduction_method, t.Collection[str]],
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
        **kwargs: dict
            All other arguments passed to Extractor.apply_cell_funs.

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
                    reduce_dims(tile_data, method=REDUCTION_FUNS[reduction])
                    for tile_data in tiles
                ]
        # calculate cell and tile properties
        d = {
            reduction: self.apply_cell_funs(
                tiles=reduced_tiles.get(reduction, [None for _ in masks]),
                masks=masks,
                cell_funs=cell_funs,
                **kwargs,
            )
            for reduction, cell_funs in reduction_cell_funs.items()
        }
        return d

    def get_cell_labels(self, tp, cell_labels, cells):
        """Get the cell labels per trap as a dict with trap_ids as keys."""
        if cell_labels is None:
            raw_cell_labels = cells.labels_at_time(tp)
            cell_labels = {
                trap_id: raw_cell_labels.get(trap_id, [])
                for trap_id in range(cells.ntraps)
            }
        return cell_labels

    def extract_one_channel(self, tree, cell_labels, img, img_bgsub, masks, **kwargs):
        """Extract as dict all metrics requiring only a single channel."""
        d = {}
        for ch, reduction_cell_funs in tree.items():
            # extract from all images including bright field
            d[ch] = self.reduce_extract(
                # use None for "general" - no fluorescence image
                tiles=img.get(ch, None),
                masks=masks,
                reduction_cell_funs=reduction_cell_funs,
                cell_labels=cell_labels,
                **kwargs,
            )
            if ch != "general" and ch in img_bgsub:
                # extract from background-corrected fluorescence images
                d[ch + "_bgsub"] = self.reduce_extract(
                    tiles=img_bgsub[ch + "_bgsub"],
                    masks=masks,
                    reduction_cell_funs=reduction_cell_funs,
                    cell_labels=cell_labels,
                    **kwargs,
                )
        return d

    def extract_multiple_channels(self, cell_labels, img, img_bgsub, masks):
        """Extract as a dict all metrics requiring multiple channels."""
        # NB multichannel functions do not use tree
        available_channels = set(list(img.keys()) + list(img_bgsub.keys()))
        d = {}
        for multichannel_fun_name, (
            channels,
            reduction,
            multichannel_function,
        ) in self.params.multichannel_ops.items():
            common_channels = set(channels).intersection(available_channels)
            # all required channels should be available
            if len(common_channels) == len(channels):
                for images, suffix in zip([img, img_bgsub], ["", "_bgsub"]):
                    # channels
                    channels_stack = np.stack(
                        [images[ch + suffix] for ch in channels],
                        axis=-1,
                    )
                    # reduce in Z
                    tiles = REDUCTION_FUNS[reduction](channels_stack, axis=1)
                    # set up dict
                    if multichannel_fun_name not in d:
                        d[multichannel_fun_name] = {}
                    if reduction not in d[multichannel_fun_name]:
                        d[multichannel_fun_name][reduction] = {}
                    # apply multichannel function
                    d[multichannel_fun_name][reduction][
                        multichannel_function + suffix
                    ] = self.apply_cell_function(
                        tiles,
                        masks,
                        multichannel_function,
                        cell_labels,
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
            For example: {'general': {'None': ['area', 'volume', 'eccentricity']}}
        tile_size : int
            Size of the tile to be extracted.
        masks : list of arrays
            A list of masks per trap with each mask having dimensions
            (ncells, tile_size, tile_size) and with one mask per cell.
        cell_labels : dict
            A dictionary with roi id as keys, on the second level cell_labels -> list[int] and max_label -> int
        as value.
        **kwargs : keyword arguments
            Passed to extractor.reduce_extract.

        Returns
        -------
        d: dict
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
        if tree is None:
            tree = self.params.tree

        # Only get last mask if it is a list of lists
        # which means that multiple tps are passed as args
        elif isinstance(masks[0], list):
            masks = masks[-1]
            cell_labels = {k: v["labels"] for k, v in cell_labels[-1].items()}
        # find the cell labels as dict with trap_ids as keys
        # If masks are 2d this can be skipped, as that info is contained
        if cell_labels is None and len(masks) and (masks[0].ndim == 3):
            cell_labels = self.get_cell_labels(tp, cell_labels, Cells(self.h5path))
        # find image data for all traps at the time point
        # stored as an array arranged as (traps, channels, 1, Z, X, Y)
        fl_channels = [x for x in tree if x != "general"]
        tiles = self.get_tiles(tp, channels=fl_channels)

        # generate boolean masks for background for each trap
        bgs = get_background_masks(masks, tile_size)
        # get images and background-corrected images as dicts
        # with fluorescnce channels as keys
        img, img_bgsub = self.get_imgs_background_subtract(
            # tree_dict, tiles, bgs
            tree,
            tiles,
            bgs,
        )
        # perform extraction
        res_one = self.extract_one_channel(
            tree, cell_labels, img, img_bgsub, masks, **kwargs
        )
        res_multiple = self.extract_multiple_channels(
            cell_labels, img, img_bgsub, masks
        )
        res = {**res_one, **res_multiple}
        return res

    def get_imgs_background_subtract(self, tree, tiles, bgs):
        """
        Get two dicts of fluorescence images.

        Return images and background subtracted image for all traps
        for one time point.
        """
        img = {}
        img_bgsub = {}
        av_channels = [x for x in tree if x != "general"]

        for ch in tree:
            if tiles is not None and len(tiles) and ch != "general":
                # image data for all traps for a particular channel and
                # time point arranged as (traps, Z, X, Y)
                # we use 0 here to access the single time point available
                img[ch] = tiles[:, av_channels.index(ch), 0]

                if bgs.any() and ch in self.params.sub_bg and img[ch] is not None:
                    # subtract median background
                    bgsub_mapping = map(
                        # move Z to last column to allow subtraction
                        lambda img, bgs: np.moveaxis(img, 0, -1)
                        # median of background over all pixels for each Z section
                        - np.median(img[:, bgs], axis=1),
                        img[ch],
                        bgs,
                    )
                    # apply map and convert to array
                    mapping_result = np.stack(list(bgsub_mapping))
                    # move Z axis back to the second column
                    img_bgsub[ch + "_bgsub"] = np.moveaxis(mapping_result, -1, 1)
            else:
                img[ch] = None
                img_bgsub[ch] = None
        return img, img_bgsub

    def _run_tp(
        self,
        tps: t.List[int] = None,
        tree=None,
        save=False,
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
            For example: {'general': {'None': ['area', 'volume', 'eccentricity']}}
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

        # Parse tracking input, which allows to pas raw tracking output
        raw_tracking_info = kwargs.get("cell_labels", {})
        tracking_vals = list(raw_tracking_info.values())
        if len(tracking_vals) and isinstance(tracking_vals[0], dict):
            kwargs["cell_labels"] = {
                k: v["labels"] for k, v in raw_tracking_info.items()
            }

        # store results in dict
        extract_dict = {}
        for tp in tps:
            # extract for each time point and convert to dict of pd.Series
            extracted_tp = self.extract_tp(
                tp=tp, tile_size=self.tiler.tile_size, tree=tree, **kwargs
            )
            new = flatten_nesteddict(
                extracted_tp,
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
        # TODO move this somewhere more sensible
        # self.add_spatial_locations_of_cells(extract_dict)
        return extract_dict

    def add_spatial_locations_of_cells(self, extract_dict):
        """Add spatial location within image of each cell to extract_dict."""
        x_df = extract_dict["general/None/centroid_x"]
        y_df = extract_dict["general/None/centroid_y"]
        extract_dict["general/None/image_x"] = x_df.copy()
        extract_dict["general/None/image_y"] = y_df.copy()
        half_width = (self.tiler.tile_size - 1) / 2
        traps = np.array(x_df.index.get_level_values("trap"))
        for tp in x_df.columns:
            tile_locs = self.tiler.tile_locs.centres_at_time(tp)
            centroid_coords = np.column_stack((x_df[tp].values, y_df[tp].values))
            coords_in_image = centroid_coords + tile_locs[traps][:, ::-1] - half_width
            extract_dict["general/None/image_x"][tp] = coords_in_image[:, 0]
            extract_dict["general/None/image_y"][tp] = coords_in_image[:, 1]
        if self.tiler.spatial_location is not None:
            extract_dict["general/None/absolute_x"] = (
                extract_dict["general/None/image_x"].copy()
                + self.tiler.spatial_location[0]
            )
            extract_dict["general/None/absolute_y"] = (
                extract_dict["general/None/image_y"].copy()
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
                if isinstance(v2, dict):
                    # measurement that returns multiple features (e.g., CellProfiler Measurements)
                    for feature, values in v2.items():
                        d["/".join((str(k0), k1, feature))] = (
                            pd.Series(*values, name=tp) if to == "series" else v2
                        )
                else:
                    d["/".join((str(k0), k1, k2))] = (
                        pd.Series(*v2, name=tp) if to == "series" else v2
                    )
    return d


def get_background_masks(masks: list[np.ndarray], tile_size: tuple[int]) -> np.ndarray:
    """
    Generate boolean background masks for all tiles.

    Combine masks per trap and then take the logical inverse.
    """
    bgs = ~np.fromiter(
        (get_foreground_from_tile(masks_in_tile, tile_size) for masks_in_tile in masks),
        dtype=(bool, tile_size),
        count=len(masks),
    )
    return bgs


def get_foreground_from_tile(
    masks_in_tile: np.ndarray, tile_size: tuple[int]
) -> np.ndarray:
    """Return the non-background pixels as true."""

    foreground = np.ones(tile_size, dtype=bool)

    if masks_in_tile.any():
        if masks_in_tile.dtype == np.dtype(
            bool
        ):  # Collapse across cell id dimension if bool
            masks_in_tile = np.any(masks_in_tile, axis=0)

        foreground = masks_in_tile > 0

    return foreground


def reduce_dims(img: np.ndarray, method: reduction_method = None) -> np.ndarray:
    """
    Collapse a z-stack into 2d array using method.

    If method is None, return the original data.

    Parameters
    ----------
    img: array
        An array of the image data arranged as (Z, Y, X).
    method: function
        The reduction function.

    Parameters
    ----------
    A 2D array of X,Y dimensions.
    """
    reduced = img
    if method is not None:
        reduced = reduce_z(img, method)
    return reduced
