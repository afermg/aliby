"""
Set up and run pipeline.

Include tiling, segmentation, extraction, and then post-processing.
"""

import logging
import multiprocessing
import os
import re
import typing as t
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace
import baby
import baby.errors
import numpy as np
from agora.abc import ParametersABC, ProcessABC
from agora.io.metadata import MetaData
from agora.io.signal import Signal
from agora.io.writers import (
    BabyWriter,
    ExtractorWriter,
    PostProcessorWriter,
    TilerWriter,
    write_meta_to_h5,
)
from extraction.core.extractor import (
    Extractor,
    ExtractorParameters,
    build_extraction_tree_from_meta,
)
from extraction.core.recursive_merge import recursive_merge_extractor
from pathos.multiprocessing import Pool
from postprocessor.core.postprocessing import (
    PostProcessor,
    PostProcessorParameters,
)
from tqdm import tqdm

from aliby.global_settings import global_settings
from aliby.baby_sitter import BabyParameters, BabyRunner
from aliby.io.dataset import dispatch_dataset
from aliby.io.image import dispatch_image
from aliby.tile.tiler import Tiler, TilerParameters


class TqdmHandler(logging.StreamHandler):
    """Route log output through tqdm.write to not disrupt progress bars."""

    def emit(self, record):
        """Write logging message."""
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


class PipelineParameters(ParametersABC):
    """Define parameters for the steps of the pipeline."""

    def __init__(
        self,
        general,
        tiler,
        baby,
        extraction,
        postprocessing,
        metadata=None,
    ):
        """Initialise parameter sets using passed dictionaries."""
        self.general = general
        self.tiler = tiler
        self.baby = baby
        self.extraction = extraction
        self.postprocessing = postprocessing
        self.metadata = metadata

    @classmethod
    def default(
        cls,
        general={},
        tiler={},
        baby={},
        extraction={},
        postprocessing={},
        metadata=None,
    ):
        """
        Initialise parameters for steps of the pipeline.

        Some parameters are extracted from the log files.

        Parameters
        ---------
        general: dict
            Parameters to set up the pipeline.
        tiler: dict
            Parameters for tiler.
        baby: dict (optional)
            Parameters for Baby.
        extraction: dict (optional)
            Parameters for extraction.
        postprocessing: dict (optional)
            Parameters for post-processing.
        metadata: dict (optional)
            Minimal information on the experiment.
            For example,
                metadata = {
                    "channels": ["Brightfield", "GFP", "cy5", "mCherry"],
                    "time_settings/ntimepoints": 240,
                    "time_settings/timeinterval": 300,
                    }
        """
        expt_id, directory = cls.resolve_expt_id(general)
        directory, _, omero_meta = cls.fetch_omero_meta(
            expt_id, general, directory
        )
        meta = cls.build_meta(metadata, directory, omero_meta)
        defaults = cls.build_general_defaults(
            expt_id, directory, meta, general
        )
        cls.apply_ref_z(defaults)
        defaults["tiler"] = cls.build_tiler_defaults(meta, tiler)
        defaults["extraction"] = cls.build_extraction_defaults(
            meta, extraction
        )
        defaults["baby"] = BabyParameters.default(**baby).to_dict()
        defaults["postprocessing"] = PostProcessorParameters.default(
            **postprocessing
        ).to_dict()
        return cls(**{k: v for k, v in defaults.items()})

    @staticmethod
    def resolve_expt_id(general):
        """
        Normalise expt_id and directory from general parameters.

        Parameters
        ----------
        general : dict
            Pipeline general parameters.

        Returns
        -------
        expt_id : str or int
            Normalised experiment identifier.
        directory : Path
            Resolved output directory.
        """
        expt_id = general["expt_id"]
        if isinstance(expt_id, Path) and expt_id.exists():
            expt_id = str(expt_id)
        directory = Path(general.get("directory", "."))
        return expt_id, directory

    @staticmethod
    def fetch_omero_meta(expt_id, general, directory):
        """
        Connect to dataset, cache logs, and retrieve channel and meta info.

        Parameters
        ----------
        expt_id : str or int
            Experiment identifier.
        general : dict
            Pipeline general parameters.
        directory : Path
            Base output directory.

        Returns
        -------
        directory : Path
            Updated directory, namespaced by the dataset's unique name.
        omero_channels : list or None
            Channel order from OMERO or user specification.
        omero_meta : dict or None
            Synthesised minimal metadata for datasets lacking log files.
        """
        with dispatch_dataset(
            expt_id,
            **{k: general.get(k) for k in ("host", "username", "password")},
        ) as conn:
            directory = directory / conn.unique_name
            if not directory.exists():
                directory.mkdir(parents=True)
            conn.cache_logs(directory)
            if "channels" in general:
                omero_channels = general["channels"]
            elif hasattr(conn, "get_channels"):
                omero_channels = conn.get_channels()
            else:
                omero_channels = None
            omero_meta = None
            if hasattr(conn, "get_minimal_meta"):
                omero_meta = conn.get_minimal_meta(omero_channels)
            if omero_meta is None and isinstance(expt_id, int):
                raise ValueError(
                    f"OMERO dataset {expt_id} contains no images."
                )
        return directory, omero_channels, omero_meta

    @staticmethod
    def build_meta(metadata, directory, omero_meta):
        """
        Construct a metadata namespace from supplied or OMERO-derived data.

        Parameters
        ----------
        metadata : dict or None
            Caller-supplied minimal metadata dict, or None.
        directory : Path
            Experiment directory, used when reading log files.
        omero_meta : dict or None
            Synthesised metadata from OMERO, used as fallback.

        Returns
        -------
        meta : SimpleNamespace
            Object with ``minimal`` and ``full`` attributes.
        """
        if metadata is not None:
            if isinstance(metadata, dict):
                return SimpleNamespace(minimal=metadata, full={})
            raise ValueError("metadata must be a dict.")
        return MetaData(directory, omero_meta=omero_meta)

    @staticmethod
    def build_general_defaults(expt_id, directory, meta, general):
        """
        Construct the general and metadata sub-dicts with user overrides applied.

        Parameters
        ----------
        expt_id : str or int
            Normalised experiment identifier.
        directory : Path
            Namespaced experiment directory.
        meta : SimpleNamespace
            Metadata object with ``minimal`` and ``full`` attributes.
        general : dict
            User-supplied general parameters to merge in.

        Returns
        -------
        defaults : dict
            Dict with ``"general"`` and ``"metadata"`` keys populated.
        """
        tps = meta.minimal["time_settings/ntimepoints"]
        defaults = {
            "general": dict(
                expt_id=expt_id,
                distributed=0,
                tps=tps,
                directory=str(directory.parent),
                filter="",
                earlystop=global_settings.earlystop,
                logfile_level="INFO",
                use_explog=True,
            ),
            "metadata": {"minimal": meta.minimal, "full": meta.full},
        }
        for k, v in general.items():
            if k not in defaults["general"]:
                defaults["general"][k] = v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    defaults["general"][k][k2] = v2
            else:
                defaults["general"][k] = v
        return defaults

    @staticmethod
    def apply_ref_z(defaults):
        """
        Update global_settings ref_z from metadata z-section counts.

        Parameters
        ----------
        defaults : dict
            Defaults dict whose ``"metadata"`` key carries the full metadata.
        """
        full = defaults["metadata"]["full"]
        if (
            "number_z_sections" in full
            and "Brightfield" in full["number_z_sections"]
        ):
            ref_z = full["number_z_sections"]["Brightfield"] // 2
            global_settings.imaging_specifications["ref_z"] = ref_z
        elif "zsectioning/nsections" in full:
            ref_z = full["zsectioning/nsections"][0] // 2
            global_settings.imaging_specifications["ref_z"] = ref_z

    @staticmethod
    def build_tiler_defaults(meta, tiler):
        """
        Build tiler parameter dict, including a backup ref channel index.

        Parameters
        ----------
        meta : SimpleNamespace
            Metadata object with ``full`` attribute.
        tiler : dict
            User-supplied tiler overrides.

        Returns
        -------
        tiler_defaults : dict
            Tiler parameters with ``backup_ref_channel`` set.
        """
        tiler_defaults = TilerParameters.default(**tiler).to_dict()
        backup_ref_channel = None
        if "channels" in meta.full and isinstance(
            tiler_defaults["ref_channel"], str
        ):
            backup_ref_channel = meta.full["channels"].index(
                tiler_defaults["ref_channel"]
            )
        tiler_defaults["backup_ref_channel"] = backup_ref_channel
        return tiler_defaults

    @staticmethod
    def build_extraction_defaults(meta, extraction):
        """
        Build extraction parameter dict, merging any user-supplied overrides.

        Parameters
        ----------
        meta : SimpleNamespace
            Metadata object with ``minimal`` attribute.
        extraction : dict
            User-supplied extraction overrides.

        Returns
        -------
        extraction_defaults : dict
            Extraction parameters after merging overrides.
        """
        extraction_defaults = build_extraction_tree_from_meta(meta.minimal)
        if extraction:
            extraction_defaults = recursive_merge_extractor(
                extraction_defaults, extraction
            )
        return extraction_defaults


class Pipeline(ProcessABC):
    """Initialise and run tiling, segmentation, extraction and post-processing."""

    def __init__(self, parameters: PipelineParameters, store=None):
        """Initialise using Pipeline parameters."""
        super().__init__(parameters)
        if store is not None:
            store = Path(store)
        # h5 file
        self.store = store
        config = self.parameters.to_dict()
        self.server_info = {
            k: config["general"].get(k)
            for k in ("host", "username", "password")
        }
        self.expt_id = config["general"]["expt_id"]
        self.setLogger(
            config["general"]["directory"], config["general"]["expt_id"]
        )

    @staticmethod
    def setLogger(
        folder,
        expt_id: str,
        file_level: str = "INFO",
        stream_level: str = "INFO",
    ):
        """Initialise and format logger."""
        # reset per-run warning deduplication flags so warnings are visible
        # on every pipeline run, not just the first in a Python session
        from agora.io import metadata_legacy
        from extraction.core.functions import cell_functions

        metadata_legacy._warned_multiple_files = False
        cell_functions._model_cache = {
            k: v
            for k, v in cell_functions._model_cache.items()
            if v is not None
        }
        logger = logging.getLogger("aliby")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(getattr(logging, file_level))
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        # to print to screen without disrupting tqdm progress bars
        ch = TqdmHandler()
        ch.setLevel(getattr(logging, stream_level))
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # create file handler to log all messages
        logfile_name = f"aliby_{str(expt_id).split('/')[-1]}.log"
        fh = logging.FileHandler(Path(folder) / logfile_name, "w")
        fh.setLevel(getattr(logging, file_level))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def setup(self):
        """Copy logfile and identify each position."""
        config = self.parameters.to_dict()
        # print configuration
        self.log("Using alibylite.", "info")
        try:
            self.log(f"Using Baby {baby.__version__}.", "info")
        except AttributeError:
            self.log("Using original Baby.", "info")
        # extract from configuration
        root_dir = Path(config["general"]["directory"])
        dispatcher = dispatch_dataset(self.expt_id, **self.server_info)
        self.log(
            f"Fetching data using {dispatcher.__class__.__name__}.", "info"
        )
        # get positions_ids and save microscopy log files
        if isinstance(self.expt_id, int):
            self.log(
                f"Connecting to OMERO at {self.server_info['host']} "
                f"(dataset {self.expt_id})...",
                "info",
            )
        try:
            with dispatcher as conn:
                position_ids = conn.get_position_ids()
                directory = self.store or root_dir / conn.unique_name
                if not directory.exists():
                    directory.mkdir(parents=True)
                conn.cache_logs(directory)
                if isinstance(self.expt_id, int):
                    channels = conn.get_channels()
                    first_img = next(iter(conn.ome_class.listChildren()))
                    ntps = first_img.getSizeT()
                    self.log(
                        f"OMERO connection OK. Dataset {self.expt_id} "
                        f"contains {len(position_ids)} position(s), "
                        f"{ntps} timepoint(s), "
                        f"channels: {channels}.",
                        "info",
                    )
        except ConnectionError as e:
            self.log(f"Cannot connect to OMERO: {e}", "error")
            raise
        except AssertionError as e:
            self.log(f"OMERO dataset {self.expt_id} not found: {e}", "error")
            raise
        self.log("Positions available:", "info")
        for i, pos in enumerate(position_ids.keys()):
            self.log(f"\t{i}: {pos.split('.')[0]}", "info")
        # add directory to configuration
        self.parameters.general["directory"] = str(directory)
        return position_ids

    def filter_positions(self, position_filter, position_ids):
        """Select particular positions."""
        if isinstance(position_filter, list):
            selected_ids = {
                k: v
                for filt in position_filter
                for k, v in self.apply_filter(position_ids, filt).items()
            }
        else:
            selected_ids = self.apply_filter(position_ids, position_filter)
        return selected_ids

    def apply_filter(self, position_ids: dict, position_filter: int or str):
        """
        Select positions.

        Either pick a particular position or use a regular expression
        to parse their file names.
        """
        if isinstance(position_filter, str):
            # pick positions using a regular expression
            position_ids = {
                k: v
                for k, v in position_ids.items()
                if re.search(position_filter, k)
            }
        elif isinstance(position_filter, int):
            # pick a particular position
            position_ids = {
                k: v
                for i, (k, v) in enumerate(position_ids.items())
                if i == position_filter
            }
        return position_ids

    def run(self):
        """Run separate pipelines for all positions in an experiment."""
        config = self.parameters.to_dict()
        position_ids = self.setup()
        # pick particular positions if desired
        position_filter = config["general"]["filter"]
        if position_filter is not None:
            position_ids = self.filter_positions(position_filter, position_ids)
        if not len(position_ids):
            raise ValueError("No images to segment.")
        print("Positions selected:")
        for pos in position_ids:
            print("\t" + pos.split(".")[0])
        print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
        if "time_settings/ntimepoints" in config["metadata"]["minimal"]:
            ntps = config["metadata"]["minimal"]["time_settings/ntimepoints"]
            print(f"Processing {ntps} timepoints.")
        # create and run pipelines
        distributed = config["general"]["distributed"]
        if distributed != 0:
            # multiple cores
            with Pool(distributed) as p:
                results = p.map(
                    self.run_one_position,
                    [position_id for position_id in position_ids.items()],
                )
        else:
            # single core
            results = [
                self.run_one_position(position_id)
                for position_id in position_ids.items()
            ]
        return results

    def generate_h5file(self, image_id):
        """Delete any existing and then create h5file for one position."""
        config = self.parameters.to_dict()
        out_dir = config["general"]["directory"]
        with dispatch_image(image_id)(image_id, **self.server_info) as image:
            out_file = Path(f"{out_dir}/{image.name}.h5")
        # remove existing h5 file
        if out_file.exists():
            os.remove(out_file)
        # write minimal microscopy metadata to h5 file
        if config["general"]["use_explog"]:
            write_meta_to_h5(out_file, config["metadata"]["minimal"])
        return out_file

    def run_one_position(
        self, name_image_id: t.Tuple[str, str or Path or int]
    ):
        """Run a pipeline for one position."""
        name, image_id = name_image_id
        config = self.parameters.to_dict()
        config["tiler"]["position_name"] = name.split(".")[0]
        earlystop = config["general"].get("earlystop", None)
        out_file = self.generate_h5file(image_id)
        # instantiate writers
        tiler_writer = TilerWriter(out_file)
        baby_writer = BabyWriter(out_file)
        extractor_writer = ExtractorWriter(out_file)
        postprocessor_writer = PostProcessorWriter(out_file)
        # start pipeline
        frac_clogged_traps = 0.0
        # image here is the connection to OMERO
        with dispatch_image(image_id)(image_id, **self.server_info) as image:
            # initialise tiler
            tiler = Tiler.from_image(
                image,
                TilerParameters.from_dict(config["tiler"]),
                microscopy_metadata=self.parameters.to_dict()["metadata"],
            )
            # initialise Baby
            babyrunner = BabyRunner.from_tiler(
                BabyParameters.from_dict(config["baby"]), tiler=tiler
            )
            # initialise Extractor
            extractor = Extractor.from_tiler(
                ExtractorParameters.from_dict(config["extraction"]),
                store=out_file,
                tiler=tiler,
            )
            # initiate progress bar
            tps = config["general"]["tps"]
            if tps > image.data.shape[0]:
                self.log(
                    f"WARNING: Data appears to have only {image.data.shape[0]}"
                    f" time points not {tps}."
                )
                tps = image.data.shape[0]
            # potentially skip initial_tp number of images
            if tps < config["tiler"]["initial_tp"]:
                raise ValueError(
                    f'Initial time point {config["tiler"]["initial_tp"]}'
                    " is greater than the number of time points."
                )
            all_tps = range(tps - config["tiler"]["initial_tp"])
            progress_bar = tqdm(all_tps, desc=image.name)
            # run through time points
            for i in progress_bar:
                if (
                    frac_clogged_traps < earlystop["thresh_pos_clogged"]
                    or i < earlystop["min_tp"]
                ):
                    # run tiler
                    result = tiler.run_tp(i)
                    tiler_writer.write(
                        data=result,
                        overwrite=[],
                        tp=i,
                    )
                    if i == 0:
                        self.log(
                            f"Found {tiler.no_tiles} traps in {image.name}.",
                            "info",
                        )
                        if tiler.no_tiles == 0:
                            break
                    # segment with Baby
                    try:
                        seg_list, rescaling, inshape = babyrunner.segment_tp(i)
                    except baby.errors.Clogging:
                        self.log(
                            "WARNING: Clogging threshold exceeded in BABY."
                        )
                    # track with Baby
                    try:
                        result = babyrunner.track_tp(
                            seg_list=seg_list,
                            rescaling=rescaling,
                            inshape=inshape,
                            tp=i,
                        )
                    except baby.errors.BadOutput:
                        self.log(
                            "WARNING: Bud has been assigned as its own mother."
                        )
                        raise ValueError("Catastrophic Baby error!")
                    # release the materialised seg list before extraction
                    # runs; otherwise it stays bound until the next
                    # timepoint's segment_tp rebinds it
                    del seg_list, rescaling, inshape
                    # check Baby's result
                    if np.any(
                        [
                            True if not value else False
                            for key, value in result.items()
                        ]
                    ):
                        self.log(
                            f"WARNING: Baby failed at timepoint {i}."
                            " Skipping the rest of this position."
                        )
                        break
                    else:
                        # Baby successful
                        baby_writer.write(
                            data=result,
                            overwrite=["mother_assign"],
                            tile_size=tiler.tile_size,
                            tp=i,
                        )
                    # run extraction
                    result = extractor.run_tp(i)
                    extractor_writer.write(data=result)
                    # check and report clogging
                    frac_clogged_traps = check_earlystop(
                        out_file,
                        earlystop,
                        tiler.tile_size,
                    )
                    if frac_clogged_traps > 0.3:
                        self.log(f"{name}: Clogged_traps:{frac_clogged_traps}")
                        frac = np.round(frac_clogged_traps * 100)
                        progress_bar.set_postfix_str(f"{frac} Clogged")
                else:
                    # stop if too many clogged traps
                    self.log(
                        f"{name}: Stopped early at time {i} with "
                        f"{frac_clogged_traps} clogged traps."
                    )
                    break
            # run post-processing
            if i == 0:
                self.log(f"Position {image.name} failed.", "info")
            else:
                result = PostProcessor(
                    out_file,
                    PostProcessorParameters.from_dict(
                        config["postprocessing"]
                    ),
                ).run()
                postprocessor_writer.write(data=result)
                self.log(
                    f"{config['tiler']['position_name']}: Analysis finished"
                    f" at time point {i} - {i/(len(all_tps)-1)*100:.0f}% complete.",
                    "info",
                )

    @property
    def display_config(self):
        """Show all parameters for each step of the pipeline."""
        config = self.parameters.to_dict()
        for step in config:
            print("\n---\n" + step + "\n---")
            pprint(config[step])
        print()


def check_earlystop(filename: str, es_parameters: dict, tile_size: int):
    """
    Check recent time points for tiles with too many cells.

    Returns the fraction of clogged tiles, where clogged tiles have
    too many cells or too much of their area covered by cells.

    Parameters
    ----------
    filename: str
        Name of h5 file.
    es_parameters: dict
        Parameters defining when early stopping should happen.
        For example:
                {'min_tp': 100,
                'thresh_pos_clogged': 0.4,
                'thresh_trap_ncells': 8,
                'thresh_trap_area': 0.9,
                'ntps_to_eval': 5}
    tile_size: int
        Size of tile.
    """
    # get the area of the cells organised by trap and cell number
    s = Signal(filename)
    df = s.get_raw("/extraction/general/null/area")
    # check the latest time points only
    cells_used = df[
        df.columns[-1 - es_parameters["ntps_to_eval"] : -1]
    ].dropna(how="all")
    # find tiles with too many cells
    traps_above_nthresh = (
        cells_used.groupby("trap").count().apply(np.mean, axis=1)
        > es_parameters["thresh_trap_ncells"]
    )
    # find tiles with cells covering too great a fraction of the tiles' area
    traps_above_athresh = (
        cells_used.groupby("trap").sum().apply(np.mean, axis=1) / tile_size**2
        > es_parameters["thresh_trap_area"]
    )
    return (traps_above_nthresh & traps_above_athresh).mean()
