"""Set up and run pipelines: tiling, segmentation, extraction, and then post-processing."""

import logging
import multiprocessing
import os
import re
import typing as t
from pathlib import Path
from pprint import pprint

import baby
import baby.errors
import numpy as np
import tensorflow as tf
from pathos.multiprocessing import Pool
from tqdm import tqdm

try:
    if baby.__version__:
        from aliby.baby_sitter import BabyParameters, BabyRunner
except AttributeError:
    from aliby.baby_client import BabyParameters, BabyRunner

from agora.abc import ParametersABC, ProcessABC
from agora.io.metadata import MetaData
from agora.io.signal import Signal
from agora.io.writer import LinearBabyWriter, StateWriter, TilerWriter, Writer
from extraction.core.extractor import (
    Extractor,
    ExtractorParameters,
    extraction_params_from_meta,
)
from extraction.core.recursive_merge import recursive_merge_extractor
from postprocessor.core.postprocessing import (
    PostProcessor,
    PostProcessorParameters,
)

import aliby.global_parameters as global_parameters
from aliby.io.dataset import dispatch_dataset
from aliby.io.image import dispatch_image
from aliby.tile.tiler import Tiler, TilerParameters

# stop warnings from TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)


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
        """
        if (
            isinstance(general["expt_id"], Path)
            and general["expt_id"].exists()
        ):
            # for zarr files
            expt_id = str(general["expt_id"])
        else:
            expt_id = general["expt_id"]
        directory = Path(general["directory"])
        # get metadata from log files either locally or via OMERO
        with dispatch_dataset(
            expt_id,
            **{k: general.get(k) for k in ("host", "username", "password")},
        ) as conn:
            directory = directory / conn.unique_name
            if not directory.exists():
                directory.mkdir(parents=True)
            # download microscopy logs for posterity
            conn.cache_logs(directory)
            # get channels to ensure metadata has the correct order
            if hasattr(conn, "get_channels"):
                OMERO_channels = conn.get_channels()
            else:
                OMERO_channels = None
        meta = MetaData(directory, OMERO_channels)
        # define default values for general parameters
        tps = meta.full["time_settings/ntimepoints"]
        defaults = {
            "general": dict(
                expt_id=expt_id,
                distributed=0,
                tps=tps,
                directory=str(directory.parent),
                filter="",
                earlystop=global_parameters.earlystop,
                logfile_level="INFO",
                use_explog=True,
            ),
            "metadata": {"minimal": meta.minimal, "full": meta.full},
        }
        # update default values for general using inputs
        for k, v in general.items():
            if k not in defaults["general"]:
                defaults["general"][k] = v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    defaults["general"][k][k2] = v2
            else:
                defaults["general"][k] = v
        # default Tiler parameters and update with any input tiler
        defaults["tiler"] = TilerParameters.default(**tiler).to_dict()
        # generate a backup channel for when logfile meta is available
        # but not image metadata.
        backup_ref_channel = None
        if "channels" in meta.full and isinstance(
            defaults["tiler"]["ref_channel"], str
        ):
            backup_ref_channel = meta.full["channels"].index(
                defaults["tiler"]["ref_channel"]
            )
        defaults["tiler"]["backup_ref_channel"] = backup_ref_channel
        # defaults for extraction
        defaults["extraction"] = extraction_params_from_meta(meta.minimal)
        # merge any input, a nested dict
        if extraction:
            defaults["extraction"] = recursive_merge_extractor(
                defaults["extraction"], extraction
            )
        # default parameters updated with any inputs
        defaults["baby"] = BabyParameters.default(**baby).to_dict()
        defaults["postprocessing"] = PostProcessorParameters.default(
            **postprocessing
        ).to_dict()
        return cls(**{k: v for k, v in defaults.items()})


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
        logger = logging.getLogger("aliby")
        logger.handlers.clear()
        logger.setLevel(getattr(logging, file_level))
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        # to print to screen
        ch = logging.StreamHandler()
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
        with dispatcher as conn:
            position_ids = conn.get_position_ids()
            directory = self.store or root_dir / conn.unique_name
            if not directory.exists():
                directory.mkdir(parents=True)
            # copy microscopy logs to h5 directory
            conn.cache_logs(directory)
        print("Positions available:")
        for i, pos in enumerate(position_ids.keys()):
            print("\t" + f"{i}: " + pos.split(".")[0])
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
            raise Exception("No images to segment.")
        else:
            print("Positions selected:")
            for pos in position_ids:
                print("\t" + pos.split(".")[0])
        print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
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
            Writer(out_file).write(
                path="/", meta=self.parameters.to_dict()["metadata"]["minimal"]
            )
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
        baby_writer = LinearBabyWriter(out_file)
        babystate_writer = StateWriter(out_file)
        # start pipeline
        initialise_tensorflow()
        frac_clogged_traps = 0.0
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
            # initialise extraction
            extraction = Extractor.from_tiler(
                ExtractorParameters.from_dict(config["extraction"]),
                store=out_file,
                tiler=tiler,
            )
            # initiate progress bar
            tps = config["general"]["tps"]
            if tps > image.data.shape[0]:
                raise Exception(
                    f"Data set appears to have only {image.data.shape[0]} time points not {tps}."
                )
            progress_bar = tqdm(range(tps), desc=image.name)
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
                        meta={"last_processed:": i},
                    )
                    if i == 0:
                        self.log(
                            f"Found {tiler.no_tiles} traps in {image.name}.",
                            "info",
                        )
                    # run Baby
                    try:
                        result = babyrunner.run_tp(i)
                    except baby.errors.Clogging:
                        self.log(
                            "WARNING: Clogging threshold exceeded in BABY."
                        )
                    except baby.errors.BadOutput:
                        self.log(
                            "WARNING: Bud has been assigned as its own mother."
                        )
                        raise Exception("Catastrophic Baby error!")
                    baby_writer.write(
                        data=result,
                        overwrite=["mother_assign"],
                        meta={"last_processed": i},
                        tp=i,
                        tile_size=tiler.tile_size,
                    )
                    babystate_writer.write(
                        data=babyrunner.crawler.tracker_states,
                        overwrite=babystate_writer.datatypes.keys(),
                        tp=i,
                    )
                    # run extraction
                    result = extraction.run_tp(i, cell_labels=None, masks=None)
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
                        f"{name}: Stopped early at time {i} with {frac_clogged_traps} clogged traps"
                    )
                    break
            # run post-processing
            PostProcessor(
                out_file,
                PostProcessorParameters.from_dict(config["postprocessing"]),
            ).run()
            self.log(
                f"{config['tiler']['position_name']}: Analysis finished successfully.",
                "info",
            )
            return 1

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
    df = s.get_raw("/extraction/general/None/area")
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


def initialise_tensorflow(version=2):
    """Initialise tensorflow."""
    if version == 2:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus), "physical GPUs,", len(logical_gpus), "logical GPUs"
            )
