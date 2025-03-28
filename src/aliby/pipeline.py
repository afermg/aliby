"""Set up and run pipelines: tiling, segmentation, extraction, and then post-processing."""
import logging
import os
import re
import traceback
import typing as t
from copy import copy
from importlib.metadata import version
from pathlib import Path
from pprint import pprint

import h5py
import numpy as np
from pathos.multiprocessing import Pool
from tqdm import tqdm

import aliby.global_parameters as global_parameters
from agora.abc import ParametersABC, ProcessABC
from agora.io.metadata import MetaData, parse_logfiles
from agora.io.reader import StateReader
from agora.io.signal import Signal
from agora.io.writer import LinearBabyWriter, StateWriter, TilerWriter
from aliby.baby_client import BabyParameters, BabyRunner
from aliby.haystack import initialise_tf
from aliby.io.dataset import dispatch_dataset
from aliby.io.image import dispatch_image
from aliby.tile.tiler import Tiler, TilerParameters
from extraction.core.extractor import Extractor, ExtractorParameters
from extraction.core.functions.defaults import exparams_from_meta
from postprocessor.core.processor import PostProcessor, PostProcessorParameters

# stop warnings from TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)


class PipelineParameters(ParametersABC):
    """Define parameters for the steps of the pipeline."""

    _pool_index = None

    def __init__(
        self,
        general,
        tiler,
        baby,
        extraction,
        postprocessing,
    ):
        """Initialise, but called by a class method - not directly."""
        self.general = general
        self.tiler = tiler
        self.baby = baby
        self.extraction = extraction
        self.postprocessing = postprocessing

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
            # download logs for metadata
            conn.cache_logs(directory)
        try:
            meta_d = MetaData(directory, None).load_logs()
        except Exception as e:
            logging.getLogger("aliby").warn(
                f"WARNING:Metadata: error when loading: {e}"
            )
            minimal_default_meta = {
                "channels": ["Brightfield"],
                "ntps": [2000],
            }
            # set minimal metadata
            meta_d = minimal_default_meta
        # define default values for general parameters
        tps = meta_d.get("ntps", 2000)
        defaults = {
            "general": dict(
                id=expt_id,
                distributed=0,
                tps=tps,
                directory=str(directory.parent),
                filter="",
                earlystop=global_parameters.earlystop,
                logfile_level="INFO",
                use_explog=True,
            )
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
        # default Tiler parameters
        defaults["tiler"] = TilerParameters.default(**tiler).to_dict()
        # generate a backup channel for when logfile meta is available
        # but not image metadata.
        backup_ref_channel = None
        if "channels" in meta_d and isinstance(
            defaults["tiler"]["ref_channel"], str
        ):
            backup_ref_channel = meta_d["channels"].index(
                defaults["tiler"]["ref_channel"]
            )
        defaults["tiler"]["backup_ref_channel"] = backup_ref_channel
        # default BABY parameters
        defaults["baby"] = BabyParameters.default(**baby).to_dict()
        # default Extraction parmeters
        defaults["extraction"] = exparams_from_meta(meta_d)
        # default PostProcessing parameters
        defaults["postprocessing"] = PostProcessorParameters.default(
            **postprocessing
        ).to_dict()
        return cls(**{k: v for k, v in defaults.items()})

    def load_logs(self):
        """Load and parse log files."""
        parsed_flattened = parse_logfiles(self.log_dir)
        return parsed_flattened


class Pipeline(ProcessABC):
    """
    Initialise and run tiling, segmentation, extraction and post-processing.

    Each step feeds the next one.

    To customise parameters for any step use the PipelineParameters class.stem
    """

    pipeline_steps = ["tiler", "baby", "extraction"]
    step_sequence = [
        "tiler",
        "baby",
        "extraction",
        "postprocessing",
    ]

    # specify the group in the h5 files written by each step
    writer_groups = {
        "tiler": ["trap_info"],
        "baby": ["cell_info"],
        "extraction": ["extraction"],
        "postprocessing": ["postprocessing", "modifiers"],
    }
    writers = {  # TODO integrate Extractor and PostProcessing in here
        "tiler": [("tiler", TilerWriter)],
        "baby": [("baby", LinearBabyWriter), ("state", StateWriter)],
    }

    def __init__(self, parameters: PipelineParameters, store=None):
        """Initialise - not usually called directly."""
        super().__init__(parameters)
        if store is not None:
            store = Path(store)
        self.store = store

    @staticmethod
    def setLogger(
        folder, file_level: str = "INFO", stream_level: str = "WARNING"
    ):
        """Initialise and format logger."""
        logger = logging.getLogger("aliby")
        logger.setLevel(getattr(logging, file_level))
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s:%(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        # for streams - stdout, files, etc.
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, stream_level))
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # create file handler that logs even debug messages
        fh = logging.FileHandler(Path(folder) / "aliby.log", "w+")
        fh.setLevel(getattr(logging, file_level))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    @classmethod
    def from_folder(cls, dir_path):
        """
        Re-process all h5 files in a given folder.

        All files must share the same parameters, even if they have different channels.

        Parameters
        ---------
        dir_path : str or Pathlib
            Folder containing the files.
        """
        # find h5 files
        dir_path = Path(dir_path)
        files = list(dir_path.rglob("*.h5"))
        assert len(files), "No valid files found in folder"
        fpath = files[0]
        # TODO add support for non-standard unique folder names
        with h5py.File(fpath, "r") as f:
            pipeline_parameters = PipelineParameters.from_yaml(
                f.attrs["parameters"]
            )
        pipeline_parameters.general["directory"] = dir_path.parent
        pipeline_parameters.general["filter"] = [fpath.stem for fpath in files]
        # fix legacy post-processing parameters
        post_process_params = pipeline_parameters.postprocessing.get(
            "parameters", None
        )
        if post_process_params:
            pipeline_parameters.postprocessing["param_sets"] = copy(
                post_process_params
            )
            del pipeline_parameters.postprocessing["parameters"]
        return cls(pipeline_parameters)

    @classmethod
    def from_existing_h5(cls, fpath):
        """
        Re-process an existing h5 file.

        Not suitable for more than one file.

        Parameters
        ---------
        fpath: str
            Name of file.
        """
        with h5py.File(fpath, "r") as f:
            pipeline_parameters = PipelineParameters.from_yaml(
                f.attrs["parameters"]
            )
        directory = Path(fpath).parent
        pipeline_parameters.general["directory"] = directory
        pipeline_parameters.general["filter"] = Path(fpath).stem
        post_process_params = pipeline_parameters.postprocessing.get(
            "parameters", None
        )
        if post_process_params:
            pipeline_parameters.postprocessing["param_sets"] = copy(
                post_process_params
            )
            del pipeline_parameters.postprocessing["parameters"]
        return cls(pipeline_parameters, store=directory)

    @property
    def _logger(self):
        return logging.getLogger("aliby")

    def run(self):
        """Run separate pipelines for all positions in an experiment."""
        # display configuration
        config = self.parameters.to_dict()
        for step in config:
            print("\n---\n" + step + "\n---")
            pprint(config[step])
        print()
        # extract from configuration
        expt_id = config["general"]["id"]
        distributed = config["general"]["distributed"]
        position_filter = config["general"]["filter"]
        root_dir = Path(config["general"]["directory"])
        self.server_info = {
            k: config["general"].get(k)
            for k in ("host", "username", "password")
        }
        dispatcher = dispatch_dataset(expt_id, **self.server_info)
        logging.getLogger("aliby").info(
            f"Fetching data using {dispatcher.__class__.__name__}"
        )
        # get log files, either locally or via OMERO
        with dispatcher as conn:
            position_ids = conn.get_images()
            directory = self.store or root_dir / conn.unique_name
            if not directory.exists():
                directory.mkdir(parents=True)
            # get logs to use for metadata
            conn.cache_logs(directory)
        print("Positions available:")
        for i, pos in enumerate(position_ids.keys()):
            print("\t" + f"{i}: " + pos.split(".")[0])
        # update configuration
        self.parameters.general["directory"] = str(directory)
        config["general"]["directory"] = directory
        self.setLogger(directory)
        # pick particular positions if desired
        if position_filter is not None:
            if isinstance(position_filter, list):
                position_ids = {
                    k: v
                    for filt in position_filter
                    for k, v in self.apply_filter(position_ids, filt).items()
                }
            else:
                position_ids = self.apply_filter(position_ids, position_filter)
        if not len(position_ids):
            raise Exception("No images to segment.")
        else:
            print("\nPositions selected:")
            for pos in position_ids:
                print("\t" + pos.split(".")[0])
        # create and run pipelines
        if distributed != 0:
            # multiple cores
            with Pool(distributed) as p:
                results = p.map(
                    lambda x: self.run_one_position(*x),
                    [
                        (position_id, i)
                        for i, position_id in enumerate(position_ids.items())
                    ],
                )
        else:
            # single core
            results = [
                self.run_one_position((position_id, position_id_path), 1)
                for position_id, position_id_path in tqdm(position_ids.items())
            ]
        return results

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

    def run_one_position(
        self,
        name_image_id: t.Tuple[str, str or Path or int],
        index: t.Optional[int] = None,
    ):
        """Set up and run a pipeline for one position."""
        self._pool_index = index
        name, image_id = name_image_id
        # session is defined by calling pipe_pipeline.
        # can it be deleted here?
        session = None
        run_kwargs = {"extraction": {"cell_labels": None, "masks": None}}
        try:
            pipe, session = self.setup_pipeline(image_id, name)
            loaded_writers = {
                name: writer(pipe["filename"])
                for k in self.step_sequence
                if k in self.writers
                for name, writer in self.writers[k]
            }
            writer_overwrite_kwargs = {
                "state": loaded_writers["state"].datatypes.keys(),
                "baby": ["mother_assign"],
            }

            # START PIPELINE
            frac_clogged_traps = 0.0
            min_process_from = min(pipe["process_from"].values())
            with dispatch_image(image_id)(
                image_id, **self.server_info
            ) as image:
                # initialise steps
                if "tiler" not in pipe["steps"]:
                    pipe["config"]["tiler"]["position_name"] = name.split(".")[
                        0
                    ]
                    pipe["steps"]["tiler"] = Tiler.from_image(
                        image,
                        TilerParameters.from_dict(pipe["config"]["tiler"]),
                    )
                if pipe["process_from"]["baby"] < pipe["tps"]:
                    session = initialise_tf(2)
                    pipe["steps"]["baby"] = BabyRunner.from_tiler(
                        BabyParameters.from_dict(pipe["config"]["baby"]),
                        pipe["steps"]["tiler"],
                    )
                    if pipe["trackers_state"]:
                        pipe["steps"]["baby"].crawler.tracker_states = pipe[
                            "trackers_state"
                        ]
                if pipe["process_from"]["extraction"] < pipe["tps"]:
                    exparams = ExtractorParameters.from_dict(
                        pipe["config"]["extraction"]
                    )
                    pipe["steps"]["extraction"] = Extractor.from_tiler(
                        exparams,
                        store=pipe["filename"],
                        tiler=pipe["steps"]["tiler"],
                    )
                    # initiate progress bar
                    pbar = tqdm(
                        range(min_process_from, pipe["tps"]),
                        desc=image.name,
                        initial=min_process_from,
                        total=pipe["tps"],
                    )
                    # run through time points
                    for i in pbar:
                        if (
                            frac_clogged_traps
                            < pipe["earlystop"]["thresh_pos_clogged"]
                            or i < pipe["earlystop"]["min_tp"]
                        ):
                            # run through steps
                            for step in self.pipeline_steps:
                                if i >= pipe["process_from"][step]:
                                    # perform step
                                    result = pipe["steps"][step].run_tp(
                                        i, **run_kwargs.get(step, {})
                                    )
                                    # write to h5 file using writers
                                    # extractor writes to h5 itself
                                    if step in loaded_writers:
                                        loaded_writers[step].write(
                                            data=result,
                                            overwrite=writer_overwrite_kwargs.get(
                                                step, []
                                            ),
                                            tp=i,
                                            meta={"last_processed": i},
                                        )
                                    # clean up
                                    if (
                                        step == "tiler"
                                        and i == min_process_from
                                    ):
                                        logging.getLogger("aliby").info(
                                            f"Found {pipe['steps']['tiler'].no_tiles} traps in {image.name}"
                                        )
                                    elif step == "baby":
                                        # write state
                                        loaded_writers["state"].write(
                                            data=pipe["steps"][
                                                step
                                            ].crawler.tracker_states,
                                            overwrite=loaded_writers[
                                                "state"
                                            ].datatypes.keys(),
                                            tp=i,
                                        )
                                    elif step == "extraction":
                                        # remove masks and labels after extraction
                                        for k in ["masks", "cell_labels"]:
                                            run_kwargs[step][k] = None
                            # check and report clogging
                            frac_clogged_traps = self.check_earlystop(
                                pipe["filename"],
                                pipe["earlystop"],
                                pipe["steps"]["tiler"].tile_size,
                            )
                            if frac_clogged_traps > 0.3:
                                self._log(
                                    f"{name}:Clogged_traps:{frac_clogged_traps}"
                                )
                                frac = np.round(frac_clogged_traps * 100)
                                pbar.set_postfix_str(f"{frac} Clogged")
                        else:
                            # stop if too many traps are clogged
                            self._log(
                                f"{name}:Stopped early at time {i} with {frac_clogged_traps} clogged traps"
                            )
                            pipe["meta"].add_fields({"end_status": "Clogged"})
                            break
                        pipe["meta"].add_fields({"last_processed": i})
                    pipe["meta"].add_fields({"end_status": "Success"})
                    # run post-processing
                    post_proc_params = PostProcessorParameters.from_dict(
                        pipe["config"]["postprocessing"]
                    )
                    PostProcessor(pipe["filename"], post_proc_params).run()
                    self._log("Analysis finished successfully.", "info")
                    return 1
        except Exception as e:
            # catch bugs during setup or run time
            logging.exception(
                f"{name}: Exception caught.",
                exc_info=True,
            )
            # print the type, value, and stack trace of the exception
            traceback.print_exc()
            raise e
        finally:
            close_session(session)

    def setup_pipeline(
        self,
        image_id: int,
        name: str,
    ) -> t.Tuple[
        Path,
        MetaData,
        t.Dict,
        int,
        t.Dict,
        t.Dict,
        t.Optional[int],
        t.List[np.ndarray],
    ]:
        """
        Initialise steps in a pipeline.

        If necessary use a file to re-start experiments already partly run.

        Parameters
        ----------
        image_id : int or str
            Identifier of a data set in an OMERO server or a filename.

        Returns
        -------
        pipe: dict
            With keys
                filename: str
                    Path to a h5 file to write to.
                meta: object
                    agora.io.metadata.MetaData object
                config: dict
                    Configuration parameters.
                process_from: dict
                    Gives time points from which each step of the
                    pipeline should start.
                tps: int
                    Number of time points.
                steps: dict
                earlystop: dict
                    Parameters to check whether the pipeline should
                    be stopped.
                trackers_state: list
                    States of any trackers from earlier runs.
        session: None
        """
        pipe = {}
        config = self.parameters.to_dict()
        # TODO Alan: Verify if session must be passed
        session = None
        pipe["earlystop"] = config["general"].get("earlystop", None)
        pipe["process_from"] = {k: 0 for k in self.pipeline_steps}
        pipe["steps"] = {}
        # check overwriting
        overwrite_id = config["general"].get("overwrite", 0)
        overwrite = {step: True for step in self.step_sequence}
        if overwrite_id and overwrite_id is not True:
            overwrite = {
                step: self.step_sequence.index(overwrite_id) < i
                for i, step in enumerate(self.step_sequence, 1)
            }
        # set up
        directory = config["general"]["directory"]
        pipe["trackers_state"] = []
        with dispatch_image(image_id)(image_id, **self.server_info) as image:
            pipe["filename"] = Path(f"{directory}/{image.name}.h5")
            # load metadata from h5 file
            pipe["meta"] = MetaData(directory, pipe["filename"])
            from_start = True if np.any(overwrite.values()) else False
            # remove existing h5 file if overwriting
            if (
                from_start
                and (
                    config["general"].get("overwrite", False)
                    or np.all(list(overwrite.values()))
                )
                and pipe["filename"].exists()
            ):
                os.remove(pipe["filename"])
            # if the file exists with no previous segmentation use its tiler
            if pipe["filename"].exists():
                self._log("Result file exists.", "info")
                if not overwrite["tiler"]:
                    tiler_params_dict = TilerParameters.default().to_dict()
                    tiler_params_dict["position_name"] = name.split(".")[0]
                    tiler_params = TilerParameters.from_dict(tiler_params_dict)
                    pipe["steps"]["tiler"] = Tiler.from_h5(
                        image, pipe["filename"], tiler_params
                    )
                    try:
                        (
                            process_from,
                            trackers_state,
                            overwrite,
                        ) = self._load_config_from_file(
                            pipe["filename"],
                            pipe["process_from"],
                            pipe["trackers_state"],
                            overwrite,
                        )
                        # get state array
                        pipe["trackers_state"] = (
                            []
                            if overwrite["baby"]
                            else StateReader(
                                pipe["filename"]
                            ).get_formatted_states()
                        )
                        config["tiler"] = pipe["steps"][
                            "tiler"
                        ].parameters.to_dict()
                    except Exception:
                        self._log("Overwriting tiling data")

            if config["general"]["use_explog"]:
                pipe["meta"].run()
            pipe["config"] = config
            # add metadata not in the log file
            pipe["meta"].add_fields(
                {
                    "aliby_version": version("aliby"),
                    "baby_version": version("aliby-baby"),
                    "omero_id": config["general"]["id"],
                    "image_id": image_id
                    if isinstance(image_id, int)
                    else str(image_id),
                    "parameters": PipelineParameters.from_dict(
                        config
                    ).to_yaml(),
                }
            )
            pipe["tps"] = min(config["general"]["tps"], image.data.shape[0])
            return pipe, session

    @staticmethod
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
            cells_used.groupby("trap").sum().apply(np.mean, axis=1)
            / tile_size**2
            > es_parameters["thresh_trap_area"]
        )
        return (traps_above_nthresh & traps_above_athresh).mean()


def close_session(session):
    if session:
        session.close()
