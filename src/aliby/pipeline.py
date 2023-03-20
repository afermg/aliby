"""Set up and run pipelines: tiling, segmentation, extraction, and then post-processing."""
import logging
import os
import re
import traceback
import typing as t
from copy import copy
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pathos.multiprocessing import Pool
from tqdm import tqdm

from agora.abc import ParametersABC, ProcessABC
from agora.io.metadata import MetaData, parse_logfiles
from agora.io.reader import StateReader
from agora.io.signal import Signal
from agora.io.writer import (
    LinearBabyWriter,
    StateWriter,
    TilerWriter,
)
from aliby.baby_client import BabyParameters, BabyRunner
from aliby.haystack import initialise_tf
from aliby.io.dataset import dispatch_dataset
from aliby.io.image import dispatch_image
from aliby.tile.tiler import Tiler, TilerParameters
from extraction.core.extractor import Extractor, ExtractorParameters
from extraction.core.functions.defaults import exparams_from_meta
from postprocessor.core.processor import PostProcessor, PostProcessorParameters


class PipelineParameters(ParametersABC):
    """Define parameters for the steps of the pipeline."""

    _pool_index = None

    def __init__(
        self, general, tiler, baby, extraction, postprocessing, reporting
    ):
        """Initialise, but called by a class method - not directly."""
        self.general = general
        self.tiler = tiler
        self.baby = baby
        self.extraction = extraction
        self.postprocessing = postprocessing
        self.reporting = reporting

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
        expt_id = general.get("expt_id", 19993)
        if isinstance(expt_id, Path):
            assert expt_id.exists()

            expt_id = str(expt_id)
            general["expt_id"] = expt_id

        directory = Path(general["directory"])

        # get log files, either locally or via OMERO
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
                earlystop=dict(
                    min_tp=100,
                    thresh_pos_clogged=0.4,
                    thresh_trap_ncells=8,
                    thresh_trap_area=0.9,
                    ntps_to_eval=5,
                ),
                logfile_level="INFO",
                use_explog=True,
            )
        }

        # update default values using inputs
        for k, v in general.items():
            if k not in defaults["general"]:
                defaults["general"][k] = v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    defaults["general"][k][k2] = v2
            else:
                defaults["general"][k] = v

        # define defaults and update with any inputs
        defaults["tiler"] = TilerParameters.default(**tiler).to_dict()

        # Generate a backup channel, for when logfile meta is available
        # but not image metadata.
        backup_ref_channel = None
        if "channels" in meta_d and isinstance(
            defaults["tiler"]["ref_channel"], str
        ):
            backup_ref_channel = meta_d["channels"].index(
                defaults["tiler"]["ref_channel"]
            )
        defaults["tiler"]["backup_ref_channel"] = backup_ref_channel

        defaults["baby"] = BabyParameters.default(**baby).to_dict()
        defaults["extraction"] = (
            exparams_from_meta(meta_d)
            or BabyParameters.default(**extraction).to_dict()
        )
        defaults["postprocessing"] = PostProcessorParameters.default(
            **postprocessing
        ).to_dict()
        defaults["reporting"] = {}

        return cls(**{k: v for k, v in defaults.items()})

    def load_logs(self):
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

    # Specify the group in the h5 files written by each step
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
    def from_yaml(cls, fpath):
        # This is just a convenience function, think before implementing
        # for other processes
        return cls(parameters=PipelineParameters.from_yaml(fpath))

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
        # general information in config
        config = self.parameters.to_dict()
        expt_id = config["general"]["id"]
        distributed = config["general"]["distributed"]
        pos_filter = config["general"]["filter"]
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
            image_ids = conn.get_images()
            directory = self.store or root_dir / conn.unique_name
            if not directory.exists():
                directory.mkdir(parents=True)
            # download logs to use for metadata
            conn.cache_logs(directory)
        # update configuration
        self.parameters.general["directory"] = str(directory)
        config["general"]["directory"] = directory
        self.setLogger(directory)
        # pick particular images if desired
        if pos_filter is not None:
            if isinstance(pos_filter, list):
                image_ids = {
                    k: v
                    for filt in pos_filter
                    for k, v in self.apply_filter(image_ids, filt).items()
                }
            else:
                image_ids = self.apply_filter(image_ids, pos_filter)
        assert len(image_ids), "No images to segment"
        # create pipelines
        if distributed != 0:
            # multiple cores
            with Pool(distributed) as p:
                results = p.map(
                    lambda x: self.run_one_position(*x),
                    [(k, i) for i, k in enumerate(image_ids.items())],
                )
        else:
            # single core
            results = []
            for k, v in tqdm(image_ids.items()):
                r = self.run_one_position((k, v), 1)
                results.append(r)
        return results

    def apply_filter(self, image_ids: dict, filt: int or str):
        """Select images by picking a particular one or by using a regular expression to parse their file names."""
        if isinstance(filt, str):
            # pick images using a regular expression
            image_ids = {
                k: v for k, v in image_ids.items() if re.search(filt, k)
            }
        elif isinstance(filt, int):
            # pick the filt'th image
            image_ids = {
                k: v for i, (k, v) in enumerate(image_ids.items()) if i == filt
            }
        return image_ids

    def run_one_position(
        self,
        name_image_id: t.Tuple[str, str or Path or int],
        index: t.Optional[int] = None,
    ):
        """Set up and run a pipeline for one position."""
        self._pool_index = index
        name, image_id = name_image_id
        # session and filename are defined by calling setup_pipeline.
        # can they be deleted here?
        session = None
        filename = None
        #
        run_kwargs = {"extraction": {"labels": None, "masks": None}}
        try:
            (
                filename,
                meta,
                config,
                process_from,
                tps,
                steps,
                earlystop,
                session,
                trackers_state,
            ) = self._setup_pipeline(image_id)
            loaded_writers = {
                name: writer(filename)
                for k in self.step_sequence
                if k in self.writers
                for name, writer in self.writers[k]
            }
            writer_ow_kwargs = {
                "state": loaded_writers["state"].datatypes.keys(),
                "baby": ["mother_assign"],
            }

            # START PIPELINE
            frac_clogged_traps = 0.0
            min_process_from = min(process_from.values())

            with dispatch_image(image_id)(
                image_id, **self.server_info
            ) as image:
                # initialise steps
                if "tiler" not in steps:
                    steps["tiler"] = Tiler.from_image(
                        image, TilerParameters.from_dict(config["tiler"])
                    )
                if process_from["baby"] < tps:
                    session = initialise_tf(2)
                    steps["baby"] = BabyRunner.from_tiler(
                        BabyParameters.from_dict(config["baby"]),
                        steps["tiler"],
                    )
                    if trackers_state:
                        steps["baby"].crawler.tracker_states = trackers_state
                # limit extraction parameters using the available channels in tiler
                if process_from["extraction"] < tps:
                    # TODO Move this parameter validation into Extractor
                    av_channels = set((*steps["tiler"].channels, "general"))
                    config["extraction"]["tree"] = {
                        k: v
                        for k, v in config["extraction"]["tree"].items()
                        if k in av_channels
                    }
                    config["extraction"]["sub_bg"] = av_channels.intersection(
                        config["extraction"]["sub_bg"]
                    )
                    av_channels_wsub = av_channels.union(
                        [c + "_bgsub" for c in config["extraction"]["sub_bg"]]
                    )
                    tmp = copy(config["extraction"]["multichannel_ops"])
                    for op, (input_ch, _, _) in tmp.items():
                        if not set(input_ch).issubset(av_channels_wsub):
                            del config["extraction"]["multichannel_ops"][op]
                    exparams = ExtractorParameters.from_dict(
                        config["extraction"]
                    )
                    steps["extraction"] = Extractor.from_tiler(
                        exparams, store=filename, tiler=steps["tiler"]
                    )
                    # set up progress meter
                    pbar = tqdm(
                        range(min_process_from, tps),
                        desc=image.name,
                        initial=min_process_from,
                        total=tps,
                    )
                    for i in pbar:
                        if (
                            frac_clogged_traps
                            < earlystop["thresh_pos_clogged"]
                            or i < earlystop["min_tp"]
                        ):
                            # run through steps
                            for step in self.pipeline_steps:
                                if i >= process_from[step]:
                                    result = steps[step].run_tp(
                                        i, **run_kwargs.get(step, {})
                                    )
                                    if step in loaded_writers:
                                        loaded_writers[step].write(
                                            data=result,
                                            overwrite=writer_ow_kwargs.get(
                                                step, []
                                            ),
                                            tp=i,
                                            meta={"last_processed": i},
                                        )
                                    # perform step
                                    if (
                                        step == "tiler"
                                        and i == min_process_from
                                    ):
                                        logging.getLogger("aliby").info(
                                            f"Found {steps['tiler'].n_tiles} traps in {image.name}"
                                        )
                                    elif step == "baby":
                                        # write state and pass info to Extractor
                                        loaded_writers["state"].write(
                                            data=steps[
                                                step
                                            ].crawler.tracker_states,
                                            overwrite=loaded_writers[
                                                "state"
                                            ].datatypes.keys(),
                                            tp=i,
                                        )
                                    elif step == "extraction":
                                        # remove mask/label after extraction
                                        for k in ["masks", "labels"]:
                                            run_kwargs[step][k] = None
                            # check and report clogging
                            frac_clogged_traps = self.check_earlystop(
                                filename, earlystop, steps["tiler"].tile_size
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
                            meta.add_fields({"end_status": "Clogged"})
                            break
                        meta.add_fields({"last_processed": i})
                    # run post-processing
                    meta.add_fields({"end_status": "Success"})
                    post_proc_params = PostProcessorParameters.from_dict(
                        config["postprocessing"]
                    )
                    PostProcessor(filename, post_proc_params).run()
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
            _close_session(session)

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

    # FIXME: Remove this functionality. It used to be for
    # older hdf5 file formats.
    def _load_config_from_file(
        self,
        filename: Path,
        process_from: t.Dict[str, int],
        trackers_state: t.List,
        overwrite: t.Dict[str, bool],
    ):
        with h5py.File(filename, "r") as f:
            for k in process_from.keys():
                if not overwrite[k]:
                    process_from[k] = self.legacy_get_last_tp[k](f)
                    process_from[k] += 1
        return process_from, trackers_state, overwrite

    # FIXME: Remove this functionality. It used to be for
    # older hdf5 file formats.
    @staticmethod
    def legacy_get_last_tp(step: str) -> t.Callable:
        """Get last time-point in different ways depending
        on which step we are using

        To support segmentation in aliby < v0.24
        TODO Deprecate and replace with State method
        """
        switch_case = {
            "tiler": lambda f: f["trap_info/drifts"].shape[0] - 1,
            "baby": lambda f: f["cell_info/timepoint"][-1],
            "extraction": lambda f: f[
                "extraction/general/None/area/timepoint"
            ][-1],
        }
        return switch_case[step]

    def _setup_pipeline(
        self, image_id: int
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
        filename: str
            Path to a h5 file to write to.
        meta: object
            agora.io.metadata.MetaData object
        config: dict
            Configuration parameters.
        process_from: dict
            Gives from which time point each step of the pipeline should start.
        tps: int
            Number of time points.
        steps: dict
        earlystop: dict
            Parameters to check whether the pipeline should be stopped.
        session: None
        trackers_state: list
            States of any trackers from earlier runs.
        """
        config = self.parameters.to_dict()
        # TODO Alan: Verify if session must be passed
        session = None
        earlystop = config["general"].get("earlystop", None)
        process_from = {k: 0 for k in self.pipeline_steps}
        steps = {}
        # check overwriting
        ow_id = config["general"].get("overwrite", 0)
        ow = {step: True for step in self.step_sequence}
        if ow_id and ow_id is not True:
            ow = {
                step: self.step_sequence.index(ow_id) < i
                for i, step in enumerate(self.step_sequence, 1)
            }

        # Set up
        directory = config["general"]["directory"]

        trackers_state: t.List[np.ndarray] = []
        with dispatch_image(image_id)(image_id, **self.server_info) as image:
            filename = Path(f"{directory}/{image.name}.h5")
            meta = MetaData(directory, filename)
            from_start = True if np.any(ow.values()) else False
            # remove existing file if overwriting
            if (
                from_start
                and (
                    config["general"].get("overwrite", False)
                    or np.all(list(ow.values()))
                )
                and filename.exists()
            ):
                os.remove(filename)
            # if the file exists with no previous segmentation use its tiler
            if filename.exists():
                self._log("Result file exists.", "info")
                if not ow["tiler"]:
                    steps["tiler"] = Tiler.from_hdf5(image, filename)
                    try:
                        (
                            process_from,
                            trackers_state,
                            ow,
                        ) = self._load_config_from_file(
                            filename, process_from, trackers_state, ow
                        )
                        # get state array
                        trackers_state = (
                            []
                            if ow["baby"]
                            else StateReader(filename).get_formatted_states()
                        )
                        config["tiler"] = steps["tiler"].parameters.to_dict()
                    except Exception:
                        self._log(f"Overwriting tiling data")

            if config["general"]["use_explog"]:
                meta.run()
            # add metadata not in the log file
            meta.add_fields(
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
            tps = min(config["general"]["tps"], image.data.shape[0])
            return (
                filename,
                meta,
                config,
                process_from,
                tps,
                steps,
                earlystop,
                session,
                trackers_state,
            )


def _close_session(session):
    if session:
        session.close()
