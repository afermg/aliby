"""
Pipeline and chaining elements.
"""
import logging
import os
import re
import traceback
import typing as t
from copy import copy
from importlib.metadata import version
from pathlib import Path, PosixPath
from time import perf_counter
from typing import Union

import h5py
import numpy as np
import pandas as pd
from pathos.multiprocessing import Pool
from tqdm import tqdm

from agora.abc import ParametersABC, ProcessABC
from agora.io.metadata import MetaData, parse_logfiles
from agora.io.reader import StateReader
from agora.io.signal import Signal
from agora.io.writer import (  # BabyWriter,
    LinearBabyWriter,
    StateWriter,
    TilerWriter,
)
from aliby.baby_client import BabyParameters, BabyRunner
from aliby.haystack import initialise_tf
from aliby.io.dataset import Dataset, DatasetLocal
from aliby.io.image import get_image_class
from aliby.tile.tiler import Tiler, TilerParameters
from extraction.core.extractor import Extractor, ExtractorParameters
from extraction.core.functions.defaults import exparams_from_meta
from postprocessor.core.processor import PostProcessor, PostProcessorParameters

# from postprocessor.compiler import ExperimentCompiler, PageOrganiser

logging.basicConfig(
    filename="aliby.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)


class PipelineParameters(ParametersABC):
    _pool_index = None

    def __init__(
        self, general, tiler, baby, extraction, postprocessing, reporting
    ):
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
        # reporting={},
    ):
        """
        Load unit test experiment
        :expt_id: Experiment id
        :directory: Output directory

        Provides default parameters for the entire pipeline. This downloads the logfiles and sets the default
        timepoints and extraction parameters from there.
        """
        expt_id = general.get("expt_id", 19993)
        if isinstance(expt_id, PosixPath):
            expt_id = str(expt_id)
            general["expt_id"] = expt_id

        directory = Path(general.get("directory", "../data"))
        dataset_wrapper = (
            lambda x: DatasetLocal(x)
            if isinstance(expt_id, str)
            else Dataset(int(x), **general.get("server_info"))
        )
        with dataset_wrapper(expt_id) as conn:
            directory = directory / conn.unique_name
            if not directory.exists():
                directory.mkdir(parents=True)
                # Download logs to use for metadata
            conn.cache_logs(directory)
        try:
            meta_d = MetaData(directory, None).load_logs()
        except Exception as e:
            print("WARNING: Metadata could not be loaded: {}".format(e))
            # Set minimal metadata
            meta_d = {
                "channels/channel": "Brightfield",
                "time_settings/ntimepoints": [200],
            }

        tps = meta_d["time_settings/ntimepoints"][0]
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
            )
        }

        for k, v in general.items():  # Overwrite general parameters
            if k not in defaults["general"]:
                defaults["general"][k] = v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    defaults["general"][k][k2] = v2
            else:
                defaults["general"][k] = v

        defaults["tiler"] = TilerParameters.default(**tiler).to_dict()
        defaults["baby"] = BabyParameters.default(**baby).to_dict()
        defaults["extraction"] = (
            exparams_from_meta(meta_d)
            or BabyParameters.default(**extraction).to_dict()
        )
        defaults["postprocessing"] = {}
        defaults["reporting"] = {}

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
    A chained set of Pipeline elements connected through pipes.
    Tiling, Segmentation,Extraction and Postprocessing should use their own default parameters.
    These can be overriden passing the key:value of parameters to override to a PipelineParameters class

    """

    iterative_steps = ["tiler", "baby", "extraction"]

    step_sequence = [
        "tiler",
        "baby",
        "extraction",
        "postprocessing",
    ]

    # Indicate step-writer groupings to perform special operations during step iteration
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
        super().__init__(parameters)

        if store is not None:
            store = Path(store)
        self.store = store

    @classmethod
    def from_yaml(cls, fpath):
        # This is just a convenience function, think before implementing
        # for other processes
        return cls(parameters=PipelineParameters.from_yaml(fpath))

    @classmethod
    def from_folder(cls, dir_path):
        """
        Constructor to re-process all files in a given folder.

        Assumes all files share the same parameters (even if they don't share
        the same channel set).

        Parameters
        ---------
        dir_path : str or Pathlib indicating the folder containing the files to process
        """
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

        # Fix legacy postprocessing parameters
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
        Constructor to process an existing hdf5 file.
        Notice that it forces a single file, not suitable for multiprocessing of certain positions.

        It i s also used as a base for a folder-wide reprocessing.
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

    def run(self):
        # Config holds the general information, use in main
        # Steps holds the description of tasks with their parameters
        # Steps: all holds general tasks
        # steps: strain_name holds task for a given strain
        config = self.parameters.to_dict()
        expt_id = config["general"]["id"]
        distributed = config["general"]["distributed"]
        pos_filter = config["general"]["filter"]
        root_dir = Path(config["general"]["directory"])

        print("Searching OMERO")
        # Do all all initialisations

        dataset_wrapper = DatasetLocal if isinstance(expt_id, str) else Dataset
        with dataset_wrapper(
            expt_id, **self.general.get("server_info", {})
        ) as conn:
            image_ids = conn.get_images()

            directory = self.store or root_dir / conn.unique_name

            if not directory.exists():
                directory.mkdir(parents=True)

            # Download logs to use for metadata
            conn.cache_logs(directory)

        # Modify to the configuration
        self.parameters.general["directory"] = str(directory)
        config["general"]["directory"] = directory

        # Filter TODO integrate filter onto class and add regex
        def filt_int(d: dict, filt: int):
            return {k: v for i, (k, v) in enumerate(d.items()) if i == filt}

        def filt_str(image_ids: dict, filt: str):
            return {k: v for k, v in image_ids.items() if re.search(filt, k)}

        def pick_filter(image_ids: dict, filt: Union[int, str]):
            if isinstance(filt, str):
                image_ids = filt_str(image_ids, filt)
            elif isinstance(filt, int):
                image_ids = filt_int(image_ids, filt)
            return image_ids

        if isinstance(pos_filter, list):
            image_ids = {
                k: v
                for filt in pos_filter
                for k, v in pick_filter(image_ids, filt).items()
            }
        else:
            image_ids = pick_filter(image_ids, pos_filter)

        assert len(image_ids), "No images to segment"

        if distributed != 0:  # Gives the number of simultaneous processes
            with Pool(distributed) as p:
                results = p.map(
                    lambda x: self.create_pipeline(*x),
                    [(k, i) for i, k in enumerate(image_ids.items())],
                    # num_cpus=distributed,
                    # position=0,
                )

        else:  # Sequential
            results = []
            for k, v in tqdm(image_ids.items()):
                r = self.create_pipeline((k, v), 1)
                results.append(r)

        return results

    def create_pipeline(
        self,
        image_id: t.Tuple[str, t.Union[str, PosixPath, int]],
        index: t.Optional[int] = None,
    ):
        """ """
        self._pool_index = index
        name, image_id = image_id
        session = None
        filename = None
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
            frac_clogged_traps = 0
            min_process_from = min(process_from.values())

            with get_image_class(image_id)(
                image_id, **self.general.get("server_info", {})
            ) as image:

                # Initialise Steps
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

                # Limit extraction parameters during run using the available channels in tiler
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
                    pbar = tqdm(
                        range(min_process_from, tps),
                        desc=image.name,
                        initial=min_process_from,
                        total=tps,
                        # position=index + 1,
                    )
                    for i in pbar:

                        if (
                            frac_clogged_traps
                            < earlystop["thresh_pos_clogged"]
                            or i < earlystop["min_tp"]
                        ):

                            for step in self.iterative_steps:
                                if i >= process_from[step]:
                                    t = perf_counter()
                                    result = steps[step].run_tp(
                                        i, **run_kwargs.get(step, {})
                                    )
                                    logging.debug(
                                        f"Timing:{step}:{perf_counter() - t}s"
                                    )
                                    if step in loaded_writers:
                                        t = perf_counter()
                                        loaded_writers[step].write(
                                            data=result,
                                            overwrite=writer_ow_kwargs.get(
                                                step, []
                                            ),
                                            tp=i,
                                            meta={"last_processed": i},
                                        )
                                        logging.debug(
                                            f"Timing:Writing-{step}:{perf_counter() - t}s"
                                        )

                                    # Step-specific actions
                                    if (
                                        step == "tiler"
                                        and i == min_process_from
                                    ):
                                        print(
                                            f"Found {steps['tiler'].n_traps} traps in {image.name}"
                                        )
                                    elif (
                                        step == "baby"
                                    ):  # Write state and pass info to ext
                                        loaded_writers["state"].write(
                                            data=steps[
                                                step
                                            ].crawler.tracker_states,
                                            overwrite=loaded_writers[
                                                "state"
                                            ].datatypes.keys(),
                                            tp=i,
                                        )
                                    elif (
                                        step == "extraction"
                                    ):  # Remove mask/label after ext
                                        for k in ["masks", "labels"]:
                                            run_kwargs[step][k] = None

                            frac_clogged_traps = self.check_earlystop(
                                filename, earlystop, steps["tiler"].tile_size
                            )
                            logging.debug(
                                f"Quality:Clogged_traps:{frac_clogged_traps}"
                            )

                            frac = np.round(frac_clogged_traps * 100)
                            pbar.set_postfix_str(f"{frac} Clogged")
                        else:  # Stop if more than X% traps are clogged
                            logging.debug(
                                f"EarlyStop:{earlystop['thresh_pos_clogged']*100}% traps clogged at time point {i}"
                            )
                            print(
                                f"Stopping analysis at time {i} with {frac_clogged_traps} clogged traps"
                            )
                            meta.add_fields({"end_status": "Clogged"})
                            break

                        meta.add_fields({"last_processed": i})

                    # Run post-processing
                    meta.add_fields({"end_status": "Success"})
                    post_proc_params = PostProcessorParameters.from_dict(
                        config["postprocessing"]
                    )
                    PostProcessor(filename, post_proc_params).run()

                    return 1

        except Exception as e:  # bug during setup or runtime
            logging.exception(
                f"Caught exception in worker thread (x = {name}):",
                exc_info=True,
            )
            print(f"Caught exception in worker thread (x = {name}):")
            # This prints the type, value, and stack trace of the
            # current exception being handled.
            traceback.print_exc()
            raise e
        finally:
            _close_session(session)

        # try:
        #     compiler = ExperimentCompiler(None, filepath)
        #     tmp = compiler.run()
        #     po = PageOrganiser(tmp, grid_spec=(3, 2))
        #     po.plot()
        #     po.save(fullpath / f"{directory}report.pdf")
        # except Exception as e:
        #     print("Report failed: {}".format(e))

    @staticmethod
    def check_earlystop(filename: str, es_parameters: dict, tile_size: int):
        s = Signal(filename)
        df = s["/extraction/general/None/area"]
        cells_used = df[
            df.columns[-1 - es_parameters["ntps_to_eval"] : -1]
        ].dropna(how="all")
        traps_above_nthresh = (
            cells_used.groupby("trap").count().apply(np.mean, axis=1)
            > es_parameters["thresh_trap_ncells"]
        )
        traps_above_athresh = (
            cells_used.groupby("trap").sum().apply(np.mean, axis=1)
            / tile_size**2
            > es_parameters["thresh_trap_area"]
        )

        return (traps_above_nthresh & traps_above_athresh).mean()

    def _load_config_from_file(
        self,
        filename: PosixPath,
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
        PosixPath,
        MetaData,
        t.Dict,
        int,
        t.Dict,
        t.Dict,
        t.Optional[int],
        t.List[np.ndarray],
    ]:
        """
        Initialise pipeline components and if necessary use
        exising file to continue existing experiments.


        Parameters
        ----------
        image_id : int
            identifier of image in OMERO server, or filename

        Returns
        ---------
        filename: str
        meta:
        config:
        process_from:
        tps:
        steps:
        earlystop:
        session:
        trackers_state:

        Examples
        --------
        FIXME: Add docs.

        """
        config = self.parameters.to_dict()
        pparams = config
        image_id = image_id
        general_config = config["general"]
        session = None
        earlystop = general_config.get("earlystop", None)
        process_from = {k: 0 for k in self.iterative_steps}
        steps = {}
        ow = {k: 0 for k in self.step_sequence}

        # check overwriting
        ow_id = general_config.get("overwrite", 0)
        ow = {step: True for step in self.step_sequence}
        if ow_id and ow_id is not True:
            ow = {
                step: self.step_sequence.index(ow_id) < i
                for i, step in enumerate(self.step_sequence, 1)
            }

        # Set up
        directory = general_config["directory"]

        trackers_state: t.List[np.ndarray] = []
        with get_image_class(image_id)(
            image_id, **self.general.get("server_info", {})
        ) as image:
            filename = Path(f"{directory}/{image.name}.h5")
            meta = MetaData(directory, filename)

            from_start = True if np.any(ow.values()) else False
            # If no previous segmentation and keep tiler
            if filename.exists():
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
                        pass

                # Delete datasets to overwrite and update pipeline data
                # Use existing parameters
                with h5py.File(filename, "a") as f:
                    pparams = PipelineParameters.from_yaml(
                        f.attrs["parameters"]
                    ).to_dict()

                    for k, v in ow.items():
                        if v:
                            for gname in self.writer_groups[k]:
                                if gname in f:
                                    del f[gname]

                        pparams[k] = config[k]
                meta.add_fields(
                    {
                        "parameters": PipelineParameters.from_dict(
                            pparams
                        ).to_yaml()
                    },
                    overwrite=True,
                )

            if from_start:  # New experiment or overwriting
                if (
                    config.get("overwrite", False) is True
                    or np.all(list(ow.values()))
                ) and filename.exists():
                    os.remove(filename)

                meta.run()
                meta.add_fields(  # Add non-logfile metadata
                    {
                        "aliby_version": version("aliby"),
                        "baby_version": version("aliby-baby"),
                        "omero_id": config["general"]["id"],
                        "image_id": image_id,
                        "parameters": PipelineParameters.from_dict(
                            pparams
                        ).to_yaml(),
                    }
                )

            tps = min(general_config["tps"], image.data.shape[0])

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
