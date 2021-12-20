"""
Pipeline and chaining elements.
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
import traceback

import itertools
import yaml
from tqdm import tqdm
from time import perf_counter

import numpy as np
import pandas as pd
from pathos.multiprocessing import Pool

from agora.base import ParametersABC, ProcessABC
from aliby.experiment import MetaData
from aliby.io.omero import Dataset, Image
from aliby.haystack import initialise_tf
from aliby.baby_client import BabyRunner, BabyParameters
from aliby.segment import Tiler, TilerParameters
from aliby.io.writer import TilerWriter, BabyWriter
from aliby.io.signal import Signal
from extraction.core.extractor import Extractor, ExtractorParameters
from extraction.core.functions.defaults import exparams_from_meta
from postprocessor.core.processor import PostProcessor, PostProcessorParameters


class PipelineParameters(ParametersABC):
    def __init__(self, general, tiler, baby, extraction, postprocessing):
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
        Load unit test experiment
        :expt_id: Experiment id
        :directory: Output directory

        Provides default parameters for the entire pipeline. This downloads the logfiles and sets the default
        timepoints and extraction parameters from there.
        """
        expt_id = general.get("expt_id", 19993)
        directory = Path(general.get("directory", "../data"))
        with Dataset(int(expt_id), **general.get("server_info")) as conn:
            directory = directory / conn.unique_name
            if not directory.exists():
                directory.mkdir(parents=True)
                # Download logs to use for metadata
            conn.cache_logs(directory)
        meta = MetaData(directory, None).load_logs()
        tps = meta["time_settings/ntimepoints"][0]
        defaults = {
            "general": dict(
                id=expt_id,
                distributed=0,
                tps=tps,
                directory=directory,
                strain="",
                earlystop=dict(
                    min_tp=180,
                    thresh_pos_clogged=0.3,
                    thresh_trap_clogged=7,
                    ntps_to_eval=5,
                ),
            )
        }
        defaults["tiler"] = TilerParameters.default().to_dict()
        defaults["baby"] = BabyParameters.default().to_dict()
        defaults["extraction"] = exparams_from_meta(meta)
        defaults["postprocessing"] = PostProcessorParameters.default().to_dict()
        for k in defaults.keys():
            exec("defaults[k].update(" + k + ")")
        return cls(**{k: v for k, v in defaults.items()})

    def load_logs(self):
        parsed_flattened = parse_logfiles(self.log_dir)
        return parsed_flattened


class Pipeline(ProcessABC):
    """
    A chained set of Pipeline elements connected through pipes.
    """

    # Tiling, Segmentation,Extraction and Postprocessing should use their own default parameters

    # Early stop for clogging
    earlystop = {
        "min_tp": 180,
        "thresh_pos_clogged": 0.3,
        "thresh_trap_clogged": 7,
        "ntps_to_eval": 5,
    }

    def __init__(self, parameters: PipelineParameters):
        super().__init__(parameters)
        self.store = self.parameters.general["directory"]

    @classmethod
    def from_yaml(cls, fpath):
        # This is just a convenience function, think before implementing
        # for other processes
        return cls(parameters=PipelineParameters.from_yaml(fpath))

    def run(self):
        # Config holds the general information, use in main
        # Steps holds the description of tasks with their parameters
        # Steps: all holds general tasks
        # steps: strain_name holds task for a given strain
        config = self.parameters.to_dict()
        expt_id = config["general"]["id"]
        distributed = config["general"]["distributed"]
        strain_filter = config["general"]["strain"]
        root_dir = config["general"]["directory"]
        root_dir = Path(root_dir)

        print("Searching OMERO")
        # Do all initialis
        with Dataset(int(expt_id), **self.general["server_info"]) as conn:
            image_ids = conn.get_images()
            directory = root_dir / conn.unique_name
            if not directory.exists():
                directory.mkdir(parents=True)
                # Download logs to use for metadata
            conn.cache_logs(directory)

        # Modify to the configuration
        self.parameters.general["directory"] = directory
        config["general"]["directory"] = directory

        # Filter TODO integrate filter onto class and add regex
        if isinstance(strain_filter, str):
            image_ids = {
                k: v for k, v in image_ids.items() if k.startswith(strain_filter)
            }
        elif isinstance(strain_filter, int):
            image_ids = {
                k: v for i, (k, v) in enumerate(image_ids.items()) if i == strain_filter
            }

        if distributed != 0:  # Gives the number of simultaneous processes
            with Pool(distributed) as p:
                results = p.map(lambda x: self.create_pipeline(x), image_ids.items())
            return results
        else:  # Sequential
            results = []
            for k, v in image_ids.items():
                r = self.create_pipeline((k, v))
                results.append(r)

    def create_pipeline(self, image_id):
        config = self.parameters.to_dict()
        name, image_id = image_id
        general_config = config["general"]
        session = None
        earlystop = general_config["earlystop"]
        try:
            directory = general_config["directory"]
            with Image(image_id, **self.general["server_info"]) as image:
                filename = f"{directory}/{image.name}.h5"
                try:
                    os.remove(filename)
                except:
                    pass

                # Run metadata first
                process_from = 0
                # if True:  # not Path(filename).exists():
                meta = MetaData(directory, filename)
                meta.run()
                tiler = Tiler.from_image(
                    image, TilerParameters.from_dict(config["tiler"])
                )
                # else: TODO add support to continue local experiments?
                #     tiler = Tiler.from_hdf5(image.data, filename)
                #     s = Signal(filename)
                #     process_from = s["/general/None/extraction/volume"].columns[-1]
                #     if process_from > 2:
                #         process_from = process_from - 3
                #         tiler.n_processed = process_from

                writer = TilerWriter(filename)
                session = initialise_tf(2)
                runner = BabyRunner.from_tiler(
                    BabyParameters.from_dict(config["baby"]), tiler
                )
                bwriter = BabyWriter(filename)

                # Limit extraction parameters during run using the available channels in tiler
                av_channels = set((*tiler.channels, "general"))
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
                for op in config["extraction"]["multichannel_ops"]:
                    config["extraction"]["multichannel_ops"][op] = [
                        x
                        for x in config["extraction"]["multichannel_ops"]
                        if len(x[0]) == len(av_channels_wsub.intersection(x[0]))
                    ]
                config["extraction"]["multichannel_ops"] = {
                    k: v
                    for k, v in config["extraction"]["multichannel_ops"].items()
                    if len(v)
                }

                exparams = ExtractorParameters.from_dict(config["extraction"])
                ext = Extractor.from_tiler(exparams, store=filename, tiler=tiler)

                # RUN
                tps = general_config["tps"]
                frac_clogged_traps = 0
                for i in tqdm(
                    range(process_from, tps), desc=image.name, initial=process_from
                ):
                    if (
                        frac_clogged_traps < earlystop["thresh_pos_clogged"]
                        or i < earlystop["min_tp"]
                    ):
                        t = perf_counter()
                        trap_info = tiler.run_tp(i)
                        logging.debug(f"Timing:Trap:{perf_counter() - t}s")
                        t = perf_counter()
                        writer.write(trap_info, overwrite=[])
                        logging.debug(f"Timing:Writing-trap:{perf_counter() - t}s")
                        t = perf_counter()
                        seg = runner.run_tp(i)
                        logging.debug(f"Timing:Segmentation:{perf_counter() - t}s")
                        # logging.debug(
                        #     f"Segmentation failed:Segmentation:{perf_counter() - t}s"
                        # )
                        t = perf_counter()
                        bwriter.write(seg, overwrite=["mother_assign"])
                        logging.debug(f"Timing:Writing-baby:{perf_counter() - t}s")
                        t = perf_counter()

                        tmp = ext.run(tps=[i])
                        logging.debug(f"Timing:Extraction:{perf_counter() - t}s")
                    else:  # Stop if more than X% traps are clogged
                        logging.debug(
                            f"EarlyStop:{earlystop['thresh_pos_clogged']*100}% traps clogged at time point {i}"
                        )
                        print(
                            f"Stopping analysis at time {i} with {frac_clogged_traps} clogged traps"
                        )
                        break

                    if (
                        i > earlystop["min_tp"]
                    ):  # Calculate the fraction of clogged traps
                        frac_clogged_traps = self.check_earlystop(filename, earlystop)
                        logging.debug(f"Quality:Clogged_traps:{frac_clogged_traps}")
                        print("Frac clogged traps: ", frac_clogged_traps)

                # Run post processing
                post_proc_params = PostProcessorParameters.from_dict(
                    self.parameters.postprocessing
                ).to_dict()
                PostProcessor(filename, post_proc_params).run()
                return True
        except Exception as e:  # bug in the trap getting
            print(f"Caught exception in worker thread (x = {name}):")
            # This prints the type, value, and stack trace of the
            # current exception being handled.
            traceback.print_exc()
            print()
            raise e
        finally:
            if session:
                session.close()

    def check_earlystop(self, filename, es_parameters):
        s = Signal(filename)
        df = s["/extraction/general/None/area"]
        frac_clogged_traps = (
            df[df.columns[-1 - es_parameters["ntps_to_eval"] : -1]]
            .dropna(how="all")
            .notna()
            .groupby("trap")
            .apply(sum)
            .apply(np.mean, axis=1)
            > es_parameters["thresh_trap_clogged"]
        ).mean()
        return frac_clogged_traps
