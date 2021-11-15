"""
Pipeline and chaining elements.
"""
import os
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
import traceback

import itertools
import yaml

import pandas as pd

from agora.base import ParametersABC, ProcessABC
from pcore.experiment import MetaData
from pcore.io.omero import Dataset, Image
from pcore.haystack import initialise_tf
from pcore.baby_client import DummyRunner
from pcore.segment import Tiler
from pcore.io.writer import TilerWriter, BabyWriter
from pcore.io.signal import Signal
from extraction.core.functions.defaults import exparams_from_meta
from extraction.core.extractor import Extractor
from extraction.core.parameters import Parameters
from postprocessor.core.processor import PostProcessor

# from pcore.experiment import ExperimentOMERO, ExperimentLocal
# from pcore.utils import timed


class PipelineParameters(ParametersABC):
    def __init__(self, general, tiler, baby, extraction, postprocessing):
        self.general = general
        self.tiler = tiler
        self.baby = baby
        self.extraction = extraction
        self.postprocessing = postprocessing

    @classmethod
    def default(cls):
        """
        Load unit test experiment
        """
        return cls(
            general=dict(
                id=19993,
                distributed=0,
                tps=2,
                directory="../data",
                strain="",
                tile_size=96,
                earlystop=dict(
                    min_tp=50,
                    thresh_pos_clogged=0.3,
                    thresh_trap_clogged=7,
                    ntps_to_eval=5,
                ),
            ),
            tiler=dict(),
            baby=dict(tf_version=2),
            extraction=dict(),
            postprocessing=dict(),
        )


class Pipeline(ProcessABC):
    """
    A chained set of Pipeline elements connected through pipes.
    """

    ## default values
    # General
    tile_size = 96
    distributed = 0
    strain = ""
    directory = "output"

    # Tiling, Segmentation,Extraction and Postprocessing should use their own default parameters

    # Early stop for clogging
    earlystop = {
        "min_tp": 50,
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
        with Dataset(int(expt_id)) as conn:
            image_ids = conn.get_images()
            directory = root_dir / conn.unique_name
            if not directory.exists():
                directory.mkdir(parents=True)
                # Download logs to use for metadata
            conn.cache_logs(directory)

        # Modify to the configuration
        config["general"]["directory"] = directory

        # Filter TODO integrate filter onto class and add regex
        image_ids = {k: v for k, v in image_ids.items() if k.startswith(strain_filter)}

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
            with Image(image_id) as image:
                filename = f"{directory}/{image.name}.h5"
                try:
                    os.remove(filename)
                except:
                    pass

                session = initialise_tf(2)
                # Run metadata first
                process_from = 0
                # if True:  # not Path(filename).exists():
                meta = MetaData(directory, filename)
                meta.run()
                tiler = Tiler(
                    image.data, image.metadata, tile_size=general_config["tile_size"]
                )
                # else: TODO add support to continue local experiments?
                #     tiler = Tiler.from_hdf5(image.data, filename)
                #     s = Signal(filename)
                #     process_from = s["/general/None/extraction/volume"].columns[-1]
                #     if process_from > 2:
                #         process_from = process_from - 3
                #         tiler.n_processed = process_from

                writer = TilerWriter(filename)
                runner = BabyRunner(tiler, baby_config=config["baby"])
                bwriter = BabyWriter(filename)
                params = (
                    Parameters.from_dict(config["extraction"])
                    if config["extraction"]
                    else Parameters(**exparams_from_meta(filename))
                )
                ext = Extractor.from_tiler(params, store=filename, tiler=tiler)

                # RUN
                tps = general_config["tps"]
                frac_clogged_traps = 0
                for i in tqdm(
                    range(process_from, tps), desc=image.name, initial=process_from
                ):
                    if frac_clogged_traps < earlystop["thresh_pos_clogged"]:
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

                        tmp = ext.extract_pos(tps=[i])
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
                )
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

        @staticmethod
        def check_earlystop(filename, es_parameters):
            s = Signal(filename)
            df = s["/extraction/general/None/area"]
            frac_clogged_traps = (
                df[df.columns[i - earlystop["ntps_to_eval"] : i]]
                .dropna(how="all")
                .notna()
                .groupby("trap")
                .apply(sum)
                .apply(np.mean, axis=1)
                > earlystop["thresh_trap_clogged"]
            ).mean()
            return frac_clogged_traps
