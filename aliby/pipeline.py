"""
Pipeline and chaining elements.
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
import traceback

from itertools import groupby
import re
import h5py
import yaml
from tqdm import tqdm
from time import perf_counter
from pathos.multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage

from aliby.experiment import MetaData
from aliby.haystack import initialise_tf
from aliby.baby_client import BabyRunner, BabyParameters
from aliby.tile.tiler import Tiler, TilerParameters
from aliby.io.omero import Dataset, Image
from agora.abc import ParametersABC, ProcessABC
from agora.io.writer import TilerWriter, BabyWriter
from agora.io.signal import Signal
from extraction.core.extractor import Extractor, ExtractorParameters
from extraction.core.functions.defaults import exparams_from_meta
from postprocessor.core.processor import PostProcessor, PostProcessorParameters
from postprocessor.compiler import ExperimentCompiler, PageOrganiser

logging.basicConfig(
    filename="aliby.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)


class PipelineParameters(ParametersABC):
    def __init__(
        self, general, tiler, baby, extraction, postprocessing, reporting=None
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
        reporting={},
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
                directory=str(directory),
                strain="",
                earlystop=dict(
                    min_tp=100,
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
    Tiling, Segmentation,Extraction and Postprocessing should use their own default parameters.
    These can be overriden passing the key:value of parameters to override to a PipelineParameters class

    """

    def __init__(self, parameters: PipelineParameters, store=None):
        super().__init__(parameters)

        if store is None:
            store = self.parameters.general["directory"]
        self.store = store

    @classmethod
    def from_yaml(cls, fpath):
        # This is just a convenience function, think before implementing
        # for other processes
        return cls(parameters=PipelineParameters.from_yaml(fpath))

    @classmethod
    def from_existing_h5(cls, fpath):
        with h5py.File(fpath, "r") as f:
            pipeline_parameters = PipelineParameters.from_yaml(f.attrs["parameters"])

        return cls(pipeline_parameters)

    def run(self):
        # Config holds the general information, use in main
        # Steps holds the description of tasks with their parameters
        # Steps: all holds general tasks
        # steps: strain_name holds task for a given strain
        config = self.parameters.to_dict()
        expt_id = config["general"]["id"]
        distributed = config["general"]["distributed"]
        pos_filter = config["general"]["filter"]
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
        filt_int = lambda d, filt: {
            k: v for i, (k, v) in enumerate(d.items()) if i == filt
        }

        filt_str = lambda d, filt: {
            k: v for k, v in image_ids.items() if re.search(filt, k)
        }

        def pick_filter(image_ids, filt):
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
                meta = MetaData(directory, filename)
                from_start = True
                if (
                    not general_config.get("overwrite", False)
                    and Path(filename).exists()
                ):
                    try:
                        print(f"Existing file {filename} will be used.")
                        with h5py.File(filename, "r") as f:
                            tiler = Tiler.from_hdf5(image.data, filename)
                            s = Signal(filename)
                            process_from = (
                                f.attrs["last_processed"]
                                or s.get_raw("/general/None/extraction/volume").columns[
                                    -1
                                ]
                                or 0
                            )
                            # get state array
                            state_array = f.get("state_array", 0)
                        if process_from > 2:
                            processFalsefrom = process_from - 3
                            tiler.n_processed = process_from
                        from_start = False
                    except:
                        pass

                if from_start:  # New experiment or overwriting
                    try:
                        os.remove(filename)
                    except:
                        pass

                    process_from = 0
                    try:
                        meta.run()
                        meta.add_fields(
                            {"omero_id,": config["general"]["id"], "image_id": image_id}
                        )
                        tiler = Tiler.from_image(
                            image, TilerParameters.from_dict(config["tiler"])
                        )
                    except:
                        # Remove and try to run again?
                        pass

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
                        labels, masks = groupby_traps(
                            seg["trap"],
                            seg["cell_label"],
                            seg["edgemasks"],
                            tiler.n_traps,
                        )
                        tmp = ext.run(tps=[i], masks=masks, labels=labels)
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

                    meta.add_fields({"last_processed": i})
                # Run post processing
                post_proc_params = PostProcessorParameters.from_dict(
                    self.parameters.postprocessing
                ).to_dict()
                PostProcessor(filename, post_proc_params).run()

                return 1
        except Exception as e:  # bug in the trap getting
            logging.exception(
                f"Caught exception in worker thread (x = {name}):", exc_info=True
            )
            print(f"Caught exception in worker thread (x = {name}):")
            # This prints the type, value, and stack trace of the
            # current exception being handled.
            traceback.print_exc()
            print()
            raise e
        finally:
            if session:
                session.close()

        try:
            compiler = ExperimentCompiler(None, filepath)
            tmp = compiler.run()
            po = PageOrganiser(tmp, grid_spec=(3, 2))
            po.plot()
            po.save(fullpath / f"{directory}report.pdf")
        except Exception as e:
            print(e)

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


def groupby_traps(traps, labels, edgemasks, ntraps):
    # Group data by traps to pass onto extractor without re-reading hdf5
    iterators = [
        groupby(zip(traps, dset), lambda x: x[0]) for dset in (labels, edgemasks)
    ]
    label_d = {key: [x[1] for x in group] for key, group in iterators[0]}
    mask_d = {
        key: np.dstack([ndimage.morphology.binary_fill_holes(x[1]) for x in group])
        for key, group in iterators[1]
    }
    labels = {i: label_d.get(i, []) for i in range(ntraps)}
    masks = {i: mask_d.get(i, []) for i in range(ntraps)}

    return labels, masks
