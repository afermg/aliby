from pathlib import Path
import json
from time import perf_counter
import logging

from pathos.multiprocessing import Pool
from multiprocessing import set_start_method
import numpy as np


# set_start_method("spawn")

from tqdm import tqdm
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import operator

from baby.brain import BabyBrain

from pcore.experiment import MetaData
from pcore.io.omero import Dataset, Image
from pcore.haystack import initialise_tf
from pcore.baby_client import DummyRunner
from pcore.segment import Tiler
from pcore.io.writer import TilerWriter, BabyWriter
from pcore.utils import timed

from pcore.io.signal import Signal
from extraction.core.functions.defaults import exparams_from_meta
from extraction.core.extractor import Extractor
from extraction.core.parameters import Parameters
from extraction.core.functions.defaults import get_params
from postprocessor.core.processor import PostProcessorParameters, PostProcessor


def pipeline(image_id, tps=10, tf_version=2):
    name, image_id = image_id
    try:
        # Initialise tensorflow
        session = initialise_tf(tf_version)
        with Image(image_id) as image:
            print(f"Getting data for {image.name}")
            tiler = Tiler(image.data, image.metadata, image.name)
            writer = TilerWriter(f"../data/test2/{image.name}.h5")
            runner = DummyRunner(tiler)
            bwriter = BabyWriter(f"../data/test2/{image.name}.h5")
            for i in tqdm(range(0, tps), desc=image.name):
                trap_info = tiler.run_tp(i)
                writer.write(trap_info, overwrite=[])
                seg = runner.run_tp(i)
                bwriter.write(seg, overwrite=["mother_assign"])
            return True
    except Exception as e:  # bug in the trap getting
        print(f"Caught exception in worker thread (x = {name}):")
        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()
        print()
        raise e
    finally:
        # Close session
        if session:
            session.close()


@timed("Position")
def create_pipeline(image_id, **config):
    name, image_id = image_id
    general_config = config.get("general", None)
    assert general_config is not None
    session = None
    earlystop = config.get(
        "earlystop",
        {
            "min_tp": 50,
            "thresh_pos_clogged": 0.3,
            "thresh_trap_clogged": 7,
            "ntps_to_eval": 5,
        },
    )
    try:
        directory = general_config.get("directory", "")
        with Image(image_id) as image:
            filename = f"{directory}/{image.name}.h5"
            # Run metadata first
            process_from = 0
            if True:  # not Path(filename).exists():
                meta = MetaData(directory, filename)
                meta.run()
                tiler = Tiler(
                    image.data,
                    image.metadata,
                    tile_size=general_config.get("tile_size", 117),
                )
            else:
                tiler = Tiler.from_hdf5(image.data, filename)
                s = Signal(filename)
                process_from = s["/general/None/extraction/volume"].columns[-1]
                if process_from > 2:
                    process_from = process_from - 3
                    tiler.n_processed = process_from

            writer = TilerWriter(filename)
            baby_config = config.get("baby", None)
            assert baby_config is not None  # TODO add defaults
            tf_version = baby_config.get("tf_version", 2)
            session = initialise_tf(tf_version)
            runner = DummyRunner(tiler)
            bwriter = BabyWriter(filename)
            params = Parameters(**exparams_from_meta(filename))
            ext = Extractor.from_tiler(params, store=filename, tiler=tiler)
            # RUN
            tps = general_config.get("tps", 0)
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
                    try:
                        seg = runner.run_tp(i)
                        logging.debug(f"Timing:Segmentation:{perf_counter() - t}s")
                    except:
                        logging.debug(
                            f"Segmentation failed:Segmentation:{perf_counter() - t}s"
                        )
                    t = perf_counter()
                    bwriter.write(seg, overwrite=["mother_assign"])
                    logging.debug(f"Timing:Writing-baby:{perf_counter() - t}s")
                    t = perf_counter()
                    ext.extract_pos(tps=[i])
                    logging.debug(f"Timing:Extraction:{perf_counter() - t}s")
                else:  # Stop if more than 10% traps are clogged
                    logging.debug(
                        f"EarlyStop:{earlystop['thresh_pos_clogged']*100}% traps clogged at time point {i}"
                    )
                    print(
                        f"Breaking experiment at time {i} with {frac_clogged_traps} clogged traps"
                    )
                    break

                if i > earlystop["min_tp"]:  # Calculate the fraction of clogged traps
                    s = Signal(filename)
                    df = s["/extraction/general/None/area"]
                    frac_clogged_traps = (
                        df[df.columns[i - earlystop["ntps_to_eval"] : i]]
                        .dropna(how="all")
                        .notna()
                        .groupby("trap")
                        .apply(sum)
                        .apply(np.nanmean, axis=1)
                        > earlystop["thresh_trap_clogged"]
                    ).mean()
                    logging.debug(f"Quality:Clogged_traps:{frac_clogged_traps}")
                    print("Frac clogged traps: ", frac_clogged_traps)

            # Run post processing
            # post_proc_params = PostProcessorParameters.default()
            # post_process(filename, post_proc_params)
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


@timed("Post-processing")
def post_process(filepath, params):
    pp = PostProcessor(filepath, params)
    tmp = pp.run()
    return tmp


# instantiating the decorator
@timed("Pipeline")
def run_config(config):
    # Config holds the general information, use in main
    # Steps holds the description of tasks with their parameters
    # Steps: all holds general tasks
    # steps: strain_name holds task for a given strain
    expt_id = config["general"].get("id")
    distributed = config["general"].get("distributed", 0)
    strain_filter = config["general"].get("strain", "")
    root_dir = config["general"].get("directory", "output")
    root_dir = Path(root_dir)

    print("Searching OMERO")
    # Do all initialisation
    with Dataset(int(expt_id)) as conn:
        image_ids = conn.get_images()
        directory = root_dir / conn.unique_name
        if not directory.exists():
            directory.mkdir(parents=True)
            # Download logs to use for metadata
        conn.cache_logs(directory)

    # Modify to the configuration
    config["general"]["directory"] = directory
    # Filter
    image_ids = {k: v for k, v in image_ids.items() if k.startswith(strain_filter)}

    if distributed != 0:  # Gives the number of simultaneous processes
        with Pool(distributed) as p:
            results = p.map(lambda x: create_pipeline(x, **config), image_ids.items())
        return results
    else:  # Sequential
        results = []
        for k, v in image_ids.items():
            r = create_pipeline((k, v), **config)
            results.append(r)


def initialise_logging(log_file: str):
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    for v in logging.Logger.manager.loggerDict.values():
        try:
            if not v.name.startswith(["extraction", "core.io"]):
                v.disabled = True
        except:
            pass


def parse_timing(log_file):
    timings = dict()
    # Open the log file
    with open(log_file, "r") as f:
        # Line by line read
        for line in f.read().splitlines():
            if not line.startswith("DEBUG:root"):
                continue
            words = line.split(":")
            # Only keep lines that include "Timing"
            if "Timing" in words:
                # Split the last two into key, value
                k, v = words[-2:]
                # Dict[key].append(value)
                if k not in timings:
                    timings[k] = []
                timings[k].append(float(v[:-1]))
    return timings


def visualise_timing(timings: dict, save_file: str):
    plt.figure().clear()
    plot_data = {
        x: timings[x]
        for x in timings
        if x.startswith(("Trap", "Writing", "Segmentation", "Extraction"))
    }
    sorted_keys, fixed_data = zip(
        *sorted(plot_data.items(), key=operator.itemgetter(1))
    )
    # Set up the graph parameters
    sns.set(style="whitegrid")
    # Plot the graph
    # sns.stripplot(data=fixed_data, size=1)
    ax = sns.boxplot(data=fixed_data, whis=np.inf, width=0.05)
    ax.set(xlabel="Stage", ylabel="Time (s)", xticklabels=sorted_keys)
    ax.tick_params(axis="x", rotation=90)
    ax.figure.savefig(save_file, bbox_inches="tight", transparent=True)
    return


# strain = "YST_1511_007"
strain = ""
# exp = 18616
# exp = 19232
exp = 19995
# exp = 19993
# exp = 20191
# exp = 19831

with Dataset(exp) as conn:
    imgs = conn.get_images()
    exp_name = conn.unique_name

with Image(list(imgs.values())[0]) as im:
    meta = im.metadata
tps = int(meta["size_t"])

config = dict(
    general=dict(
        id=exp,
        distributed=5,
        tps=tps,
        directory="../data/",
        strain=strain,
        tile_size=96,
    ),
    # general=dict(id=19303, distributed=0, tps=tps, strain=strain, directory="../data/"),
    tiler=dict(),
    baby=dict(tf_version=2),
    earlystop=dict(
        min_tp=300,
        thresh_pos_clogged=0.3,
        thresh_trap_clogged=7,
        ntps_to_eval=5,
    ),
)

# Run
run_config(config)
