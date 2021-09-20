from pathlib import Path
import json
from time import perf_counter
import logging

from core.experiment import MetaData
from pathos.multiprocessing import Pool
from multiprocessing import set_start_method
import numpy as np
from postprocessor.core.processor import PostProcessorParameters, PostProcessor

#set_start_method("spawn")

from tqdm import tqdm
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import operator

from baby.brain import BabyBrain

from core.io.omero import Dataset, Image
from core.haystack import initialise_tf
from core.baby_client import DummyRunner
from core.segment import Tiler
from core.io.writer import TilerWriter, BabyWriter
from core.utils import timed

from extraction.core.extractor import Extractor
from extraction.core.parameters import Parameters
from extraction.core.functions.defaults import get_params

import warnings
# TODO This is for extraction issue #9, remove when fixed
warnings.simplefilter('ignore', RuntimeWarning)


def pipeline(image_id, tps=10, tf_version=2):
    name, image_id = image_id
    try:
        # Initialise tensorflow
        session = initialise_tf(tf_version)
        with Image(image_id) as image:
            print(f'Getting data for {image.name}')
            tiler = Tiler(image.data, image.metadata, image.name)
            writer = TilerWriter(f'../data/test2/{image.name}.h5')
            runner = DummyRunner(tiler)
            bwriter = BabyWriter(f'../data/test2/{image.name}.h5')
            for i in tqdm(range(0, tps), desc=image.name):
                trap_info = tiler.run_tp(i)
                writer.write(trap_info, overwrite=[])
                seg = runner.run_tp(i)
                bwriter.write(seg, overwrite=['mother_assign'])
            return True
    except Exception as e:  # bug in the trap getting
        print(f'Caught exception in worker thread (x = {name}):')
        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()
        print()
        raise e
    finally:
        # Close session
        if session:
            session.close()


@timed('Position')
def create_pipeline(image_id, **config):
    name, image_id = image_id
    general_config = config.get('general', None)
    assert general_config is not None
    session = None
    try:
        directory = general_config.get('directory', '')
        with Image(image_id) as image:
            filename = f'{directory}/{image.name}.h5'
            # Run metadata first
            meta = MetaData(directory, filename)
            meta.run()
            tiler = Tiler(image.data, image.metadata)
            writer = TilerWriter(filename)
            baby_config = config.get('baby', None)
            assert baby_config is not None  # TODO add defaults
            tf_version = baby_config.get('tf_version', 1)
            session = initialise_tf(tf_version)
            runner = DummyRunner(tiler)
            bwriter = BabyWriter(filename)
            # FIXME testing here the extraction
            params = Parameters(**get_params("batgirl_fast"))
            ext = Extractor.from_object(params,
                                        store=filename,
                                        object=tiler)
            # RUN
            tps = general_config.get('tps', 0)
            for i in tqdm(range(0, tps), desc=image.name):
                t = perf_counter()
                trap_info = tiler.run_tp(i)
                logging.debug(f'Timing:Trap:{perf_counter() - t}s')
                t = perf_counter()
                writer.write(trap_info, overwrite=[])
                logging.debug(f'Timing:Writing-trap:{perf_counter() - t}s')
                t = perf_counter()
                seg = runner.run_tp(i)
                logging.debug(f'Timing:Segmentation:{perf_counter() - t}s')
                t = perf_counter()
                bwriter.write(seg, overwrite=['mother_assign'])
                logging.debug(f'Timing:Writing-baby:{perf_counter() - t}s')
                t = perf_counter()
                ext.extract_pos(tps=[i])
                logging.debug(f'Timing:Extraction:{perf_counter() - t}s')
            # Run post processing
            post_proc_params = PostProcessorParameters.default()
            post_process(filename, post_proc_params)
            return True
    except Exception as e:  # bug in the trap getting
        print(f'Caught exception in worker thread (x = {name}):')
        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()
        print()
        raise e
    finally:
        if session:
            session.close()

@timed('Post-processing')
def post_process(filepath, params):
    pp = PostProcessor(filepath, params)
    tmp = pp.run()
    return tmp

@timed('Pipeline')
def run_config(config):
    # Config holds the general information, use in main
    # Steps holds the description of tasks with their parameters
    # Steps: all holds general tasks
    # steps: strain_name holds task for a given strain
    expt_id = config['general'].get('id')
    distributed = config['general'].get('distributed', 0)
    strain_filter = config['general'].get('strain', '')
    root_dir = config['general'].get('directory', 'output')
    root_dir = Path(root_dir)

    print('Searching OMERO')
    # Do all initialisation
    with Dataset(int(expt_id)) as conn:
        image_ids = conn.get_images()
        directory = root_dir / conn.name
        if not directory.exists():
            directory.mkdir(parents=True)
            # Download logs to use for metadata
        conn.cache_logs(directory)

    # Modify to the configuration
    config['general']['directory'] = directory
    # Filter
    image_ids = {k: v for k, v in image_ids.items() if k.startswith(
        strain_filter)}

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
            if not v.name.startswith(['extraction', 'core.io']):
                v.disabled = True
        except:
            pass
        
        
def parse_timing(log_file):
    timings = dict()
    # Open the log file
    with open(log_file, 'r') as f:
        # Line by line read
        for line in f.read().splitlines():
            if not line.startswith('DEBUG:root'):
                continue
            words = line.split(':')
            # Only keep lines that include "Timing"
            if 'Timing' in words:
                # Split the last two into key, value
                k,v = words[-2:]
                # Dict[key].append(value)
                if k not in timings:
                    timings[k] = []
                timings[k].append(float(v[:-1]))
    return timings


def visualise_timing(timings: dict, save_file: str):
    plt.figure().clear()
    plot_data = {x: timings[x] for x in timings if x.startswith(('Trap', 'Writing', 'Segmentation', 'Extraction'))}
    sorted_keys, fixed_data = zip(*sorted(plot_data.items(), key=operator.itemgetter(1)))
    #Set up the graph parameters
    sns.set(style='whitegrid')
    #Plot the graph
    #sns.stripplot(data=fixed_data, size=1)
    ax = sns.boxplot(data=fixed_data, whis=np.inf, width=.05)
    ax.set(xlabel="Stage", ylabel="Time (s)", xticklabels=sorted_keys)
    ax.tick_params(axis='x', rotation=90);
    ax.figure.savefig(save_file, bbox_inches='tight', transparent=True)
    return


if __name__ == "__main__":
    strain = 'Vph1'
    tps =390
    config = dict(
        general=dict(
            id=19303,
            distributed=5,
            tps=tps,
            strain=strain,
            directory='../data/'
        ),
        tiler=dict(),
        baby=dict(tf_version=2)
    )
    log_file = '../data/2tozero_Hxts_02/issues.log'
    initialise_logging(log_file)
    save_timings = f"../data/2tozero_Hxts_02/timings_{strain}_{tps}.pdf"
    timings_file = f"../data/2tozero_Hxts_02/timings_{strain}_{tps}.json"
    # Run
    #run_config(config)
    # Get timing results
    timing = parse_timing(log_file)
    # Visualise timings and save
    visualise_timing(timing, save_timings)
    # Dump the rest to json
    with open(timings_file, 'w') as fd: 
        json.dump(timing, fd)
