from pathlib import Path
from time import perf_counter
import logging

from core.experiment import MetaData
from pathos.multiprocessing import Pool
from multiprocessing import set_start_method
import numpy as np
import pandas as pd
from postprocessor.core.processor import PostProcessorParameters, PostProcessor


from tqdm import tqdm
import traceback

from baby.brain import BabyBrain

from core.io.omero import Dataset, Image
from core.haystack import initialise_tf
from core.baby_client import DummyRunner
from core.segment import Tiler
from core.io.writer import TilerWriter, BabyWriter, BabyFolded

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
            run_config = {"with_edgemasks": True, "assign_mothers": True}
            for i in tqdm(range(0, tps), desc=image.name):
                trap_info = tiler.run_tp(i)
                writer.write(trap_info, overwrite=[])
                seg = runner.run_tp(i, **run_config)
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


def create_pipeline(image_id, **config):
    name, image_id = image_id
    general_config = config.get('general', None)
    assert general_config is not None
    session = None
    try:
        directory = general_config.get('directory', '')
        # Run metadata first
        with Image(image_id) as image:
            linear_file = f'{directory}/{image.name}.h5'
            folded_file = f'{directory}/{image.name}_folded.h5'
            for f in [linear_file, folded_file]:
                meta = MetaData(directory, f)
                # Run metadata first so it can be used by other processes
                meta.run()
            tiler = Tiler(image.data, image.metadata)
            writer = TilerWriter(linear_file)
            writer2 = TilerWriter(folded_file)
            baby_config = config.get('baby', None)
            assert baby_config is not None  # TODO add defaults
            tf_version = baby_config.get('tf_version', 1)
            session = initialise_tf(tf_version)
            runner = DummyRunner(tiler)
            bwriter = BabyWriter(linear_file)
            bwriter2 = BabyFolded(folded_file)
            # FIXME testing here the extraction
            params = Parameters(**get_params("batgirl_fast"))
            ext = Extractor.from_object(params,
                                        store=linear_file,
                                        object=tiler)
            ext2 = Extractor.from_object(params,
                                        store=folded_file,
                                        object=tiler)
            # RUN
            tps = general_config.get('tps', 0)
            for i in tqdm(range(0, tps), desc=image.name):
                t = perf_counter()
                trap_info = tiler.run_tp(i)
                logging.debug(f'Timing:Trap:{perf_counter() - t}s')
                t = perf_counter()
                writer.write(trap_info, overwrite=[])
                writer2.write(trap_info, overwrite=[])
                logging.debug(f'Timing:Writing-trap:{perf_counter() - t}s')
                t = perf_counter()
                seg = runner.run_tp(i)
                logging.debug(f'Timing:Segmentation:{perf_counter() - t}s')
                t = perf_counter()
                bwriter.write(seg, overwrite=['mother_assign'])
                logging.debug(f'Timing:Writing-baby-linear:{perf_counter() - t}s')
                t = perf_counter()
                bwriter2.write(seg, overwrite=['mother_assign'])
                logging.debug(f'Timing:Writing-baby-folded:{perf_counter() - t}s')
                t = perf_counter()
                ext.extract_pos(tps=[i])
                logging.debug(f'Timing:Extraction_linear:{perf_counter() - t}s')
                t = perf_counter()
                ext2.extract_pos(tps=[i])
                logging.debug(f'Timing:Extraction_folded:{perf_counter() - t}s')
            # Run post processing
            params = PostProcessorParameters.default()
            post_process(linear_file, params)
            post_process(folded_file, params)
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


def post_process(filepath, params):
    pp = PostProcessor(filepath, params)
    tmp = pp.run()
    return tmp


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

    params = PostProcessorParameters.default()
    if distributed != 0:  # Gives the number of simultaneous processes
        with Pool(distributed) as p:
            results = p.map(lambda x: create_pipeline(x, **config), image_ids.items())
        return results
    else:  # Sequential
        results = []
        for k, v in image_ids.items():
            r = create_pipeline((k, v), **config)
            results.append(r)

import logging
def initialise_logging(log_file: str):
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    for v in logging.Logger.manager.loggerDict.values():
        try:
            if not v.name.startswith(['extraction', 'core.io']):
                v.disabled = True
        except:
            pass

def parse_timing(log_file):
    import matplotlib.pyplot as plt
    import numpy as np

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

import matplotlib.pyplot as plt
import seaborn as sns
import operator
def visualise_timing(timings: dict, save_file: str):
    plt.figure().clear()
    plot_data = {x: timings[x] for x in timings if x.startswith(('Trap', 'Writing', 'Segmentation', 'Extraction'))}
    sorted_keys, fixed_data = zip(*sorted(plot_data.items(), key=operator.itemgetter(1)))
    #Set up the graph parameters
    sns.set(style='whitegrid')
    #Plot the graph
    sns.stripplot(data=fixed_data, size=1)
    ax = sns.boxplot(data=fixed_data, whis=np.inf, width=.05)
    ax.set(xlabel="Stage", ylabel="Time (s)", xticklabels=sorted_keys)
    ax.tick_params(axis='x', rotation=90);
    ax.figure.savefig(save_file, bbox_inches='tight', transparent=True)
    return

from pathlib import Path
def visualise_sizes(directory, save_fig):
    plt.figure().clear()
    directory = Path(directory)
    sizes = [(f.name, f.stat().st_size) for f in directory.iterdir() if f.name.endswith('h5')]
    data = pd.DataFrame(sizes, columns=['Name', 'Bytes'])
    data['Folded'] = data['Name'].str.contains('folded')
    sns.set(palette='pastel', style='whitegrid')
    sns.stripplot(data=data, x='Folded', y='Bytes')
    ax = sns.boxplot(data=data, x='Folded', y='Bytes')
    sns.despine(offset=10, trim=True)
    ax.figure.savefig(save_fig, bbox_inches='tight', transparent=True)
    return

if __name__ == "__main__":
    #set_start_method("spawn")
    config = dict(
        general=dict(
            id=19303,
            distributed=5,
            tps=390,
            strain='Hxt1',
            directory='../data/'
        ),
        baby=dict(tf_version=2)
    )
    directory = '../data/2tozero_Hxts_02'
    log_file = '../data/2tozero_Hxts_02/folding_test.log'
    save_timings = '../data/2tozero_Hxts_02/timing_folding_test.pdf'
    save_sizes = '../data/2tozero_Hxts_02/sizes_folding_test.pdf'
    initialise_logging(log_file)
    # Run test
    run_config(config)
    # Get timing results
    timing = parse_timing(log_file)
    # Visualise timings and save
    visualise_timing(timing, save_timings)
    # Visualse 
    visualise_sizes(directory, save_sizes)

