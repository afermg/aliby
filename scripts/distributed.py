from pathlib import Path
from time import perf_counter
import logging

from pathos.multiprocessing import Pool
from multiprocessing import set_start_method
import numpy as np
from postprocessor.core.processor import PostProcessorParameters, PostProcessor

set_start_method("spawn")

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
        brain = BabyBrain(session=session, **DummyRunner.model_config)
        with Image(image_id) as image:
            print(f'Getting data for {image.name}')
            tiler = Tiler(image.data, image.metadata, image.name)
            writer = TilerWriter(f'../data/test2/{image.name}.h5')
            runner = DummyRunner(tiler, brain)
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
        with Image(image_id) as image:
            tiler_config = config.get('tiler', None)
            assert tiler_config is not None  # TODO add defaults
            tiler = Tiler(image.data, image.metadata)
            writer = TilerWriter(f'{directory}/{image.name}.h5')
            baby_config = config.get('baby', None)
            assert baby_config is not None  # TODO add defaults
            tf_version = baby_config.get('tf_version', 1)
            session = initialise_tf(tf_version)
            brain = BabyBrain(session=session, **DummyRunner.model_config)
            runner = DummyRunner(tiler, brain)
            bwriter = BabyWriter(f'{directory}/{image.name}.h5')
            bwriter2 = BabyFolded(f'{directory}/{image.name}.h5')
            # FIXME testing here the extraction
            params = Parameters(**get_params("batgirl_fast"))
            ext = Extractor.from_object(params,
                                        store=f'{directory}/{image.name}.h5',
                                        object=tiler)
            # RUN
            run_config = baby_config.get('run', dict())
            tps = general_config.get('tps', 0)
            for i in tqdm(range(0, tps), desc=image.name):
                t = perf_counter()
                trap_info = tiler.run_tp(i)
                logging.debug(f'Timing:Trap:{perf_counter() - t}s')
                t = perf_counter()
                writer.write(trap_info, overwrite=[])
                logging.debug(f'Timing:Writing-trap:{perf_counter() - t}s')
                t = perf_counter()
                seg = runner.run_tp(i, **run_config)
                logging.debug(f'Timing:Segmentation:{perf_counter() - t}s')
                t = perf_counter()
                bwriter.write(seg, overwrite=['mother_assign'])
                logging.debug(f'Timing:Writing-baby-linear:{perf_counter() - t}s')
                t = perf_counter()
                bwriter2.write(seg, overwrite=['mother_assign'])
                logging.debug(f'Timing:Writing-baby-folded:{perf_counter() - t}s')
                t = perf_counter()
                ext.extract_pos(tps=[i])
                logging.debug(f'Timing:Extraction:{perf_counter() - t}s')
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

    # Modify to the configuration
    config['general']['directory'] = directory
    # Filter
    image_ids = {k: v for k, v in image_ids.items() if k.startswith(
        strain_filter)}

    if distributed != 0:  # Gives the number of simultaneous processes
        with Pool(distributed) as p:
            results = p.map(lambda x: create_pipeline(x, **config), image_ids)
        return results
    else:  # Sequential
        results = []
        for k, v in image_ids.items():
            r = create_pipeline((k, v), **config)
            results.append(r)

    # Post process!
    params = PostProcessorParameters.default()
    if distributed != 0:
        with Pool(distributed) as p:
            results = p.map(lambda pos_name, _: post_process(directory /
                                                             f'{pos_name}.h5',
                                                             params),
                            image_ids)
    else:
        for pos_name in image_ids:
            tmp = post_process(directory / f'{pos_name}.h5', params)
    return


if __name__ == "__main__":
    import logging
    log_file = '../data/2tozero_Hxts_02/issues.log'
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    for v in logging.Logger.manager.loggerDict.values():
        try:
            if not v.name.startswith(['extraction', 'core.io']):
                v.disabled = True
        except:
            pass
    config = dict(
        general=dict(
            id=19303,
            distributed=0,
            tps=390,
            strain='Hxt1_025',
            directory='../data/'
        ),
        tiler=dict(),
        baby=dict(
            tf_version=2,
            run=dict(
                with_edgemasks=True,
                assign_mothers=True
            )
        )
    )

    run_config(config)
