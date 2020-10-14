import argparse
import itertools

import logging
# Log to file
logger = logging.getLogger('run_pipeline')
hdlr = logging.FileHandler('run_pipeline.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)
# Also send to stdout
logger.addHandler(logging.StreamHandler())


import os


import sqlalchemy as sa
import numpy as np

from core.pipeline import Pipeline
from core.pipeline import ExperimentLocal, Tiler, BabyClient

from database.records import Base


def define_parser():
    parser = argparse.ArgumentParser(description='Run microscopy pipeline')
    parser.add_argument('root_dir', type=str, help='The experiment root directory')
    parser.add_argument('--camera', default="prime95b")
    parser.add_argument('--channel', default="brightfield")
    parser.add_argument('--zoom', default="60x")
    parser.add_argument('--n_stacks', default="5z")
    parser.add_argument('--time', type=int, default=100)
    return parser

def setup(root_dir, config):
    raw_expt = ExperimentLocal(root_dir, finished=False)
    tiler = Tiler(raw_expt, finished=False)

    sql_db = "sqlite:///{}.db".format(raw_expt.exptID)
    store = "{}.hdf5".format(raw_expt.exptID)

    baby_client = BabyClient(tiler, **config)

    pipeline = Pipeline(pipeline_steps=[raw_expt, tiler, baby_client], 
                        database=sql_db, 
                        store=store)
    return pipeline, raw_expt

def run(pipeline, positions, timepoints): 
    for tp_range in timepoints:
        logger.info("Running timepoints: {}".format(tp_range))
        run_step = list(itertools.product(positions, tp_range))
        pipeline.run_step(run_step)
    pipeline.store_to_h5()

def clean_up(exptID, error=False):
    if error:
        os.remove('{}.db'.format(exptID))
        os.remove('{}.hdf5'.format(exptID))
    else: 
        pass

if __name__ == '__main__':
    parser = define_parser()
    args = parser.parse_args()

    config = {"camera" : args.camera,
              "channel" : args.channel, 
              "zoom" : args.zoom, 
              "n_stacks" : args.n_stacks}
    logger.info("Baby configuration: ", config)
    
    logger.info("Setting up pipeline.")
    pipeline, raw_expt = setup(args.root_dir, config)
    positions = raw_expt.positions
   
    # Todo: get the timepoints from the metadata 
    #       or force processing even though the experiment is finished
    tps = args.time
    logger.info("Experiment: {}".format(raw_expt.exptID))
    logger.info("Positions: {}, timepoints: {}".format(len(positions), tps))

    timepoints = np.arange(tps).reshape(12, -1).tolist()
    logger.info("Running pipeline")
    try:
        run(pipeline, positions, timepoints)
    except Exception as e: 
        logger.info("Cleaning up on error")
        clean_up(raw_expt.exptID, error=True)
        raise e

    logger.info("Cleaning up.")
    clean_up(raw_expt.exptID, error=False)

    

