import argparse 
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import json
import logging
from logging.handlers import RotatingFileHandler

from core.experiment import Experiment
logger = logging.getLogger('core')
logger.handlers = []
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.WARNING)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

file_handler = RotatingFileHandler(filename='test.log',
                                   maxBytes=1e5,
                                   backupCount=1)

file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s '
                                   '- %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.debug('Set up the loggers as test.')

parser = argparse.ArgumentParser(description='Load experiment from the database.')
parser.add_argument('--config', dest='config_file', type=str)
parser.add_argument('--id', type=int)

args = parser.parse_args()

with open(args.config_file, 'r') as fd:
    config = json.load(fd)

if not args.id:
    expt_id = config['experiment']
else:
    expt_id = args.id

try:
    expt = Experiment.from_source(expt_id, config['user'],
                              config['password'], config['host'],
                              config['port'], save_dir='data/')
    print(expt.name)
    print(expt.metadata.channels)
    print(expt.metadata.times)
    print(expt.metadata.switch_params)
    print(expt.metadata.zsections)
    print(expt.metadata.positions)
finally:
    expt.connection.seppuku()
