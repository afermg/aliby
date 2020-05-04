# TODO turn into Unittest test case

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

with open('config.json', 'r') as fd:
    config = json.load(fd)

expt = Experiment.from_source(config['experiment'], config['user'],
                              config['password'], config['host'],
                              config['port'])

print(expt.metadata.channels)
print(expt.metadata.times)
print(expt.metadata.switch_params)
print(expt.metadata.zsections)
print(expt.metadata.positions)

# print(expt.get_hypercube(x=None, y=None, width=None, height=None,
#                         z_positions=[0], channels=[0], timepoints=[0]))
# expt.cache_locally(root_dir='/Users/s1893247/PhD/pipeline-core/data/',
#                   positions=['pos001', 'pos002', 'pos003'],
#                   channels=['Brightfield', 'GFP'],
#                   timepoints=range(3),
#                   z_positions=None)

expt.connection.seppuku()
