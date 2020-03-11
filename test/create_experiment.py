import os
import sys 
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import json

with open('config.json', 'r') as fd:
    config = json.load(fd)

from core.experiment import ExperimentOMERO

expt = ExperimentOMERO(10421, config['user'], config['password'], config['host'],
        config['port'])

print(expt.metadata.channels)
print(expt.metadata.times)
print(expt.metadata.switch_params)
print(expt.metadata.zsections)
print(expt.metadata.positions)

print(expt.get_hypercube(x=None, y=None, width=None, height=None,
                         z_positions=[0], channels=[0], timepoints=[0]))


expt.connection.seppuku()
