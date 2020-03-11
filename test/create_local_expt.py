import os
import sys 
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
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

if __name__ == "__main__":
    #root_directory = '/Users/s1893247/PhD/pipeline-core/data
    # /sga_glc0_1_Mig1Nhp_Maf1Nhp_Msn2Maf1_Mig1Mig1_Msn2Dot6_05'

    root_directory = '/Users/s1893247/PhD/omero_connect_demo/test_data'
    expt = Experiment.from_source(root_directory)

    print(expt.metadata.channels)
    print(expt.metadata.times)
    print(expt.metadata.switch_params)
    print(expt.metadata.zsections)
    print(expt.metadata.positions)


    print(expt.current_position.image_mapper.keys())
    print(map(lambda x: x.name, expt.current_position.image_mapper['GFP'][0]))

    print(expt.get_hypercube(x=0, y=0, width=100, height=None,
                             z_positions=[0, 2, 4], channels=[0, 1],
                             timepoints=[0]).shape)

    expt.current_position = expt.positions[-1]
    print(map(lambda x: x.name, expt.current_position.image_mapper['GFP'][0]))
