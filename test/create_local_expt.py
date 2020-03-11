import os
import sys 
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

root_directory = '/Users/s1893247/PhD/omero_connect_demo/test_data'

from core.experiment import ExperimentLocal

expt = ExperimentLocal(root_directory)

print(expt.metadata.channels)
print(expt.metadata.times)
print(expt.metadata.switch_params)
print(expt.metadata.zsections)
print(expt.metadata.positions)

print(expt.current_position.image_mapper.keys())
print(map(lambda x: x.name, expt.current_position.image_mapper['GFP'][0]))

print(expt.get_hypercube(x=0, y=0, width=100, height=None,
                         z_positions=[0, 2, 4], channels=[0, 1], timepoints=[
        0]).shape)


#expt.connection.seppuku()
