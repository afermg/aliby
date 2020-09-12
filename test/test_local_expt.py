import logging
from logging.handlers import RotatingFileHandler
import unittest
from pathlib import Path

from core.experiment import Experiment

## LOGGING
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

data_directory = Path(__file__).parent.parent / 'data/'
root_directory = data_directory / 'glclvl_0.1_mig1_msn2_maf1_sfp1_dot6_03'

class TestCase(unittest.TestCase):
    def setUp(self):
        self.expt = Experiment.from_source(root_directory, finished=True)

    def test_experiment_shape(self):
        print("C: {}, T: {}, X: {}, Y: {}, Z: {}".format(*self.expt.shape))
        self.assertEqual(len(self.expt.shape), 5)

    def test_experiment_slicing(self):
        test_slice = self.expt[(0, 2), 0:3, :100, 100:200, 0:5:2]
        self.assertTupleEqual(test_slice.shape, (2, 3, 100, 100, 3))

    def test_experiment_metadata(self):
        self.assertTrue('positions' in self.expt.metadata.keys())
        self.assertTrue('channels' in self.expt.metadata.keys())
        self.assertTrue('time_settings' in self.expt.metadata.keys())
        self.assertTrue('zsectioning' in self.expt.metadata.keys())



if __name__ == "__main__":
    unittest.main()


