"""
Testing suite for connection to OMERO
"""
# TODO use omero.gateway.scripts.testdb_create module
from __future__ import print_function

import sys
print(sys.path)

import unittest
import json

import omero_py
from core.connect import Database, Dataset

class TestConnections(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('config.json', 'r') as fd:
            config = json.load(fd)
        cls._config = config
        cls._db = Database(cls._config['user'], 
                                   cls._config['password'], 
                                   cls._config['host'], 
                                   cls._config['port'])

    @classmethod
    def tearDownClass(cls):
        cls._db.disconnect()
    
    def testConnection(self):
        self.assertTrue(self._db.connect())

    def testUser(self):
        self.assertEquals(self._db.user['Username'], self._config['user'])

    def testDataset(self):
        dataset = self._db.getDatasets(1)
        self.assertTrue(dataset is not None)
        # FIXME cannot do this as it is instanciated anew.
        # self._ds = dataset

    def testImage(self):
        dataset = self._db.getDatasets(1)
        self.assertTrue(dataset is not None)
       
        #FIXME return single dataset
        image = dataset[0].getImages(1)
        self.assertTrue(image is not None)
        # FIXME cannot do this as it is instanciated anew.
        self._im = image
        #FIXME return single dataset
        # TODO put this in the core module, not in the tests
        pixels = image[0].image.getPrimaryPixels()
        array = pixels.getPlane()
        self.assertTrue(array is not None)
        print(array.shape)
        print(array[0:10, 0:10])

if __name__ == "__main__":
    unittest.main()
