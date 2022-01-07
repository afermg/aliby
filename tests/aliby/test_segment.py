import unittest
import numpy as np

from aliby.tile.tiler import align_timelapse_images


class TestCase(unittest.TestCase):
    def setUp(self):
        self.data = np.ones((1, 3, 5, 5, 5))

    def test_align_timelapse_images(self):
        drift, references = align_timelapse_images(self.data)
        self.assertEqual(references, [0])
        self.assertItemsEqual(drift.flatten(), np.zeros_like(drift.flatten()))


if __name__ == "__main__":
    unittest.main()
