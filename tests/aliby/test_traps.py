import unittest
import numpy as np

from aliby.tile.traps import identify_trap_locations


class TestCase(unittest.TestCase):
    def setUp(self):
        self.data = np.pad(np.ones((5, 5)), 10, mode="constant")
        self.template = np.pad(np.ones((5, 5)), 2, mode="constant")

    def test_identify_trap_locations(self):
        coords = identify_trap_locations(
            self.data, self.template, optimize_scale=False, downscale=1
        )
        self.assertEqual(len(coords), 1)
        self.assertItemsEqual(coords[0], [12, 12])


if __name__ == "__main__":
    unittest.main()
