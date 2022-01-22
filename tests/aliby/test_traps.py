import unittest
import numpy as np

from aliby.tile.traps import identify_trap_locations


class TestCase(unittest.TestCase):
    def setUp(self):
        self.trap_size = 5
        self.tile_size = 9
        assert self.trap_size % 2
        assert self.tile_size % 2
        self.img_size = 16
        self.data = np.pad(
            np.ones((self.trap_size, self.trap_size)),
            (self.img_size - self.tile_size) // 2,
            mode="constant",
        )
        self.template = np.pad(
            np.ones((self.trap_size, self.trap_size)),
            (self.tile_size - self.trap_size) // 2,
            mode="constant",
        )
        self.expected_location = int(
            (np.ceil((self.img_size - self.tile_size + self.trap_size) / 2) - 1)
        )

    def test_identify_trap_locations(self):
        coords = identify_trap_locations(
            self.data,
            self.template,
            optimize_scale=False,
            downscale=1,
        )
        self.assertEqual(len(coords), 1)
        self.assertEqual(
            coords[0].tolist(),
            [self.expected_location, self.expected_location],
        )


class TestMultipleCase(TestCase):
    def setUp(self):
        self.nrows = 4
        self.ncols = 4
        super().setUp()
        row = np.concatenate([self.data for i in range(self.ncols)])
        self.data = np.concatenate([row for i in range(self.nrows)], axis=1)

    def test_identify_trap_locations(self):
        coords = identify_trap_locations(
            self.data,
            self.template,
            optimize_scale=False,
            downscale=1,
        )
        self.expected_locations = set(
            [
                (
                    self.expected_location + i * (self.img_size - self.trap_size),
                    self.expected_location + j * (self.img_size - self.trap_size),
                )
                for i in range(self.nrows)
                for j in range(self.ncols)
            ]
        )
        ntraps = self.nrows * self.ncols
        self.assertEqual(len(coords), ntraps)
        self.assertEqual(
            ntraps,
            len(self.expected_locations.intersection([tuple(x) for x in coords])),
        )


if __name__ == "__main__":
    unittest.main()
