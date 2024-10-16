import os
import unittest
from pathlib import Path

import pytest

from aliby.baby_client import BabyRunner

# from aliby.experiment import ExperimentOMERO, ExperimentLocal
from aliby.tile.tiler import Tiler


@pytest.fixture
def server_user():
    return os.environ.get("SERVER_USER")


@pytest.fixture
def server_password():
    return os.environ.get("SERVER_PASSWORD")


@pytest.mark.skip(reason="Local analysis not yet implemented")
class TestLocal(unittest.TestCase):
    def setUp(self) -> None:
        self.root_dir = "./data/" ".1_mig1_msn2_maf1_sfp1_dot6_03"
        self.raw_expt = ExperimentLocal(self.root_dir, finished=True)
        self.tiler = Tiler(self.raw_expt, finished=False)

        config = {
            "camera": "evolve",
            "channel": "Brightfield",
            "zoom": "60x",
            "n_stacks": "5z",
            "default_image_size": 80,
        }

        self.store = "test.hdf5"
        self.baby_runner = BabyRunner(self.tiler, **config)

    def test_local(self):
        steps = [
            ("pos001", 0),
            ("pos001", 1),
            ("pos001", 2),
            ("pos001", 3),
            ("pos001", 4),
        ]

        trap_store = self.root_dir + "/traps.csv"
        drift_store = self.root_dir + "/drifts.csv"
        baby_store = self.root_dir + "/baby.csv"

        self.raw_expt.run(steps)
        self.tiler.run(steps, trap_store=trap_store, drift_store=drift_store)
        self.baby_runner.run(steps, store=baby_store)

    def tearDown(self) -> None:
        for p in Path(self.root_dir).glob("*.csv"):
            p.unlink()


# FIXME reimplement this being careful with credentials
@pytest.mark.skip(reason="credentials are not present in current repo")
class TestRemote(unittest.TestCase):
    def setUp(self) -> None:
        self.root_dir = "./"
        self.raw_expt = ExperimentOMERO(
            51,
            username="root",
            password="omero-root-password",
            host="localhost",
            save_dir=self.root_dir,
        )
        self.tiler = Tiler(self.raw_expt, finished=False)

        config = {
            "camera": "evolve",
            "channel": "Brightfield",
            "zoom": "60x",
            "n_stacks": "5z",
            "default_image_size": 80,
        }

        self.baby_runner = BabyRunner(self.tiler, **config)

    def test_remote(self):
        steps = [
            ("pos001", 0),
            ("pos001", 1),
            ("pos001", 2),
            ("pos002", 0),
            ("pos002", 1),
        ]

        run_config = {"with_edgemasks": True}

        pos_store = self.root_dir + "/positions.csv"
        trap_store = self.root_dir + "/traps.csv"
        drift_store = self.root_dir + "/drifts.csv"
        baby_store = self.root_dir + "/baby.h5"

        self.raw_expt.run(steps, pos_store)
        self.tiler.run(steps, trap_store=trap_store, drift_store=drift_store)
        self.baby_runner.run(steps, store=baby_store, **run_config)

    def tearDown(self) -> None:
        # rm_tree(self.root_dir)
        pass


def rm_tree(path):
    path = Path(path)
    for child in path.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    path.rmdir()


if __name__ == "__main__":
    unittest.main()
