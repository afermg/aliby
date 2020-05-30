import unittest
from core.pipeline import Pipeline
from core.pipeline import ExperimentLocal, Tiler, BabyClient

import pandas as pd


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = pd.HDFStore('store.h5')
        root_dir = '/Users/s1893247/PhD/pipeline-core/data/glclvl_0' \
                   '.1_mig1_msn2_maf1_sfp1_dot6_03'
        raw_expt = ExperimentLocal(root_dir)
        tiler = Tiler(raw_expt)

        config = {"camera": "evolve",
                  "channel": "brightfield",
                  "zoom": "60x",
                  "n_stacks": "5z"}

        baby_client = BabyClient(raw_expt, **config)
        self.pipeline = Pipeline(self.store,
                                 pipeline_steps=[raw_expt, tiler, baby_client])

    def test_run(self):
        self.pipeline.run(max_runs=1)

    def tearDown(self) -> None:
        self.store.close()



if __name__ == '__main__':
    unittest.main()
