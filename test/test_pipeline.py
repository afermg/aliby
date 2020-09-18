import os
import unittest

import sqlalchemy as sa

from core.pipeline import Pipeline
from core.pipeline import ExperimentLocal, Tiler, BabyClient

from database.records import Base


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = '/Users/s1893247/PhD/pipeline-core/data/glclvl_0' \
                   '.1_mig1_msn2_maf1_sfp1_dot6_03'
        raw_expt = ExperimentLocal(root_dir, finished=False)
        tiler = Tiler(raw_expt, finished=False)

        config = {"camera": "evolve",
                  "channel": "brightfield",
                  "zoom": "60x",
                  "n_stacks": "5z"}

        self.sql_db = "sqlite:///test.db"
        self.store = "test.hdf5"


        baby_client = BabyClient(tiler, **config)
        self.pipeline = Pipeline(pipeline_steps=[raw_expt, tiler,
                                                 baby_client],
                                 database=self.sql_db,
                                 store=self.store)

    def test_run(self):
        self.pipeline.run_step([('pos001', 0), ('pos001', 1), ('pos001', 2),
                           ('pos001', 3), ('pos001', 4)])
        self.pipeline.store_to_h5()

    def tearDown(self) -> None:
        # Clear the database
        engine = sa.create_engine(self.sql_db)
        Base.metadata.drop_all(engine)
        # Delete the HDF5 file
        os.remove(self.store)


if __name__ == '__main__':
    unittest.main()
