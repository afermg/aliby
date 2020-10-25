"""
Pipeline and chaining elements.
"""
from abc import ABC, abstractmethod
from typing import Iterable, List
import logging

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from database.records import Base
from core.experiment import ExperimentLocal
from core.segment import Tiler
from core.baby_client import BabyClient


class PipelineStep(ABC):
    @abstractmethod
    def run(self, keys: List[str], store: pd.HDFStore=None, session=None) -> \
            List[str]:
        """
        Abstract run method, when implemented by subclasses, runs analysis
        on the keys and saves results in store.
        :param keys: list of keys on which to run analysis
        :return: A set of keys now available for anlaysis for the next step.
        """
        return keys


class Pipeline:
    """
    A chained set of Pipeline elements connected through pipes.
    """

    def __init__(self, pipeline_steps: Iterable, database: str, store: str):
        # Setup steps
        self.steps = pipeline_steps
        # Database session
        self.engine = sa.create_engine(database)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(self.engine)
        self.session = Session()
        self.store = store

    def run_step(self, keys):
        for pipeline_step in self.steps:
            keys = pipeline_step.run(keys, session=self.session,
                                     store=self.store)
            self.session.commit()
        return keys

    def run(self, keys, max_runs=2):
        # Todo : create  a pipeline step that checks the data
        runs = 0
        while runs <= max_runs and keys is not None:
            # TODO make run functions return None when finished
            for pipeline_step in self.steps:
                keys = pipeline_step.run(keys)
            runs += 1

    def store_to_h5(self):
        store = pd.HDFStore(self.store)
        for table in ['positions', 'traps', 'drifts', 'cells', 'cell_info']:
            store[table] = pd.read_sql_table(table, self.engine)
        store.close()
        return
