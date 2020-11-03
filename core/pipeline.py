"""
Pipeline and chaining elements.
"""
from abc import ABC, abstractmethod
import itertools
from typing import Iterable, List
import logging

import h5py
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from core.utils import PersistentDict
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


def create_keys(expt, strain, timepoints=None, positions=None, exclude=None):
    """
    Create a set of keys for use with the pipeline based on a given experiment
    and an end time point.

    :param expt: The Experiment object on which to work
    :param strain: The name of the strain, which is assumed to be in the
    position names. If it is not, use a list in the `positions` argument
    instead.
    :param timepoints: The number of timepoints to run. If set to None
    (default) will run all of the timepoints in the experiment.
    :param positions: The positions to run. Overrides the `strain` argument.
    :param exclude: The positions to exclude to exclude. Needs to be an
    iterable.
    """
    # TODO: Make it possible to use groups defined in metadata to choose the
    # positions to run.
    # TODO: make it possible to timepoints not from 0 
    if timepoints is None:
        # Run full experiment
        timepoints = expt.shape[1]
    if positions is None:
        # Use strain to try to find the positions
        positions = [p for p in expt.positions if p.startswith(strain)]
    if exclude is not None:
        positions = list(set(positions) - set(exclude))
    return list(itertools.product(positions, range(timepoints)))

class Pipeline:
    """
    A chained set of Pipeline elements connected through pipes.
    """

    def __init__(self, pipeline_steps: Iterable, database: str, store: str,
                 hdf5=True):
        # Setup steps
        self.steps = pipeline_steps
        # Database session
        self.engine = sa.create_engine(database)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(self.engine)
        self.session = Session()
        if hdf5 is True:
            self.store = h5py.File(store, 'r+') # TODO open with r+, w or a?
        else:
            self.store = PersistentDict(store)

    def run_step(self, keys):
        """
        Run a single step of the pipeline on what is described by keys.

        :param keys: list of tuples where each item is a (position,
        timepoint) combination on which to run the pipeline

        :return: Returns the keys which have been effectively processed.
        """
        try:
            for pipeline_step in self.steps:
                keys = pipeline_step.run(keys, session=self.session,
                                     store=self.store)
                self.session.commit()
            return keys
        except Exception as e:
            self.store.close()
            # close session?
            raise e

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
