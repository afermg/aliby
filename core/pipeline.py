"""
Pipeline and chaining elements.
"""
from abc import ABC, abstractmethod
from typing import Iterable, List

import pandas as pd
import tables as tb

from core.experiment import Experiment, ExperimentLocal, ExperimentOMERO
from core.segment import Tiler
from core.baby_client import BabyClient


class Results:
    """
    Object storing the data from the Pipeline.
    Uses pandas' HDFStore object.

    In addition, it implements:
     - IO functionality (read from file, write to file)

    """
    def __init__(self):
        pass

    def to_store(self):
        pass

    def from_json(self):
        pass


class PipelineStep(ABC):
    @abstractmethod
    def run(self, keys: List[str], store: pd.HDFStore) -> List[str]:
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

    def __init__(self, store: pd.HDFStore, pipeline_steps: Iterable):
        self.store = store
        # Setup steps
        self.steps = pipeline_steps

    def run(self, max_runs=2):
        keys = []
        runs = 0
        while runs <= max_runs and keys is not None:
            # TODO make run functions return None when finished
            for pipeline_step in self.steps:
                keys = pipeline_step.run(keys, self.store)
            runs += 1

# Todo future: could have more flexibility by using pytables directly. At
#  the moment using pandas.HDFstore, does not allow for writing arrays to
#  the file, which would be more convenient for variable-sized data.
class Store:
    """
    Implements an interface to pytables.
    """
    def __init__(self, filename: str, title: str = "Test file"):
        self.h5file = tb.open_file(filename, mode='w', title=title)

    def add_group(self, root: str, name: str, description=""):
        # Todo: infer root from name
        group = self.h5file.create_group(root, name, description)
        return group

    def add_array(self, group, name, values=None, title=""):
        self.h5file.create_array(group, name, values, title)

    def close(self):
        self.h5file.close()




