"""
Pipeline and chaining elements.
"""
from abc import ABC, abstractmethod
import itertools
from typing import List

import pandas as pd

from core.experiment import ExperimentOMERO, ExperimentLocal


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


def create_keys(expt, strain='', timepoints=None, positions=None, exclude=None):
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
    if positions is None:
        # Use strain to try to find the positions
        positions = [p for p in expt.positions if p.startswith(strain)]
    if exclude is not None:
        positions = list(set(positions) - set(exclude))
    # Get the correct time points for each position
    run_tps = dict()
    for pos in positions:
        # Note that the different posistions can have different shapes
        position = expt.get_position(pos)
        n_tps = position.shape[1]
        if timepoints is None:
            # Run full experiment
            run_tps[pos] = list(range(0, n_tps))
        else:
            if max(timepoints) > n_tps:
                raise ValueError(f'Position {pos} only has {n_tps} time '
                                 f'points but you asked for {timepoints}')
            # If all the time points are available for that position
            run_tps[pos] = timepoints
    keys = [(pos, tp) for pos in run_tps for tp in run_tps[pos]]
    return keys

class Pipeline:
    """
    A chained set of Pipeline elements connected through pipes.
    """
    def __init__(self, config_file):
        config = parse_config(config_file)
        self.store = config['directory']
        self.experiment = experiment_factory(config)

import yaml
def parse_config(yaml_file):
    with open(yaml_file) as fd:
        config = yaml.safe_load(fd)
    return config

def experiment_factory(config):
    if 'experiment' not in config:
        return None
    expt_conf = config['experiment']
    # Choose local or remote
    try:
        save_dir = expt_conf['local']
        if 'remote' in expt_conf:
            omero = expt_conf['remote']
            omero['save_dir'] = save_dir
            expt = ExperimentOMERO(**omero)
        else:
            expt = ExperimentLocal(save_dir, expt_conf.get('finished', True))
    except Exception as e:
        raise Exception('Configuration incorrect: experiment needs a local '
                        'directory') from e
    return expt, expt_conf.get('run', None)


