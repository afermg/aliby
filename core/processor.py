from typing import List, Dict, Union
import pandas as pd

from postprocessor.core.processes.base import ParametersABC
from postprocessor.core.processes.merger import MergerParameters, Merger
from postprocessor.core.processes.picker import PickerParameters, Picker
from postprocessor.core.io.writer import Writer
from postprocessor.core.io.signal import Signal

from core.cells import Cells


class PostProParameters(ParametersABC):
    """
    Anthology of parameters used for postprocessing
    :merger:
    :picker: parameters for picker
    :processes: List of processes that can be found in ./processes
    :datasets: Dictionary
    """

    def __init__(self, merger=None, picker=None, processes=[], datasets=None):
        self.merger: MergerParameters = merger
        self.picker: PickerParameters = picker
        self.processes: List = processes

        self.datasets: Dict = datasets

    def __getitem__(self, item):
        return getattr(self, item)

    @classmethod
    def default(cls, kind=None):
        if kind == "defaults" or kind == None:
            return cls(
                merger=MergerParameters.default(),
                picker=PickerParameters.default(),
                datasets={
                    "merger": "/extraction/general/None/area",
                    "picker": "/extraction/general/None/area",
                    "processes": [],
                },
            )


class PostProcessor:
    def __init__(self, fname, parameters, signals):
        self.parameters = parameters
        self._signals = Signals(fname)
        self._writer = Writer(fname)

        self.datasets = parameters["datasets"]
        self.merger = Merger(parameters["merger"])
        self.picker = Picker(
            parameters=parameters["picker"], cell=Cells.from_source(fname)
        )
        self.processes = [
            self.get_process(process) for process in parameters["processes"]
        ]

    def run(self):
        self.merger.run(signals[self.datasets["merger"]])
        self.picker.run(signals[self.datasets["picker"]])
        for process, dataset in zip(self.processes, self.datasets["processes"]):
            process_result = process.run(signals.get_dataset(dataset))
            self.writer.write(process_result, dataset)
