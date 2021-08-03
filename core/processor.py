from pydoc import locate
from typing import List, Dict, Union
import pandas as pd

from postprocessor.core.processes.base import ParametersABC
from postprocessor.core.processes.merger import mergerParameters, merger
from postprocessor.core.processes.picker import pickerParameters, picker
from core.io.writer import Writer
from core.io.signal import Signal

from core.cells import Cells


class PostProcessorParameters(ParametersABC):
    """
    Anthology of parameters used for postprocessing
    :merger:
    :picker: parameters for picker
    :processes: Dict processes:[objectives], 'processes' are defined in ./processes/
        while objectives are relative or absolute paths to datasets. If relative paths the
        post-processed addresses are used.

    #TODO Use cells to fetch updated cell indices
    """

    def __init__(self, merger=None, picker=None, processes=[]):
        self.merger: mergerParameters = merger
        self.picker: pickerParameters = picker
        self.processes: Dict = processes

    def __getitem__(self, item):
        return getattr(self, item)

    @classmethod
    def default(cls, kind=None):
        if kind == "defaults" or kind == None:
            return cls(
                merger=mergerParameters.default(),
                picker=pickerParameters.default(),
                processes={
                    "merger": "/extraction/general/None/area",
                    "picker": "/extraction/general/None/area",
                    "processes": {"dSignal": ["/general/None/area"]},
                },
            )

    def to_dict(self):
        return {k: _if_dict(v) for k, v in self.__dict__.items()}


class PostProcessor:
    def __init__(self, filename, parameters):
        self.parameters = parameters
        self._signal = Signal(filename)
        self._writer = Writer(filename)

        # self.outpaths = parameters["outpaths"]
        self.merger = merger(parameters["merger"])
        self.picker = picker(
            parameters=parameters["picker"], cells=Cells.from_source(filename)
        )
        self.process_dict = {
            process: locate(process) for process in parameters["processes"].keys()
        }
        self.processes = parameters["processes"]

    @staticmethod
    def get_process(process):
        """
        Dynamically import a process class from the 'processes' folder.
        """
        return locate("postprocessor.core.processes." + process + "." + process)

    def run(self):
        new_ids = self.merger.run(self._signal[self.processes["merger"]])
        for name, ids in new_ids.items():
            self._writer.write(ids, "/postprocessing/cell_info/" + name)
        picks = self.picker.run(self._signal[self.processes["picker"][0]])
        for process, datasets in self.processes.values():
            for dataset in datasets:
                if isinstance(dataset, list):  # multisignal process
                    result = self.processes["process"].run(
                        [self._signal[d] for d in dataset]
                    )
                    prefix = "".join(
                        prefix + c[0]
                        for c in takewhile(
                            lambda x: all(x[0] == y for y in x), zip(*dataset)
                        )
                    )
                    outpath = (
                        prefix
                        + "_".join(  # TODO check that it always finishes in '/'
                            [d[len(prefix) :].replace("/", "_") for d in dataset]
                        )
                    )
                elif isinstance(dataset, str):
                    result = self.processes["process"].run(self._signal[dataset])
                    outpath = dataset[1:].replace("/", "_")
                else:
                    raise ("Not appropiate dataset")
                self.writer.write(result, "/postprocessing/" + process + "/" + outpath)


def _if_dict(item):
    if hasattr(item, "to_dict"):
        item = item.to_dict()
    return item
