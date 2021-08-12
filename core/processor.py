import h5py
from typing import List, Dict, Union
from pydoc import locate

import numpy as np
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

    def __init__(
        self,
        merger=None,
        picker=None,
        processes={},
        process_parameters={},
        process_outpaths={},
    ):
        self.merger: mergerParameters = merger
        self.picker: pickerParameters = picker
        self.processes: Dict = processes
        self.process_parameters: Dict = process_parameters
        self.process_outpaths: Dict = process_outpaths

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
                    "picker": ["/extraction/general/None/area"],
                    "processes": {"dsignal": ["/extraction/general/None/area"]},
                    "process_parameters": {},
                    "process_outpaths": {},
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
        self.process_classfun = {
            process: self.get_process(process)
            for process in parameters["processes"]["processes"].keys()
        }
        self.process_parameters = {
            process: self.get_parameters(process)
            for process in parameters["processes"]["processes"]
        }
        self.processes = parameters["processes"]

    @staticmethod
    def get_process(process):
        """
        Dynamically import a process class from the 'processes' folder.
        Assumes process filename and class name are the same
        """
        return locate("postprocessor.core.processes." + process + "." + process)

    @staticmethod
    def get_parameters(process):
        """
        Dynamically import a process class from the 'processes' folder.
        Assumes process filename and class name are the same
        """
        return locate(
            "postprocessor.core.processes." + process + "." + process + "Parameters"
        )

    def run(self):
        new_ids = self.merger.run(self._signal[self.processes["merger"]])
        for name, ids in new_ids.items():
            self._writer.write("/postprocessing/cell_info/" + name, ids)
        picks = self.picker.run(self._signal[self.processes["picker"][0]])
        for process, datasets in self.processes["processes"].items():
            parameters = (
                self.process_parameters[process].from_dict(
                    self.process_parameters[process]
                )
                if process in self.parameters["processes"]["process_parameters"]
                else self.process_parameters[process].default()
            )
            loaded_process = self.process_classfun[process](parameters)
            for dataset in datasets:
                if isinstance(dataset, list):  # multisignal process
                    signal = [self._signal[d] for d in dataset]
                elif isinstance(dataset, str):
                    signal = self._signal[dataset]
                else:
                    raise ("Incorrect dataset")

                result = loaded_process.run(signal)

                if process in self.parameters.to_dict()["process_outpaths"]:
                    outpath = self.parameters.to_dict()["process_outpaths"][process]
                elif isinstance(dataset, list):
                    # If no outpath defined, place the result in the minimum common
                    # branch of all signals used
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
                    outpath = dataset[1:].replace("/", "_")
                else:
                    raise ("Putpath not defined", type(dataset))

                self.write_result(
                    "/postprocessing/" + process + "/" + outpath, result, metadata={}
                )

    def write_result(
        self, path: str, result: Union[List, pd.DataFrame, np.ndarray], metadata: Dict
    ):
        self._writer.write(path, result, meta=metadata)


def _if_dict(item):
    if hasattr(item, "to_dict"):
        item = item.to_dict()
    return item
