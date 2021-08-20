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

    """

    def __init__(
        self,
        targets={},
        parameters={},
        outpaths={},
    ):
        self.targets: Dict = targets
        self.parameters: Dict = parameters
        self.outpaths: Dict = outpaths

    def __getitem__(self, item):
        return getattr(self, item)

    @classmethod
    def default(cls, kind=None):
        if kind == "defaults" or kind == None:
            return cls(
                targets={
                    "prepost": {
                        "merger": "/extraction/general/None/area",
                        "picker": ["/extraction/general/None/area"],
                    },
                    "processes": {
                        "dsignal": ["/extraction/general/None/area"],
                        # "savgol": ["/extraction/general/None/area"],
                    },
                },
                parameters={
                    "prepost": {
                        "merger": mergerParameters.default(),
                        "picker": pickerParameters.default(),
                    }
                },
                outpaths={},
            )

    def to_dict(self):
        return {k: _if_dict(v) for k, v in self.__dict__.items()}


class PostProcessor:
    def __init__(self, filename, parameters):
        self.parameters = parameters
        self._filename = filename
        self._signal = Signal(filename)
        self._writer = Writer(filename)

        # self.outpaths = parameters["outpaths"]
        self.merger = merger(parameters["parameters"]["prepost"]["merger"])

        self.picker = picker(
            parameters=parameters["parameters"]["prepost"]["picker"],
            cells=Cells.from_source(filename),
        )
        self.classfun = {
            process: self.get_process(process)
            for process in parameters["targets"]["processes"]
        }
        self.parameters_classfun = {
            process: self.get_parameters(process)
            for process in parameters["targets"]["processes"]
        }
        self.targets = parameters["targets"]

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

    def run_prepost(self):
        """Important processes run before normal post-processing ones"""
        merge_events = self.merger.run(self._signal[self.targets["prepost"]["merger"]])

        with h5py.File(self._filename, "r") as f:
            prev_idchanges = self._signal.get_id_changes()

        changes_history = prev_idchanges + [merge_events]  # + [picks]
        self._writer.write("/id_changes", data=changes_history)
        # self._writer.write(
        #     "/postprocessing/merge_events/",
        #     data=merge_events,
        #     meta={"source": "/cell_info/"},
        # )
        # changes_history += picks
        # picks = self.picker.run(self._signal[self.targets["prepost"]["picker"][0]])
        # self._writer.write()

    def run(self):
        self.run_prepost()

        for process, datasets in self.targets["processes"].items():
            if process in self.parameters["parameters"].get(
                "processes", {}
            ):  # If we assigned parameters
                parameters = self.parameters_classfun[process](self.parameters[process])

            else:
                parameters = self.parameters_classfun[process].default()

            loaded_process = self.classfun[process](parameters)
            for dataset in datasets:
                if isinstance(dataset, list):  # multisignal process
                    signal = [self._signal[d] for d in dataset]
                elif isinstance(dataset, str):
                    signal = self._signal[dataset]
                else:
                    raise ("Incorrect dataset")

                result = loaded_process.run(signal)

                if process in self.parameters.to_dict()["outpaths"]:
                    outpath = self.parameters.to_dict()["outpaths"][process]
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
                    raise ("Outpath not defined", type(dataset))

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
