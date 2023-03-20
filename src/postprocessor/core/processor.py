import logging
import typing as t
from itertools import takewhile
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from agora.abc import ParametersABC, ProcessABC
from agora.io.cells import Cells
from agora.io.signal import Signal
from agora.io.writer import Writer
from agora.utils.indexing import (
    _3d_index_to_2d,
    _assoc_indices_to_3d,
)
from agora.utils.merge import merge_association
from agora.utils.kymograph import get_index_as_np
from postprocessor.core.abc import get_parameters, get_process
from postprocessor.core.lineageprocess import (
    LineageProcess,
    LineageProcessParameters,
)
from postprocessor.core.reshapers.merger import Merger, MergerParameters
from postprocessor.core.reshapers.picker import Picker, PickerParameters


class PostProcessorParameters(ParametersABC):
    """
    Anthology of parameters used for postprocessing
    :merger:
    :picker: parameters for picker
    :processes: list processes:[objectives], 'processes' are defined in ./processes/
        while objectives are relative or absolute paths to datasets. If relative paths the
        post-processed addresses are used. The order of processes matters.

    """

    def __init__(
        self,
        targets={},
        param_sets={},
        outpaths={},
    ):
        self.targets: Dict = targets
        self.param_sets: Dict = param_sets
        self.outpaths: Dict = outpaths

    def __getitem__(self, item):
        return getattr(self, item)

    @classmethod
    def default(cls, kind=[]):
        """Sequential postprocesses to be operated"""
        targets = {
            "prepost": {
                "merger": "/extraction/general/None/area",
                "picker": ["/extraction/general/None/area"],
            },
            "processes": [
                [
                    "buddings",
                    [
                        "/extraction/general/None/volume",
                    ],
                ],
                [
                    "dsignal",
                    [
                        "/extraction/general/None/volume",
                    ],
                ],
                [
                    "bud_metric",
                    [
                        "/extraction/general/None/volume",
                    ],
                ],
                [
                    "dsignal",
                    [
                        "/postprocessing/bud_metric/extraction_general_None_volume",
                    ],
                ],
            ],
        }
        param_sets = {
            "prepost": {
                "merger": MergerParameters.default(),
                "picker": PickerParameters.default(),
            }
        }
        outpaths = {}
        outpaths["aggregate"] = "/postprocessing/experiment_wide/aggregated/"

        if "ph_batman" in kind:
            targets["processes"]["dsignal"].append(
                [
                    "/extraction/em_ratio/np_max/mean",
                    "/extraction/em_ratio/np_max/median",
                    "/extraction/em_ratio_bgsub/np_max/mean",
                    "/extraction/em_ratio_bgsub/np_max/median",
                ]
            )
            targets["processes"]["aggregate"].append(
                [
                    [
                        "/extraction/em_ratio/np_max/mean",
                        "/extraction/em_ratio/np_max/median",
                        "/extraction/em_ratio_bgsub/np_max/mean",
                        "/extraction/em_ratio_bgsub/np_max/median",
                        "/extraction/gsum/np_max/median",
                        "/extraction/gsum/np_max/mean",
                    ]
                ],
            )

        return cls(targets=targets, param_sets=param_sets, outpaths=outpaths)


class PostProcessor(ProcessABC):
    def __init__(self, filename, parameters):
        super().__init__(parameters)
        self._filename = filename
        self._signal = Signal(filename)
        self._writer = Writer(filename)

        dicted_params = {
            i: parameters["param_sets"]["prepost"][i]
            for i in ["merger", "picker"]
        }

        for k in dicted_params.keys():
            if not isinstance(dicted_params[k], dict):
                dicted_params[k] = dicted_params[k].to_dict()

        self.merger = Merger(
            MergerParameters.from_dict(dicted_params["merger"])
        )

        self.picker = Picker(
            PickerParameters.from_dict(dicted_params["picker"]),
            cells=Cells.from_source(filename),
        )
        self.classfun = {
            process: get_process(process)
            for process, _ in parameters["targets"]["processes"]
        }
        self.parameters_classfun = {
            process: get_parameters(process)
            for process, _ in parameters["targets"]["processes"]
        }
        self.targets = parameters["targets"]

    def run_prepost(self):
        # TODO Split function
        """Important processes run before normal post-processing ones"""
        record = self._signal.get_raw(self.targets["prepost"]["merger"])
        merges = np.array(self.merger.run(record), dtype=int)

        self._writer.write(
            "modifiers/merges", data=[np.array(x) for x in merges]
        )

        lineage = _assoc_indices_to_3d(self.picker.cells.mothers_daughters)
        lineage_merged = []

        if merges.any():  # Update lineages after merge events

            merged_indices = merge_association(lineage, merges)
            # Remove repeated labels post-merging
            lineage_merged = np.unique(merged_indices, axis=0)

        self.lineage = _3d_index_to_2d(
            lineage_merged if len(lineage_merged) else lineage
        )

        self._writer.write(
            "modifiers/lineage_merged", _3d_index_to_2d(lineage_merged)
        )

        picked_indices = self.picker.run(
            self._signal[self.targets["prepost"]["picker"][0]]
        )
        if picked_indices.any():
            self._writer.write(
                "modifiers/picks",
                data=pd.MultiIndex.from_arrays(
                    picked_indices, names=["trap", "cell_label"]
                ),
                overwrite="overwrite",
            )

    @staticmethod
    def pick_mother(a, b):
        """Update the mother id following this priorities:

        The mother has a lower id
        """
        x = max(a, b)
        if min([a, b]):
            x = [a, b][np.argmin([a, b])]
        return x

    def run(self):
        # TODO Documentation :) + Split
        self.run_prepost()

        for process, datasets in tqdm(self.targets["processes"]):
            if process in self.parameters["param_sets"].get(
                "processes", {}
            ):  # If we assigned parameters
                parameters = self.parameters_classfun[process](
                    self.parameters[process]
                )

            else:
                parameters = self.parameters_classfun[process].default()

            loaded_process = self.classfun[process](parameters)
            if isinstance(parameters, LineageProcessParameters):
                loaded_process.lineage = self.lineage

            for dataset in datasets:
                if isinstance(dataset, list):  # multisignal process
                    signal = [self._signal[d] for d in dataset]
                elif isinstance(dataset, str):
                    signal = self._signal[dataset]
                else:
                    raise Exception("Unavailable record")

                if len(signal) and (
                    not isinstance(loaded_process, LineageProcess)
                    or len(loaded_process.lineage)
                ):
                    result = loaded_process.run(signal)
                else:
                    result = pd.DataFrame(
                        [], columns=signal.columns, index=signal.index
                    )
                    result.columns.names = ["timepoint"]

                if process in self.parameters["outpaths"]:
                    outpath = self.parameters["outpaths"][process]
                elif isinstance(dataset, list):
                    # If no outpath defined, place the result in the minimum common
                    # branch of all signals used
                    prefix = "".join(
                        c[0]
                        for c in takewhile(
                            lambda x: all(x[0] == y for y in x), zip(*dataset)
                        )
                    )
                    outpath = (
                        prefix
                        + "_".join(  # TODO check that it always finishes in '/'
                            [
                                d[len(prefix) :].replace("/", "_")
                                for d in dataset
                            ]
                        )
                    )
                elif isinstance(dataset, str):
                    outpath = dataset[1:].replace("/", "_")
                else:
                    raise ("Outpath not defined", type(dataset))

                if process not in self.parameters["outpaths"]:
                    outpath = "/postprocessing/" + process + "/" + outpath

                if isinstance(result, dict):  # Multiple Signals as output
                    for k, v in result.items():
                        self.write_result(
                            outpath + f"/{k}",
                            v,
                            metadata={},
                        )
                else:
                    self.write_result(
                        outpath,
                        result,
                        metadata={},
                    )

    def write_result(
        self,
        path: str,
        result: Union[List, pd.DataFrame, np.ndarray],
        metadata: Dict,
    ):
        if not result.any().any():
            logging.getLogger("aliby").warning(f"Record {path} is empty")
        self._writer.write(path, result, meta=metadata, overwrite="overwrite")
