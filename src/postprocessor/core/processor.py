import typing as t
from itertools import takewhile

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

    Supply parameters for picker, merger, and processes.
    The order of processes matters.

    'processes' are defined in ./processes/ while objectives are relative or absolute paths to datasets. If relative paths the post-processed addresses are used.
    """

    def __init__(
        self,
        targets: t.Dict = {},
        param_sets: t.Dict = {},
        outpaths: t.Dict = {},
    ):
        self.targets = targets
        self.param_sets = param_sets
        self.outpaths = outpaths

    def __getitem__(self, item):
        return getattr(self, item)

    @classmethod
    def default(cls, kind=[]):
        """
        Include buddings and bud_metric and estimates of their time derivatives.

        Parameters
        ----------
        kind: list of str
            If "ph_batman" included, add targets for experiments using pHlourin.
        """
        # each subitem specifies the function to be called and the location
        # on the h5 file to be written
        targets = {
            "prepost": {
                "merger": "/extraction/general/None/area",
                "picker": ["/extraction/general/None/area"],
            },
            "processes": [
                [
                    "buddings",
                    ["/extraction/general/None/volume"],
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
        # pHlourin experiments are special
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
        """
        Initialise PostProcessor

        Parameters
        ----------
        filename: str or PosixPath
            Name of h5 file.
        parameters: PostProcessorParameters object
            An instance of PostProcessorParameters.
        """
        super().__init__(parameters)
        self._filename = filename
        self._signal = Signal(filename)
        self._writer = Writer(filename)
        # parameters for merger and picker
        dicted_params = {
            i: parameters["param_sets"]["prepost"][i]
            for i in ["merger", "picker"]
        }
        for k in dicted_params.keys():
            if not isinstance(dicted_params[k], dict):
                dicted_params[k] = dicted_params[k].to_dict()
        # merger and picker
        self.merger = Merger(
            MergerParameters.from_dict(dicted_params["merger"])
        )
        self.picker = Picker(
            PickerParameters.from_dict(dicted_params["picker"]),
            cells=Cells.from_source(filename),
        )
        # processes, such as buddings
        self.classfun = {
            process: get_process(process)
            for process, _ in parameters["targets"]["processes"]
        }
        # parameters for the process in classfun
        self.parameters_classfun = {
            process: get_parameters(process)
            for process, _ in parameters["targets"]["processes"]
        }
        # locations to be written in the h5 file
        self.targets = parameters["targets"]

    def run_prepost(self):
        """Using picker, get and write lineages, returning mothers and daughters."""
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
                    picked_indices.T,
                    # names=["trap", "cell_label", "mother_label"],
                    names=["trap", "cell_label"],
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
        """
        Write the results to the h5 file.
        Processes include identifying buddings and finding bud metrics.
        """
        # run merger, picker, and find lineages
        self.run_prepost()
        # run processes
        for process, datasets in tqdm(self.targets["processes"]):
            if process in self.parameters["param_sets"].get("processes", {}):
                # parameters already assigned
                parameters = self.parameters_classfun[process](
                    self.parameters[process]
                )
            else:
                # assign parameters
                parameters = self.parameters_classfun[process].default()
            # load process
            loaded_process = self.classfun[process](parameters)
            if isinstance(parameters, LineageProcessParameters):
                loaded_process.lineage = self.lineage

            # apply process to each dataset
            for dataset in datasets:
                self.run_process(dataset, process, loaded_process)

    def run_process(self, dataset, process, loaded_process):
        """Run process on a single dataset and write the result."""
        # define signal
        if isinstance(dataset, list):
            # multisignal process
            signal = [self._signal[d] for d in dataset]
        elif isinstance(dataset, str):
            signal = self._signal[dataset]
        else:
            raise ("Incorrect dataset")
        # run process on signal
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
        # define outpath, where result will be written
        if process in self.parameters["outpaths"]:
            outpath = self.parameters["outpaths"][process]
        elif isinstance(dataset, list):
            # no outpath is defined
            # place the result in the minimum common branch of all signals
            prefix = "".join(
                c[0]
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
        # add postprocessing to outpath when required
        if process not in self.parameters["outpaths"]:
            outpath = "/postprocessing/" + process + "/" + outpath
        # write result
        if isinstance(result, dict):
            # multiple Signals as output
            for k, v in result.items():
                self.write_result(
                    outpath + f"/{k}",
                    v,
                    metadata={},
                )
        else:
            # a single Signal as output
            self.write_result(
                outpath,
                result,
                metadata={},
            )

    def write_result(
        self,
        path: str,
        result: t.Union[t.List, pd.DataFrame, np.ndarray],
        metadata: t.Dict,
    ):
        self._writer.write(path, result, meta=metadata, overwrite="overwrite")
