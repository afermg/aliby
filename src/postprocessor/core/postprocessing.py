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
    assoc_indices_to_2d,
    assoc_indices_to_3d,
)
from agora.utils.merge import merge_lineage
from postprocessor.core.abc import get_parameters, get_process
from postprocessor.core.lineageprocess import (
    LineageProcess,
    LineageProcessParameters,
)
from postprocessor.core.reshapers.merger import Merger, MergerParameters
from postprocessor.core.reshapers.picker import Picker, PickerParameters


class PostProcessorParameters(ParametersABC):
    """
    Parameters used for post-processing.

    Define defaults for picker, merger, and bud metrics.
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
        """Access attributes like dict keys."""
        return getattr(self, item)

    @classmethod
    def default(cls):
        """Include buddings and bud volumes."""
        # each subitem specifies the function to be called
        # and the h5-file location for the results
        targets = {
            "merging_picking": {
                "merger": "/extraction/general/None/area",
                "picker": "/extraction/general/None/area",
            },
            # lists because bud_metric can be applied to multiple signals
            "processes": [
                ["buddings", ["/extraction/general/None/volume"]],
                ["bud_metric", ["/extraction/general/None/volume"]],
            ],
        }
        param_sets = {
            "merging_picking": {
                "merger_params": MergerParameters.default(),
                "picker_params": PickerParameters.default(),
            }
        }
        outpaths = {}
        return cls(targets=targets, param_sets=param_sets, outpaths=outpaths)


class PostProcessor(ProcessABC):
    """Process data from h5 files."""

    def __init__(self, filename, parameters):
        """
        Initialise PostProcessor.

        Parameters
        ----------
        filename: str or PosixPath
            Name of h5 file.
        parameters: object
            An instance of PostProcessorParameters.
        """
        super().__init__(parameters)
        self.filename = filename
        self.signal = Signal(filename)
        self.writer = Writer(filename)
        # parameters for merger and picker
        dicted_params = {
            i: parameters["param_sets"]["merging_picking"][i + "_params"]
            for i in ["merger", "picker"]
        }
        for k in dicted_params.keys():
            if not isinstance(dicted_params[k], dict):
                dicted_params[k] = dicted_params[k].to_dict()
        # initialise merger and picker
        self.merger = Merger(
            MergerParameters.from_dict(dicted_params["merger"])
        )
        self.picker = Picker(
            PickerParameters.from_dict(dicted_params["picker"]),
            cells=Cells.from_source(filename),
        )
        # get processes, such as buddings
        self.process_funcs = {
            process: get_process(process)
            for process, _ in parameters["targets"]["processes"]
        }
        # get parameters for the processes
        self.parameters_process_funcs = {
            process: get_parameters(process)
            for process, _ in parameters["targets"]["processes"]
        }
        # locations to be written in the h5 file
        self.targets = parameters["targets"]

    def run_merging_picking(self):
        """
        Run merger, get lineages, and then run picker.

        Necessary before any processes can run.
        """
        # run merger
        record = self.signal.get_raw(self.targets["merging_picking"]["merger"])
        merges = self.merger.run(record)
        # get lineages from cells object attached to picker
        lineage = assoc_indices_to_3d(self.picker.cells.mothers_daughters)
        if merges.any():
            # update lineages and merges after merging
            new_lineage, new_merges = merge_lineage(lineage, merges)
        else:
            new_lineage = lineage
            new_merges = merges
        self.lineage = assoc_indices_to_2d(new_lineage)
        self.writer.write(
            "modifiers/merges", data=[np.array(x) for x in new_merges]
        )
        self.writer.write(
            "modifiers/lineage_merged", assoc_indices_to_2d(new_lineage)
        )
        # run picker
        picked_indices = np.array(
            self.picker.run(
                self.signal.get(self.targets["merging_picking"]["picker"])
            )
        )
        if picked_indices.any():
            self.writer.write(
                "modifiers/picks",
                data=pd.MultiIndex.from_arrays(
                    picked_indices.T, names=["trap", "cell_label"]
                ),
                overwrite="overwrite",
            )

    def run(self):
        """
        Write the results to the h5 file.

        Processes include identifying buddings and finding bud metrics.
        """
        # run merger, picker, and find lineages
        self.run_merging_picking()
        # run processes: process is a str; data sets is a list of str
        for process, datasets in tqdm(self.targets["processes"]):
            if process in self.parameters["param_sets"].get("processes", {}):
                # parameters already assigned
                parameters = self.parameters_process_funcs[process](
                    self.parameters[process]
                )
            else:
                # assign default parameters
                parameters = self.parameters_process_funcs[process].default()
            # load process - instantiate an object in the class
            loaded_process = self.process_funcs[process](parameters)
            if isinstance(parameters, LineageProcessParameters):
                loaded_process.lineage = self.lineage
            # apply process to each data set
            for dataset in datasets:
                self.run_process(dataset, process, loaded_process)

    def run_process(self, dataset, process, loaded_process):
        """Run processes to obtain single data sets and write the results."""
        # get pre-processed data
        if isinstance(dataset, list):
            signal = [self.signal.get(d) for d in dataset]
        elif isinstance(dataset, str):
            signal = self.signal.get(dataset)
        else:
            raise ("Incorrect dataset")
        # run process on signal
        if signal is None:
            return None
        elif len(signal) and (
            not isinstance(loaded_process, LineageProcess)
            or len(loaded_process.lineage)
        ):
            result = loaded_process.run(signal)
        else:
            result = pd.DataFrame(
                [], columns=signal.columns, index=signal.index
            )
            result.columns.names = ["timepoint"]
        # use outpath to write result
        if process in self.parameters["outpaths"]:
            # outpath already defined
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
            outpath = prefix + "_".join(
                [d[len(prefix) :].replace("/", "_") for d in dataset]
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
        """Write to h5 file."""
        self.writer.write(path, result, meta=metadata, overwrite="overwrite")

    @staticmethod
    def pick_mother(a, b):
        """
        Update the mother id.

        Ensure the mother has a lower id.
        """
        x = max(a, b)
        if min([a, b]):
            x = [a, b][np.argmin([a, b])]
        return x
