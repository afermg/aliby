"""Run picking, merging, and find buddings and bud metrics."""

import typing as t
from itertools import takewhile

import numpy as np
import pandas as pd
from tqdm import tqdm

from agora.abc import ParametersABC, ProcessABC
from agora.io.cells import Cells
from agora.io.signal import Signal

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
        """Initialise, with no defaults."""
        self.targets = targets
        self.param_sets = param_sets
        self.outpaths = outpaths

    def __getitem__(self, item):
        """Access attributes, like dict keys."""
        return getattr(self, item)

    @classmethod
    def default(cls):
        """Include buddings and bud volumes."""
        # each subitem specifies the function to be called
        # and the h5-file location for the results
        targets = {
            "merging_picking": {
                "merger": "/extraction/general/null/area",
                "picker": "/extraction/general/null/area",
            },
            # lists because bud_metric can be applied to multiple signals
            "bud_processes": [
                ["buddings", ["/extraction/general/null/volume"]],
                [
                    "bud_metric",
                    [
                        "/extraction/general/null/volume",
                        "/extraction/general/null/area",
                    ],
                ],
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
        self.signal = Signal(filename)
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
        # get bud processes
        self.bud_process_funcs = {
            bud_process: get_process(bud_process)
            for bud_process, _ in parameters["targets"]["bud_processes"]
        }
        # get parameters for bud processes
        self.parameters_bud_process_funcs = {
            bud_process: get_parameters(bud_process)
            for bud_process, _ in parameters["targets"]["bud_processes"]
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
        new_lineage = assoc_indices_to_2d(new_lineage)
        # run picker
        picked_indices = np.array(
            self.picker.run(
                self.signal.get(self.targets["merging_picking"]["picker"])
            )
        )
        return new_merges, new_lineage, picked_indices

    def run(self):
        """
        Write the results to the h5 file.

        Processes include identifying buddings and finding bud metrics.
        """
        # run merger, picker, and find lineages
        merges, lineage, picked_indices = self.run_merging_picking()
        # store result using their h5 dataset names
        res = {
            "merges": merges,
            "lineage_merged": lineage,
            "picks": picked_indices,
        }
        # run processes: process is a str; data sets is a list of str
        for bud_process, datasets in tqdm(self.targets["bud_processes"]):
            if bud_process in self.parameters["param_sets"].get(
                "bud_processes", {}
            ):
                # parameters already assigned
                parameters = self.parameters_bud_process_funcs[bud_process](
                    self.parameters[bud_process]
                )
            else:
                # assign default parameters
                parameters = self.parameters_bud_process_funcs[
                    bud_process
                ].default()
            # load bud_process - instantiate an object in the class
            loaded_bud_process = self.bud_process_funcs[bud_process](
                parameters
            )
            if isinstance(parameters, LineageProcessParameters):
                loaded_bud_process.lineage = lineage
            # apply bud process to each data set
            for dataset in datasets:
                bud_outpath, bud_result = self.run_bud_process(
                    dataset, bud_process, loaded_bud_process
                )
                res[bud_outpath] = bud_result
        return res

    def run_bud_process(self, dataset, bud_process, loaded_bud_process):
        """Run processes to obtain single data sets and write the results."""
        # get pre-processed data
        if isinstance(dataset, list):
            signal = [self.signal.get(d) for d in dataset]
        elif isinstance(dataset, str):
            signal = self.signal.get(dataset)
        else:
            raise TypeError("postprocessing: Incorrect dataset.")
        # run process on signal
        if signal is None:
            return None
        elif len(signal) and (
            not isinstance(loaded_bud_process, LineageProcess)
            or len(loaded_bud_process.lineage)
        ):
            result = loaded_bud_process.run(signal)
        else:
            result = pd.DataFrame(
                [], columns=signal.columns, index=signal.index
            )
            result.columns.names = ["timepoint"]
        # use outpath to write result
        if bud_process in self.parameters["outpaths"]:
            # outpath already defined
            outpath = self.parameters["outpaths"][bud_process]
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
            raise Exception(f"Outpath not defined {type(dataset)}")
        # add postprocessing to outpath when required
        if bud_process not in self.parameters["outpaths"]:
            outpath = "/postprocessing/" + bud_process + "/" + outpath
        return outpath, result

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
