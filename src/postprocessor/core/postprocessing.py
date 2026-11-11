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
        targets: dict | None = None,
        param_sets: dict | None = None,
        outpaths: dict | None = None,
    ):
        """Initialise, with no defaults."""
        self.targets = {} if targets is None else targets
        self.param_sets = {} if param_sets is None else param_sets
        self.outpaths = {} if outpaths is None else outpaths

    def __getitem__(self, item):
        """Access attributes, like dict keys."""
        return getattr(self, item)

    @classmethod
    def default(
        cls,
        picker: dict | None = None,
        merger: dict | None = None,
        bud_processes: dict | None = None,
        targets: dict | None = None,
        outpaths: dict | None = None,
    ):
        """
        Include buddings and bud volumes.

        Parameters
        ----------
        picker: dict, optional
            Parameters of the picker to use in place of its defaults,
            such as {"picker_sequence": [["condition", "present", 3]]}.
        merger: dict, optional
            Parameters of the merger to use in place of its defaults,
            such as {"tolerance": 0.1}.
        bud_processes: dict, optional
            Parameters for bud processes, by the name of the process.
        targets: dict, optional
            Targets to use in place of the defaults, by their keys:
            "merging_picking", a dict of the Signals the merger and the
            picker use, and "bud_processes", a list of each process with
            the Signals it is applied to.
        outpaths: dict, optional
            Where in the h5 file to write a bud process's result, by the
            name of the process.

        Examples
        --------
        Pass these to PipelineParameters.default as

            postprocessing={
                "picker": {"picker_sequence": [["condition", "present", 3]]}
            }
        """
        default_targets = cls.default_targets()
        if targets is not None:
            unknown = set(targets) - set(default_targets)
            if unknown:
                raise ValueError(
                    f"Unknown targets {sorted(unknown)}: use"
                    f" {sorted(default_targets)}."
                )
            default_targets.update(targets)
        param_sets = {
            "merging_picking": {
                "merger_params": MergerParameters.default(**(merger or {})),
                "picker_params": PickerParameters.default(**(picker or {})),
            }
        }
        if bud_processes:
            param_sets["bud_processes"] = dict(bud_processes)
        return cls(
            targets=default_targets,
            param_sets=param_sets,
            outpaths=dict(outpaths or {}),
        )

    @staticmethod
    def default_targets() -> dict:
        """Return the Signals that are merged, picked and found for buds."""
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
        return targets


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
        # buddings and budmetric functions are in postprocessor/core/reshapers
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

        Cells are picked from the Signal as this run merges it and with
        the lineage this run finds. Signal.get applies the merges and
        picks in the h5 file instead, which are a previous run's or, in
        a new file, none: the picks then depended on how many times the
        file had been processed and could only ever lose cells.
        """
        # run merger
        record = self.get_raw(self.targets["merging_picking"]["merger"])
        self.no_cells = record is None
        if self.no_cells:
            # no cell at any time point: nothing to merge, relate or pick
            return np.array([]), np.empty((0, 3), dtype=int), np.array([])
        merges = self.merger.run(record)
        # get lineages from cells object attached to picker
        lineage = assoc_indices_to_3d(self.picker.cells.mothers_daughters)
        if lineage.any() and merges.any():
            # update lineages and merges after merging
            new_lineage, new_merges = merge_lineage(lineage, merges)
        else:
            new_lineage = lineage
            new_merges = merges
        new_lineage = assoc_indices_to_2d(new_lineage)
        # run picker on the merged Signal, with the merged lineage
        self.picker.lineage = new_lineage
        merged_record = self.signal.apply_merging_picking(
            self.signal.get_raw(self.targets["merging_picking"]["picker"]),
            merges=new_merges,
            picks=False,
        )
        picked_indices = np.array(self.picker.run(merged_record))
        return new_merges, new_lineage, picked_indices

    def get_raw(self, dataset: str, **kwargs) -> pd.DataFrame | None:
        """
        Get a raw Signal, or None if the h5 file does not hold it.

        A Signal with no value at any time point is not written, so the
        file of a position with no cells has no area and no volume.
        """
        try:
            return self.signal.get_raw(dataset, **kwargs)
        except KeyError:
            return None

    def get_signal(self, dataset: str):
        """
        Get a Signal with this run's merges and picks applied.

        Time is not in minutes because this data is rewritten to the h5
        file.
        """
        record = self.get_raw(dataset, in_minutes=False)
        if record is None:
            return None
        picked_merged = self.signal.apply_merging_picking(
            record, merges=self.merges, picks=self.picks
        )
        return self.signal.add_name(picked_merged, dataset)

    def run(self):
        """
        Write the results to the h5 file.

        Processes include identifying buddings and finding bud metrics.
        """
        # run merger, picker, and find lineages
        merges, lineage, picked_indices = self.run_merging_picking()
        # the bud processes use these, not those in the h5 file
        self.merges = merges
        self.picks = set(map(tuple, picked_indices.tolist()))
        # store result using their h5 dataset names
        res = {
            "merges": merges,
            "lineage_merged": lineage,
            "picks": picked_indices,
        }
        if self.no_cells:
            self.log(
                "No cells at any time point: nothing to postprocess.", "info"
            )
            return res
        # run processes: process is a str; datasets is a list of str
        for bud_process, datasets in tqdm(self.targets["bud_processes"]):
            assigned = self.parameters["param_sets"].get("bud_processes", {})
            if bud_process in assigned:
                # parameters already assigned
                parameters = self.parameters_bud_process_funcs[
                    bud_process
                ].from_dict(assigned[bud_process])
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
                outpath_result = self.run_bud_process(
                    dataset, bud_process, loaded_bud_process
                )
                if outpath_result is None:
                    self.log(
                        f"{bud_process} skipped: no Signal {dataset} in"
                        " the h5 file."
                    )
                    continue
                bud_outpath, bud_result = outpath_result
                res[bud_outpath] = bud_result
        return res

    def run_bud_process(self, dataset, bud_process, loaded_bud_process):
        """Run processes to obtain single data sets and write the results."""
        # get pre-processed data
        if isinstance(dataset, list):
            signal = [self.get_signal(d) for d in dataset]
        elif isinstance(dataset, str):
            signal = self.get_signal(dataset)
        else:
            raise TypeError("postprocessing: Incorrect dataset.")
        # run process on signal
        if signal is None:
            return None
        if len(signal) and (
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
            raise ValueError(f"Outpath not defined {type(dataset)}")
        # add postprocessing to outpath when required
        if bud_process not in self.parameters["outpaths"]:
            outpath = "/postprocessing/" + bud_process + "/" + outpath
        return outpath, result
