import typing as t
from itertools import takewhile
from typing import Dict, List, Union

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from agora.abc import ParametersABC, ProcessABC
from agora.io.cells import Cells
from agora.io.signal import Signal
from agora.io.writer import Writer
from agora.utils.indexing import (
    _assoc_indices_to_3d,
    validate_association,
)
from agora.utils.kymograph import get_index_as_np
from postprocessor.core.abc import get_parameters, get_process
from postprocessor.core.lineageprocess import LineageProcessParameters
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
        merge_events = self.merger.run(record)

        self._writer.write(
            "modifiers/merges", data=[np.array(x) for x in merge_events]
        )

        lineage = _assoc_indices_to_3d(self.picker.cells.mothers_daughters)

        with h5py.File(self._filename, "a") as f:
            merge_events = f["modifiers/merges"][()]
        multii = pd.MultiIndex(
            [[], [], []],
            [[], [], []],
            names=["trap", "mother_label", "daughter_label"],
        )
        self.lineage_merged = multii

        indices = get_index_as_np(record)
        if merge_events.any():  # Update lineages after merge events
            # We validate merges that associate existing mothers and daughters
            valid_merges, valid_indices = validate_association(merges, indices)

            grouped_merges = group_merges(merges)
            # Sumarise the merges linking the first and final id
            # Shape (X,2,2)
            summarised = np.array(
                [(x[0][0], x[-1][1]) for x in grouped_merges]
            )
            # List the indices that weill be deleted, as they are in-between
            # Shape (Y,2)
            to_delete = np.vstack(
                [
                    x.reshape(-1, x.shape[-1])[1:-1]
                    for x in grouped_merges
                    if len(x) > 1
                ]
            )

            flat_indices = lineage.reshape(-1, 2)
            valid_merges, valid_indices = validate_association(
                summarised, flat_indices
            )
            # Replace
            id_eq_matrix = compare_indices(flat_indices, to_delete)

            # Update labels of merged tracklets
            flat_indices[valid_indices] = summarised[valid_merges, 1]

            # Remove labels that will be removed when merging
            flat_indices = flat_indices[id_eq_matrix.any(axis=1)]

            lineage_merged = flat_indices.reshape(-1, 2)

            self.lineage_merged = pd.MultiIndex.from_arrays(
                np.unique(
                    np.append(
                        trap_mother,
                        trap_daughter[:, 1].reshape(-1, 1),
                        axis=1,
                    ),
                    axis=0,
                ).T,
                names=["trap", "mother_label", "daughter_label"],
            )

        # Remove after implementing outside
        # self._writer.write(
        #     "modifiers/picks",
        #     data=pd.MultiIndex.from_arrays(
        #         # TODO Check if multiindices are still repeated
        #         np.unique(indices, axis=0).T if indices.any() else [[], []],
        #         names=["trap", "cell_label"],
        #     ),
        #     overwrite="overwrite",
        # )

        # combined_idx = ([], [], [])

        # multii = pd.MultiIndex.from_arrays(
        #     combined_idx,
        #     names=["trap", "mother_label", "daughter_label"],
        # )
        # self._writer.write(
        #     "postprocessing/lineage",
        #     data=multii,
        #     # TODO check if overwrite is still needed
        #     overwrite="overwrite",
        # )

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

            if isinstance(parameters, LineageProcessParameters):
                lineage = self._signal.lineage(
                    # self.parameters.lineage_location
                )
                loaded_process = self.classfun[process](parameters)
                loaded_process.lineage = lineage

            else:
                loaded_process = self.classfun[process](parameters)

            for dataset in datasets:
                if isinstance(dataset, list):  # multisignal process
                    signal = [self._signal[d] for d in dataset]
                elif isinstance(dataset, str):
                    signal = self._signal[dataset]
                else:
                    raise ("Incorrect dataset")

                if len(signal):
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
        self._writer.write(path, result, meta=metadata, overwrite="overwrite")


def union_find(lsts):
    sets = [set(lst) for lst in lsts if lst]
    merged = True
    while merged:
        merged = False
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = True
                    common |= x
            results.append(common)
        sets = results
    return sets


def group_merges(merges: np.ndarray) -> t.List[t.Tuple]:
    # Return a list where the cell is present as source and target
    # (multimerges)

    sources_targets = compare_indices(merges[:, 0, :], merges[:, 1, :])
    is_multimerge = sources_targets.any(axis=0) | sources_targets.any(axis=1)
    is_monomerge = ~is_multimerge

    multimerge_subsets = union_find(list(zip(*np.where(sources_targets))))
    return [
        *[merges[np.array(tuple(x))] for x in multimerge_subsets],
        *[[event] for event in merges[is_monomerge]],
    ]
