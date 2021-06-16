import pandas as pd
from postprocessor.core.base import ParametersABC
from core import Cells


class PostProParameters(ParametersABC):
    """
    Anthology of parameters used for postprocessing
    """

    def __init__(self, merger=None, picker=None, processes=None, datasets=None):
        self.merger = merger
        self.picker = picker
        self.processes = processes

        self.datasets = datasets

    def __getitem__(self, item):
        return getattr(self, item)


class PostProcessor:
    def __init__(self, fname, parameters, signals):
        self.parameters = parameters

        self._signals = Signals(fname)
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


class Signals:
    """
    Class that fetches data from the hdf5 storage for post-processing
    """

    def __init__(self, file):
        self._hdf = h5py.File(file, "r")

    def __getitem__(self, dataset):
        dset = self._hdf[dataset][()]
        attrs = self._hdf[dataset].attrs
        first_dataset = "/" + dataset.split("/")[0] + "/"
        timepoints = self._hdf[first_dataset].attrs["processed_timepoints"]

        if "cell_label" in self._hdf[dataset].attrs:
            ids = pd.MultiIndex.from_tuples(
                zip(attrs["trap"], attrs["cell_label"]), names=["trap", "cell_label"]
            )
        else:
            ids = pd.Index(attrs["trap"], names=["trap"])

        return pd.DataFrame(dset, index=ids, columns=timepoints)

    @staticmethod
    def _if_ext_or_post(name):
        if name.startswith("extraction") or name.startswith("postprocessing"):
            if len(name.split("/")) > 3:
                return name

    @property
    def datasets(self):
        return signals._hdf.visit(self._if_ext_or_post)

    def close(self):
        self._hdf.close()
