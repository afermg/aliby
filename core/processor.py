import pandas as pd


class Parameters:
    def __init__(self, merger=None, picker=None, processes=None, branches=None):
        self.merger = merger
        self.picker = picker
        self.processes = processes
        self.branches = branches

    def __getitem__(self, item):
        return getattr(self, item)


class PostProcessor:
    def __init__(self, parameters, signals):
        self.parameters = parameters

        self.merger = Merger(parameters["merger"])
        self.picker = Picker(parameters["picker"])
        self.processes = [
            self.get_process(process) for process in parameters["processes"]
        ]
        self.branches = parameters["branches"]

    def run(self):
        self.merger.run(signals.get_branch(self.branches["merger"]))
        self.picker.run(signals.get_branch(self.branches["picker"]))
        for process, branch in zip(self.processes, self.branches["processes"]):
            process.run(signals.get_branch(branch))


class Signals:
    """
    Class that fetches data from the hdf5 storage for post-processing
    """

    def __init__(self, file):
        self._hdf = h5py.File(file, "r")

    @staticmethod
    def _if_ext_or_post(name):
        if name.startswith("extraction") or name.startswith("postprocessing"):
            if len(name.split("/")) > 3:
                return name

    @property
    def branches(self):
        return signals._hdf.visit(self._if_ext_or_post)

    def get_branch(self, branch):
        return self._hdf[branch][()]

    def branch_to_df(self, branch):
        dset = self._hdf[branch][()]
        attrs = self._hdf[branch].attrs
        first_branch = "/" + branch.split("/")[0] + "/"
        timepoints = self._hdf[first_branch].attrs["processed_timepoints"]

        if "cell_label" in self._hdf[branch].attrs:
            ids = pd.MultiIndex.from_tuples(
                zip(attrs["trap"], attrs["cell_label"]), names=["trap", "cell_label"]
            )
        else:
            ids = pd.Index(attrs["trap"], names=["trap"])

        return pd.DataFrame(dset, index=ids, columns=timepoints)

    def close(self):
        self._hdf.close()
