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
    def __init__(self, file):
        self._file = h5py.File(file, "r")

    def get_branch(self, branch):
        return self._file[branch][()]

    def branch_to_df(self):
        pass

    def close(self):
        self._file.close()
