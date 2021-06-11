class PostProcessor:
    def __init__(self, parameters):
        self.parameters = parameters

        self.merger = Merger(parameters["merger"])
        self.picker = Picker(parameters["picker"])
        self.processes, self.branches = [
            (self.get_process(process), branch)
            for process, branch in parameters["processes"]
        ]

    def run(self):
        pass
