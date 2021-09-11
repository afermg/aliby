from postprocessor.core.processes.base import ParametersABC, ProcessABC
from postprocessor.core.functions.tracks import clean_tracks, merge_tracks, join_tracks


class ParametersTemplate(ParametersABC):
    """
    Parameters
    """

    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    def default(cls):
        return cls.from_dict({})


class ProcessTemplate(ProcessABC):
    """
    Template for process class.
    """

    def __init__(self, parameters: ParametersTemplate):
        super().__init__(parameters)

    def run(self):
        pass
