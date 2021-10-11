from agora.base import ParametersABC, ProcessABC


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
