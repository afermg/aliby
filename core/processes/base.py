from pathlib import Path, PosixPath
from typing import Union
from abc import ABC, abstractmethod


class ParametersABC(ABC):
    """
    Base class to add yaml functionality to parameters

    """

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    def to_yaml(self, path: Union[PosixPath, str]):
        return dump(self.__dict__, path)

    @classmethod
    def from_yaml(cls, path: Union[PosixPath, str]):
        with open(Path(file)) as f:
            params = safe_load(f)
        return cls(**params)

    @classmethod
    @abstractmethod
    def default(cls):
        pass


class ProcessABC(ABC):
    "Base class for processes"

    def __init__(self, parameters):
        self._parameters = parameters

        for k, v in parameters.to_dict().items():  # access parameters directly
            setattr(self, k, v)

    @property
    def parameters(self):
        return self._parameters

    @abstractmethod
    def run(self):
        pass
