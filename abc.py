from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path, PosixPath
from typing import Union
from copy import copy

from yaml import dump, safe_load


class ParametersABC(ABC):
    """
    Base class to add yaml functionality to parameters

    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self, iterable="null"):
        """
        Recursive function that converts class to nested dictionary.
        """
        if isinstance(iterable, dict):
            if any(
                [
                    True
                    for x in iterable.values()
                    if isinstance(x, Iterable) or hasattr(x, "to_dict")
                ]
            ):
                return {
                    k: v.to_dict() if hasattr(v, "to_dict") else self.to_dict(v)
                    for k, v in iterable.items()
                }
            return iterable
        elif iterable == "null":
            return self.to_dict(self.__dict__)
        else:
            return iterable

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    def to_yaml(self, path: Union[PosixPath, str] = None):
        """Return instance as yaml stream and optionally export to file.

        Returns the yaml version of the class instance. If path is provided, it
        is saved there as well.

        Parameters
        ----------
        path : Union[PosixPath, str]
            Output path.

        """

        if path:
            with open(Path(path), "w") as f:
                dump(self.to_dict(), f)
        return dump(self.to_dict())

    @classmethod
    def from_yaml(cls, source: Union[PosixPath, str]):
        """Returns class from a yaml filename or stdin"""
        is_buffer = True
        try:
            if Path(source).exists():
                is_buffer = False
        except:
            pass
        if is_buffer:
            params = safe_load(source)
        else:
            with open(source) as f:
                params = safe_load(f)

        return cls(**params)

    @classmethod
    def default(cls, **kwargs):
        overriden_defaults = copy(cls._defaults)

        for k, v in kwargs.items():
            overriden_defaults[k] = v

        return cls.from_dict(overriden_defaults)


class ProcessABC(ABC):
    """Base class for processes"""

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
