from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path, PosixPath
from typing import Union
from copy import copy
from yaml import dump, safe_load


class ParametersABC(ABC):
    """
    Defines parameters as attributes and allows parameters to
    be converted to either a dictionary or to yaml.

    No attribute should be called "parameters"!
    """

    def __init__(self, **kwargs):
        '''
        Defines parameters as attributes
        '''
        assert "parameters" not in kwargs, "No attribute should be named parameters"
        for k, v in kwargs.items():
            setattr(self, k, v)

    ###

    def to_dict(self, iterable="null"):
        """
        Recursive function to return a nested dictionary of the
        attributes of the class instance.
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
            else:
                return iterable
        elif iterable == "null":
            # use instance's built-in __dict__ dictionary of attributes
            return self.to_dict(self.__dict__)
        else:
            return iterable

    def to_yaml(self, path: Union[PosixPath, str] = None):
        """
        Returns a yaml stream of the attributes of the class instance.
        If path is provided, the yaml stream is saved there.

        Parameters
        ----------
        path : Union[PosixPath, str]
            Output path.
        """
        if path:
            with open(Path(path), "w") as f:
                dump(self.to_dict(), f)
        return dump(self.to_dict())

    ###

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    @classmethod
    def from_yaml(cls, source: Union[PosixPath, str]):
        """
        Returns instance from a yaml filename or stdin
        """
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

###


class ProcessABC(ABC):
    """
    Base class for processes.
    Defines parameters as attributes and requires run method to be defined.
    """

    def __init__(self, parameters):
        """
        Arguments
        ---------
        parameters: instance of ParametersABC
        """
        self._parameters = parameters
        # convert parameters to dictionary
        # and then define each parameter as an attribute
        for k, v in parameters.to_dict().items():
            setattr(self, k, v)

    @property
    def parameters(self):
        return self._parameters

    @abstractmethod
    def run(self):
        pass
