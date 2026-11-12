import logging
import typing as t
from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy

from flatten_dict import flatten, unflatten

from agora.logging_timer import timer

# marks a call of to_dict with nothing passed; a value that a parameter
# could have, as "null" was, sent to_dict round for ever
whole_instance = object()


class ParametersABC(ABC):
    """
    Define parameters typically for a step in the pipeline.

    Output is a dict.
    No attribute should be called "parameters"!
    """

    def __init__(self, **kwargs):
        """Define parameters as attributes."""
        assert (
            "parameters" not in kwargs
        ), "No attribute should be named parameters"
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self, iterable=whole_instance) -> t.Dict:
        """
        Return a nested dictionary of the attributes of the class instance.

        Use recursion.
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
                    k: (
                        v.to_dict()
                        if hasattr(v, "to_dict")
                        else self.to_dict(v)
                    )
                    for k, v in iterable.items()
                }
            else:
                return iterable
        elif iterable is whole_instance:
            # use instance's built-in __dict__ dictionary of attributes
            return self.to_dict(self.__dict__)
        else:
            return iterable

    def log(self, message: str, level: str = "warning"):
        """Log messages at the corresponding level."""
        logger = logging.getLogger("aliby")
        getattr(logger, level)(f"{self.__class__.__name__}: {message}")

    @classmethod
    def from_dict(cls, d: dict):
        """Initialise from a dict of parameters."""
        return cls(**d)

    @classmethod
    def default(cls, **kwargs):
        """
        Initialise, replacing any of the default parameters.

        Copy the defaults in depth: a list among them, such as the
        picker's sequence, was shared by every instance, and changing
        one instance's changed the defaults of all that followed.
        """
        overriden_defaults = deepcopy(cls._defaults)
        for k, v in kwargs.items():
            overriden_defaults[k] = v
        return cls.from_dict(overriden_defaults)

    def update(self, name: str, new_value):
        """Update a parameter in the nested dict of parameters."""
        flat_params_dict = flatten(self.to_dict(), keep_empty_types=(dict,))
        names_found = [
            param for param in flat_params_dict.keys() if name in param
        ]
        if len(names_found) == 1:
            keys = names_found.pop()
            if type(flat_params_dict[keys]) is not type(new_value):
                self.log(f"Changing the type of {name} is risky.")
            flat_params_dict[keys] = new_value
            params_dict = unflatten(flat_params_dict)
            # replace all old values
            for key, value in params_dict.items():
                setattr(self, key, value)
        else:
            self.log(f"{name} was neither recognised nor updated.")


class ProcessABC(ABC):
    """
    Base class for processes.

    Define parameters as attributes and requires a run method.
    """

    def __init__(self, parameters):
        """
        Initialise by defining parameters as attributes.

        Arguments
        ---------
        parameters: instance of ParametersABC
        """
        self._parameters = parameters
        # convert parameters to dictionary
        for k, v in parameters.to_dict().items():
            # define each parameter as an attribute
            setattr(self, k, v)

    @property
    def parameters(self):
        """Get process's parameters."""
        return self._parameters

    def log(self, message: str, level: str = "warning"):
        """Log messages at the corresponding level."""
        logger = logging.getLogger("aliby")
        getattr(logger, level)(f"{self.__class__.__name__}: {message}")


class StepABC(ProcessABC):
    """
    Base class used for steps in aliby's pipeline.

    Includes a setup step, logging and time benchmarking.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _run_tp(self):
        pass

    @timer
    def run_tp(self, tp: int, **kwargs):
        """Time and log the timing of a step."""
        return self._run_tp(tp, **kwargs)
