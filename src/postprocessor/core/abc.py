"""An extension of ProcessABC that allows PostProcesses to be functions."""

import re
from itertools import product
from pydoc import locate

from agora.abc import ParametersABC, ProcessABC


class PostProcessABC(ProcessABC):
    """
    Extend ProcessABC to add as_function.

    Allow PostProcesses to be called as functions.
    """

    def __init__(self, *args, **kwargs):
        """Initialise using ProcessABC."""
        super().__init__(*args, **kwargs)

    @classmethod
    def as_function(cls, data, *extra_data, **kwargs):
        """Return the instance as a function by calling 'run'."""
        # Find the parameter's default
        parameters = cls.default_parameters(**kwargs)
        return cls(parameters=parameters).run(data, *extra_data)

    @classmethod
    def default_parameters(cls, *args, **kwargs):
        """Get default parameters."""
        return get_parameters(cls.__name__).default(*args, **kwargs)


def get_process(
    process: str, suffix: str = ""
) -> PostProcessABC or ParametersABC or None:
    """
    Dynamically import a process class from the available process locations.

    Assume identical process filename and class names.

    There are three potential types of processes:
    Processes return the same shape as their input;
    MultiSignal either take or return multiple datasets or both;
    Reshapers, merger and picker, return a different shape for processes.

    suffix : str
        Name of suffix, generally "" (empty) or "Parameters".
    """
    base_location = "postprocessor.core"
    possible_locations = ("reshapers",)
    valid_syntaxes = (
        _to_snake_case(process),
        _to_pascal_case(_to_snake_case(process)),
    )
    found = None
    for possible_location, process_syntax in product(
        possible_locations, valid_syntaxes
    ):
        location = (
            f"{base_location}.{possible_location}."
            + f"{_to_snake_case(process)}.{process_syntax}{suffix}"
        )
        # instantiate class but not a class object
        found = locate(location)
        if found is not None:
            break
    else:
        raise FileNotFoundError(
            f"{process} not found in locations {possible_locations} "
            f"at {base_location}"
        )
    return found


def get_parameters(process: str) -> ParametersABC:
    """
    Dynamically import parameters from the 'processes' directory.

    Assume parameters have the same name as the file with
    'Parameters' added at the end.
    """
    return get_process(process, suffix="Parameters")


def _to_pascal_case(snake_str: str) -> str:
    """Convert a snake_case string to PascalCase."""
    # Based on https://stackoverflow.com/a/19053800
    components = snake_str.split("_")
    return "".join(x.title() for x in components)


def _to_snake_case(pascal_str: str) -> str:
    """Convert a PascalCase string to snake_case."""
    # Based on https://stackoverflow.com/a/12867228
    return re.sub("(?!^)([A-Z]+)", r"_\1", pascal_str).lower()
