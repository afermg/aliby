#!/usr/bin/env python3

import re
from itertools import product
from pydoc import locate

from agora.abc import ParametersABC, ProcessABC


class PostProcessABC(ProcessABC):
    """
    Extend ProcessABC to add as_function, allowing for all PostProcesses to be called as functions
    almost directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def as_function(cls, data, *extra_data, **kwargs):
        # FIXME can this be a __call__ method instead?
        # Find the parameter's default
        parameters = cls.default_parameters(**kwargs)
        return cls(parameters=parameters).run(data, *extra_data)

    @classmethod
    def default_parameters(cls, *args, **kwargs):
        return get_parameters(cls.__name__).default(*args, **kwargs)


def get_process(process, suffix="") -> PostProcessABC or ParametersABC or None:
    """
    Dynamically import a process class from the available process locations.
    Assumes process filename and class name are the same

    Processes return the same shape as their input.
    MultiSignal either take or return multiple datasets (or both).
    Reshapers return a different shape for processes: Merger and Picker belong here.

    suffix : str
        Name of suffix, generally "" (empty) or "Parameters".
    """
    base_location = "postprocessor.core"
    possible_locations = ("processes", "multisignal", "reshapers")
    valid_syntaxes = (
        _to_snake_case(process),
        _to_pascal_case(_to_snake_case(process)),
    )

    found = None
    for possible_location, process_syntax in product(
        possible_locations, valid_syntaxes
    ):

        location = f"{base_location}.{possible_location}.{_to_snake_case(process)}.{process_syntax}{suffix}"
        found = locate(location)
        if found is not None:
            break
    else:

        raise Exception(
            f"{process} not found in locations {possible_locations} at {base_location}"
        )
    return found


def get_parameters(process: str) -> ParametersABC:
    """
    Dynamically import parameters from the 'processes' folder.
    Assumes parameter is the same name as the file with 'Parameters' added at the end.
    """
    return get_process(process, suffix="Parameters")


def _to_pascal_case(snake_str: str) -> str:
    # Convert a snake_case string to PascalCase.
    # Based on https://stackoverflow.com/a/19053800
    components = snake_str.split("_")
    return "".join(x.title() for x in components)


def _to_snake_case(pascal_str: str) -> str:
    # Convert a snake_case string to PascalCase.
    # Based on https://stackoverflow.com/a/12867228
    return re.sub("(?!^)([A-Z]+)", r"_\1", pascal_str).lower()
