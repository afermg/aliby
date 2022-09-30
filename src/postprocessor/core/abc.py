#!/usr/bin/env python3

from pydoc import locate

from agora.abc import ProcessABC


class PostProcessABC(ProcessABC):
    """
    Extend ProcessABC to add as_function, allowing for all PostProcesses to be called as functions
    almost directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def as_function(cls, data, *extra_data, **kwargs):
        # Find the parameter's default
        parameters = cls.default_parameters(**kwargs)
        return cls(parameters=parameters).run(data, *extra_data)

    @classmethod
    def default_parameters(cls, *args, **kwargs):
        return get_parameters(cls.__name__).default(*args, **kwargs)


def get_process(process, suffix=""):
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

    found = None
    for possible_location in possible_locations:
        location = (
            f"{base_location}.{possible_location}.{process}.{process}{suffix}"
        )
        found = locate(location)
        if found is not None:
            return found

    raise Exception(
        f"{process} not found in locations {possible_locations} at {base_location}"
    )


def get_parameters(process):
    """
    Dynamically import parameters from the 'processes' folder.
    Assumes parameter is the same name as the file with 'Parameters' added at the end.
    """
    return get_process(process, suffix="Parameters")
