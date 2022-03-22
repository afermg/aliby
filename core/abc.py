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
    def as_function(cls, data, *args, **kwargs):
        # Find the parameter's default
        parameters = get_parameters(cls.__name__).default(*args, **kwargs)
        return cls(parameters=parameters).run(data)


def get_process(process, suffix=""):
    """
    Dynamically import a process class from the 'processes' folder.
    Assumes process filename and class name are the same
    """
    location = f"postprocessor.core.processes.{process}.{process}{suffix}"
    found = locate(location)
    if found == None:
        raise Exception(f"{location} not found")
    return found


def get_parameters(process):
    """
    Dynamically import parameters from the 'processes' folder.
    Assumes parameter is the same name as the file with 'Parameters' added at the end.
    """
    return get_process(process, suffix="Parameters")
