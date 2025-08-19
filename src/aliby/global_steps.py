"""
Define the processes that take entire time series (global) data as inputs.

The most common example is tracking methods, which require object object masks alongside images.
"""

from typing import Callable

import nahual.client.trackastra as tr
import numpy
import pyarrow


def nahual_trackastra_load_model(
    address: str, parameters: dict[str, str]
) -> dict[str, str]:
    model_info = tr.load_model(parameters, address=address)
    return model_info


def nahual_trackastra_process_data(
    address: str, input_data: numpy.ndarray
) -> pyarrow.Table:
    """
    Send data and receive the response, finally convert the resultant
    dictionaries into a pyarrow table.
    """
    tracking = tr.process_data(input_data, address=address)

    tracks = pyarrow.Table.from_pylist(tracking)

    return tracks


def dispatch_global_step(step_name: str) -> tuple[Callable, Callable]:
    """
    Return the functions to load and process data.
    """
    if step_name == "nahual_trackastra":
        return nahual_trackastra_load_model, nahual_trackastra_process_data
    else:
        raise Exception(f"Step name {step_name} not found.")
