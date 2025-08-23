"""
Define the processes that take entire time series (global) data as inputs.

The most common example is tracking methods, which require object object masks alongside images.
"""

from functools import partial
from typing import Callable

import numpy
import pyarrow
from nahual.process import dispatch_setup_process


def nahual_trackastra_process_format(
    input_data: numpy.ndarray, address: str, process: Callable
) -> pyarrow.Table:
    """
    Expand the list  of data and receive the response, finally convert the resultant
    dictionaries into a pyarrow table.
    """
    tracking = process(data=input_data, address=address)

    tracks = pyarrow.Table.from_pydict(tracking)

    return tracks


def dispatch_global_step(step_name: str) -> tuple[Callable, Callable]:
    """
    Return the functions to load and process data.
    """
    if step_name == "nahual_trackastra":
        setup, process = dispatch_setup_process("trackastra")
        process_format = partial(nahual_trackastra_process_format, process=process)

    return setup, process_format
