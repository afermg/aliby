#!/usr/bin/env jupyter
"""
Convenience decorators to extend commonly-used methods or functions.
"""
import typing as t
from functools import wraps


def _first_arg_str_to_df(
    fn: t.Callable,
):
    """Enable Signal-like classes to convert strings to data sets."""
    @wraps(fn)
    def format_input(*args, **kwargs):
        cls = args[0]
        data = args[1]
        if isinstance(data, str):
            # get data from h5 file
            data = cls.get_raw(data)
        # replace path in the undecorated function with data
        return fn(cls, data, *args[2:], **kwargs)
    return format_input
