#!/usr/bin/env jupyter
"""
Convenience decorators that extend commonly-used methods or functions.
"""
import typing as t
from functools import wraps


def _first_arg_str_to_df(
    fn: t.Callable,
):
    """Ensures Signal-like classes convert strings to datasets when calling them"""

    @wraps(fn)
    def format_input(*args, **kwargs):
        cls = args[0]
        data = args[1]
        if isinstance(data, str):
            data = cls.get_raw(data)
        return fn(cls, data, *args[2:], **kwargs)

    return format_input
