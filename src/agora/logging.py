#!/usr/bin/env jupyter
"""
Add general logging functions and decorators
"""

import logging
from time import perf_counter


def timer(func):
    # Log duration of a function into aliby logfile
    def wrap_func(*args, **kwargs):
        t1 = perf_counter()
        result = func(*args, **kwargs)
        logging.getLogger("aliby").debug(
            f"{func.__qualname__} took {(perf_counter()-t1):.4f}s"
        )
        return result

    return wrap_func
