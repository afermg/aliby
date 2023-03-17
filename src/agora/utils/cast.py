#!/usr/bin/env jupyter

"""
Convert some types to others
"""


def _str_to_int(x: str or None):
    """
    Cast string as int if possible. If Nonetype return None.
    """
    if x is not None:
        try:
            return int(x)
        except:
            return x
