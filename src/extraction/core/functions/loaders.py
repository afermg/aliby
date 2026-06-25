import typing as t
from functools import partial
from inspect import getfullargspec, getmembers, isfunction

import numpy as np
from cp_measure.bulk import get_core_measurements, get_correlation_measurements

from extraction.core.functions import cell, trap

"""
Load functions for analysing cells and their background.

Note that inspect.getmembers returns a list of function names
and functions, and inspect.getfullargspec returns a
function's arguments.
"""


def load_cellfuns_core():
    """Load functions from the cell module and return as a dict."""
    return {
        f[0]: f[1]
        for f in getmembers(cell)
        if isfunction(f[1]) and f[1].__module__.startswith("extraction.core.functions")
    }


def load_cellfuns(
    cp_measure_kwargs: t.Mapping[str, t.Mapping[str, t.Any]] | None = None,
):
    """
    Create a dict of core functions for use on cell_masks.

    The core functions only work on a single mask.

    Parameters
    ----------
    cp_measure_kwargs : mapping of {feature_name: kwargs_dict} or None
        Optional per-feature kwargs forwarded to the underlying
        ``cp_measure`` function (e.g.
        ``{"intensity": {"edge_measurements": False}}``). Keys must
        match cp_measure feature names from
        ``get_core_measurements()`` / ``get_correlation_measurements()``.
        Defaults to ``None`` → no extra kwargs (existing behaviour).
        Must be plain dicts of primitives so the resulting
        ``functools.partial`` round-trips through cloudpickle when
        joblib loky workers fan out.
    """
    cp_measure_kwargs = dict(cp_measure_kwargs or {})

    # create dict of the core functions from cell.py - these functions apply to a single mask
    cell_funs = load_cellfuns_core()
    # create a dict of functions that apply the core functions to an array of cell_masks
    CELL_FUNS = {}

    for f_name, f in cell_funs.items():
        if isfunction(f):
            args = getfullargspec(f).args
            if len(args) == 1:
                # function that applies f to m, a binary mask
                new_fun = partial(ignore_pixels, cell_fun=f)
            else:
                # function that applies f to m and img, a binary mask and a 2 or 3d image.
                new_fun = f

            CELL_FUNS[f_name] = new_fun

    # Add cp_measure measurements. Per-feature kwargs (if any) are baked
    # into the partial so they survive cloudpickle when joblib loky
    # serialises CELL_FUNS for cross-process fan-out.
    for fun_name, f in get_core_measurements().items():
        kw = dict(cp_measure_kwargs.get(fun_name, {}))
        CELL_FUNS[fun_name] = partial(wrap_cp_measure_features, fun=f, fun_kwargs=kw)

    for fun_name, f in get_correlation_measurements().items():
        kw = dict(cp_measure_kwargs.get(fun_name, {}))
        CELL_FUNS[fun_name] = partial(wrap_cp_corr_features, fun=f, fun_kwargs=kw)

    return CELL_FUNS


def load_trapfuns():
    """Load functions that are applied to an entire tile."""
    TRAP_FUNS = {
        f[0]: f[1]
        for f in getmembers(trap)
        if isfunction(f[1]) and f[1].__module__.startswith("extraction.core.functions")
    }
    return TRAP_FUNS


def load_funs(
    cp_measure_kwargs: t.Mapping[str, t.Mapping[str, t.Any]] | None = None,
):
    """Combine all automatically loaded functions.

    Parameters
    ----------
    cp_measure_kwargs : mapping of {feature_name: kwargs_dict} or None
        Forwarded to :func:`load_cellfuns` so cp_measure feature
        partials carry per-feature kwargs. Default ``None`` preserves
        the historical behaviour for callers that don't customise it.
    """
    CELL_FUNS = load_cellfuns(cp_measure_kwargs=cp_measure_kwargs)
    TRAP_FUNS = load_trapfuns()
    # return dict of cell funs, dict of trap funs, and dict of both
    return CELL_FUNS, TRAP_FUNS, {**TRAP_FUNS, **CELL_FUNS}


def load_redfuns() -> t.Dict[str, t.Callable]:
    """
    Load functions to reduce a multidimensional image by one dimension.

    Parameters
    ----------
    additional_reducers: function or a dict of functions (optional)
        Functions to perform the reduction.
    """
    RED_FUNS = {
        "max": np.maximum,
        "mean": np.mean,
        "median": np.median,
        "div": np.divide,
        "add": np.add,
        "None": None,
    }
    return RED_FUNS


# Functional solutions to complex problems
# currently all these wrappers assume that the input mask is a binary array
# In the future we may want to replace this with an array of integer labels instead.


def wrap_cp_measure_features(
    mask: np.ndarray,
    pixels: np.ndarray,
    fun: t.Callable = None,
    fun_kwargs: t.Mapping[str, t.Any] | None = None,
) -> t.Callable:
    """Apply a cp_measure single-image feature ``fun`` to ``(mask, pixels)``.

    ``fun_kwargs`` is an optional per-feature kwargs dict (e.g.
    ``{"edge_measurements": False}`` for ``intensity``); it is baked
    into the partial by :func:`load_cellfuns`, so each invocation only
    needs to spread it on top of the positional args.
    """
    kw = fun_kwargs or {}
    results = fun(mask.astype(np.uint16), pixels, **kw)
    return results


def wrap_cp_corr_features(
    mask: np.ndarray,
    pixels1: np.ndarray,
    pixels2: np.ndarray,
    fun: t.Callable = None,
    fun_kwargs: t.Mapping[str, t.Any] | None = None,
) -> t.Callable:
    """Apply a cp_measure two-image correlation ``fun`` to ``(pixels1, pixels2, mask)``.

    ``fun_kwargs`` mirrors :func:`wrap_cp_measure_features` and is also
    baked in at partial-construction time.
    """
    kw = fun_kwargs or {}
    results = fun(pixels1, pixels2, mask, **kw)
    return results


def ignore_pixels(mask, pixels, cell_fun):
    return cell_fun(mask)
