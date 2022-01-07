import numpy as np
from inspect import getmembers, isfunction, getargspec
from extraction.core.functions import cell, trap
from extraction.core.functions.custom import localisation
from extraction.core.functions.math import div0
from extraction.core.functions.distributors import trap_apply


def load_cellfuns_core():
    # Generate str -> trap_function dict from functions in core.cell
    return {f[0]: f[1] for f in getmembers(cell) if isfunction(f[1])}


def load_custom_args():
    """
    Load custom functions. If they have extra arguments also load these
    """
    funs = {f[0]: f[1] for f in getmembers(localisation) if isfunction(f[1])}
    args = {
        k: getargspec(v).args[2:]
        for k, v in funs.items()
        if set(["cell_mask", "trap_image"]).intersection(getargspec(v).args)
    }

    return ({k: funs[k] for k in args.keys()}, {k: v for k, v in args.items() if v})


def load_cellfuns():
    # Generate str -> trap_function dict from core.cell and core.trap functions
    cell_funs = load_cellfuns_core()
    CELLFUNS = {}
    for k, f in cell_funs.items():
        if isfunction(f):

            def tmp(f):
                return lambda m, img: trap_apply(f, m, img)

            CELLFUNS[k] = tmp(f)
    return CELLFUNS


def load_trapfuns():
    TRAPFUNS = {f[0]: f[1] for f in getmembers(trap) if isfunction(f[1])}
    return TRAPFUNS


def load_funs():
    CELLFUNS = load_cellfuns()
    TRAPFUNS = load_trapfuns()

    return CELLFUNS, TRAPFUNS, {**TRAPFUNS, **CELLFUNS}


def load_redfuns():  # TODO make defining reduction functions more flexible
    RED_FUNS = {
        "np_max": np.maximum,
        "np_mean": np.mean,
        "np_median": np.median,
        "None": None,
    }
    return RED_FUNS


def load_mergefuns():
    MERGE_FUNS = {"div0": div0, "np_add": np.add}
    return MERGE_FUNS
