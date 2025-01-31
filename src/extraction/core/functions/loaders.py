import typing as t
from types import FunctionType
from inspect import getfullargspec, getmembers, isfunction, isbuiltin

import numpy as np
from cp_measure.bulk import get_core_measurements
from skimage.measure import regionprops_table

from extraction.core.functions import cell, trap
from extraction.core.functions.custom import localisation
from extraction.core.functions.distributors import trap_apply
from extraction.core.functions.math_utils import div0
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
        if isfunction(f[1])
        and f[1].__module__.startswith("extraction.core.functions")
    }



def load_cellfuns():
    """
    Create a dict of core functions for use on cell_masks.

    The core functions only work on a single mask.
    """
    # create dict of the core functions from cell.py - these functions apply to a single mask
    cell_funs = load_cellfuns_core()
    # create a dict of functions that apply the core functions to an array of cell_masks
    CELL_FUNS = {}
    def trap_apply_on_mask(f:FunctionType):
        """
        Wrapper to ignore pixels and curry the function to be called.
        """
        def tmp(masks, pixels, cell_fun):
            return trap_apply(masks, cell_fun=cell_fun)
        
        return partial(tmp, cell_fun=f)
    
    for f_name, f in cell_funs.items():
        if isfunction(f):

            args = getfullargspec(f).args
            if len(args) == 1:
                # function that applies f to m, an array of masks
                new_fun =  trap_apply_on_mask(f)
            else:
                # function that applies f to m and img, the trap_image
                new_fun =  partial(trap_apply, cell_fun=f)

            CELL_FUNS[f_name] = new_fun

    # Add automatically sklearn functions
    # TODO use heuristics to make execution more efficient
    SK_FUNS = [
                # "area",
                "area_bbox",
                "area_convex",
                "area_filled",
                "axis_major_length",
                "axis_minor_length",
                "bbox",
                # "centroid",
                "centroid_local",
                "centroid_weighted",
                "centroid_weighted_local",
                "coords_scaled",
                "coords",
                # "eccentricity",
                "equivalent_diameter_area",
                "euler_number",
                "extent",
                "feret_diameter_max",
                # "image",
                # "image_convex",
                # "image_filled",
                # "image_intensity",
                "inertia_tensor",
                "inertia_tensor_eigvals",
                "intensity_max",
                "intensity_mean",
                "intensity_min",
                "moments",
                "moments_central",
                "moments_hu",
                "moments_normalized",
                "moments_weighted",
                "moments_weighted_central",
                "moments_weighted_hu",
                "moments_weighted_normalized",
                "num_pixels",
                "orientation",
                "perimeter",
                "perimeter_crofton",
                "slice",
                "solidity",
    ]

    for fun_name in SK_FUNS:
        CELL_FUNS[fun_name] = partial(get_sk_features, feature=fun_name)
        
    # Add CellProfiler measurements
    for fun_name, f in get_core_measurements().items():
        CELL_FUNS[fun_name] = partial(wrap_cp_measure_features, fun=f)
            
    return CELL_FUNS


def load_trapfuns():
    """Load functions that are applied to an entire tile."""
    TRAP_FUNS = {
        f[0]: f[1]
        for f in getmembers(trap)
        if isfunction(f[1])
        and f[1].__module__.startswith("extraction.core.functions")
    }
    return TRAP_FUNS


def load_funs():
    """Combine all automatically loaded functions."""
    CELL_FUNS = load_cellfuns()
    TRAP_FUNS = load_trapfuns()
    # return dict of cell funs, dict of trap funs, and dict of both
    return CELL_FUNS, TRAP_FUNS, {**TRAP_FUNS, **CELL_FUNS}


def load_redfuns(
    additional_reducers: t.Optional[
        t.Union[t.Dict[str, t.Callable], t.Callable]
    ] = None,
) -> t.Dict[str, t.Callable]:
    """
    Load functions to reduce a multidimensional image by one dimension.

    Parameters
    ----------
    additional_reducers: function or a dict of functions (optional)
        Functions to perform the reduction.
    """
    RED_FUNS = {
        "max": np.nanmax,
        "mean": np.nanmean,
        "median": np.nanmedian,
        "div0": div0,
        "add": np.nansum,
        "None": None,
    }
    if additional_reducers is not None:
        if isinstance(additional_reducers, FunctionType):
            additional_reducers = [
                (additional_reducers.__name__, additional_reducers)
            ]
        RED_FUNS.update(additional_reducers)
    return RED_FUNS

from functools import partial

# Functional solutions to complex problems

def get_sk_features(masks: np.ndarray, pixels: np.ndarray, feature: str):
    """
    Freeze the sklearn function to use feature.
    """
    return [regionprops_table(
            mask,
            intensity_image=pixels,
            properties=(feature,),
            cache=False,
        )[feature][0] for mask in masks.astype(np.uint8)] # Assumes masks.dims=(N,Y,X)
    

def wrap_cp_measure_features(masks:np.ndarray, pixels = np.ndarray, fun:t.Callable=None)->t.Callable:
    results = [{k:v[0] for k,v in fun(m, pixels).items()} for m in masks.astype(np.uint8)]
    return results
    
