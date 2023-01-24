#!/usr/bin/env python3
import numpy as np
import pandas as pd
from postprocessor.core.processes.interpolate import (
    interpolate,
    interpolateParameters,
)


def dummy_signal_array(n_cells, n_tps):
    """Creates dummy signal array, i.e. increasing gradient"""
    signal = np.array([np.linspace(1, 2, n_tps) for _ in range(n_cells)])
    return signal


def test_dummy_signal_array():
    ds = dummy_signal_array(5, 10)
    # Check dimensions
    assert ds.shape[0] == 5
    assert ds.shape[1] == 10


def randomly_add_na(input_array, num_of_na):
    """Randomly replaces a 2d numpy array with NaNs, number of NaNs specified"""
    input_array.ravel()[
        np.random.choice(input_array.size, num_of_na, replace=False)
    ] = np.nan
    return input_array


def test_interpolate():
    dummy_array = dummy_signal_array(5, 10)
    # Poke holes so interpolate can fill
    holey_array = randomly_add_na(dummy_array, 15)

    dummy_signal = pd.DataFrame(dummy_array)
    holey_signal = pd.DataFrame(holey_array)

    interpolate_runner = interpolate(interpolateParameters.default())
    interpolated_signal = interpolate_runner.run(holey_signal)

    subtr = interpolated_signal - dummy_signal
    # Check that interpolated values are the ones that exist in the dummy
    assert np.nansum(subtr.to_numpy()) == 0
    # TODO: Check that if there are NaNs remaining after interpolation, they
    # are at the ends
