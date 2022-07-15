import numpy as np
import pandas as pd
from agora.abc import ParametersABC
from scipy.signal import savgol_filter

from postprocessor.core.abc import PostProcessABC


class savgolParameters(ParametersABC):
    """
    Parameters

        window : int (odd)
            Window length of datapoints. Must be odd and smaller than x
        polynom : int
            The order of polynom used. Must be smaller than the window size
    """

    _defaults = {"window": 7, "polynom": 3}


class savgol(PostProcessABC):
    """
    Apply Savitzky-Golay filter (works with NaNs, but it might return
    NaN regions).
    """

    def __init__(self, parameters: savgolParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        try:
            post_savgol = pd.DataFrame(
                savgol_filter(
                    signal, self.parameters.window, self.parameters.polynom
                ),
                index=signal.index,
                columns=signal.columns,
            )
        except Exception as e:
            print(e)

            def savgol_on_srs(x):
                return self.non_uniform_savgol(
                    x.index,
                    x.values,
                    self.parameters.window,
                    self.parameters.polynom,
                )

            post_savgol = signal.apply(savgol_on_srs, 1).apply(pd.Series)
        return post_savgol


def non_uniform_savgol(x, y, window: int, polynom: int):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    source: https://dsp.stackexchange.com/a/64313

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    _raiseif(
        len(x) != len(y),
        '"x" and "y" must be of the same size',
        ValueError,
    )
    _raiseif(
        len(x) < window,
        "The data size must be larger than the window size",
        ValueError,
    )
    _raiseif(
        not isinstance(window, int),
        '"window" must be an integer',
        TypeError,
    )
    _raiseif(window % 2, 'The "window" must be an odd integer', ValueError)

    _raiseif(
        not isinstance(polynom, int),
        '"polynom" must be an integer',
        TypeError,
    )

    _raiseif(
        polynom >= window,
        '"polynom" must be less than "window"',
        ValueError,
    )

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))  # Matrix
    tA = np.empty((polynom, window))  # Transposed matrix
    t = np.empty(window)  # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed


def _raiseif(cond, msg="", exc=AssertionError):
    if cond:
        raise exc(msg)
