"""Gaussian process fit of a Signal."""
import logging

import gaussianprocessderivatives as gp
import numpy as np
import pandas as pd
from agora.abc import ParametersABC

from postprocessor.core.abc import PostProcessABC


def estimate_gr(volume, dt, noruns, bounds, verbose):
    """
    Parameters
    ----------

    volume : pd.Series
        The volume series of a given cell
    dt : float
        The time interval in hours
    noruns : int
        The number of runs for optimisation
    bounds : dict
        The hyperparameter bounds used for optimisation
    verbose : bool
        If True, prints results

    Returns
    -------
    """
    volume = volume.values
    n = len(volume)
    idx = np.arange(n)
    t = idx * dt
    y = volume[volume > 0]
    x = t[volume > 0]
    idx = idx[volume > 0]
    # Fit the GP
    mg = gp.maternGP(bounds, x, y)
    mg.findhyperparameters(noruns=noruns)
    if verbose:
        mg.results()  # Prints out the hyperparameters
    mg.predict(x, derivs=2)  # Saves the predictions to object
    # Save the predictions to a csv file so they can be reused
    results = dict(
        time=mg.x,
        volume=mg.y,
        fit_time=mg.xnew,
        fit_volume=mg.f,
        growth_rate=mg.df,
        d_growth_rate=mg.ddf,
        volume_var=mg.fvar,
        growth_rate_var=mg.dfvar,
        d_growth_rate_var=mg.ddfvar,
    )
    for name, value in results.items():
        results[name] = np.full((n,), np.nan)
        results[name][idx] = value
    return results


# Give that to a writer: NOTE the writer needs to learn how to write the
# output of a process that returns multiple signals like this one does.


class gpsignalParameters(ParametersABC):
    """
    Parameters
    ----------
        dt : float
            The time step between time points, in minutes
        noruns : int
            The number of times the optimisation is tried
        bounds : dict
            Hyperparameter bounds for the Matern Kernel
        verbose : bool
            Determines whether to print hyperparameter results
    """

    _defaults = dict(
        dt=5,
        noruns=5,
        bounds={0: (-2, 3), 1: (-2, 1), 2: (-4, -1)},
        verbose=False,
    )


class gpsignal(PostProcessABC):
    """Gaussian process fit of a Signal."""

    def __init__(self, parameters: gpsignalParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        results = signal.apply(
            lambda x: estimate_gr(x, **self.parameters.to_dict()),
            result_type="expand",
            axis=1,
        )
        multi_signal = {
            name: pd.DataFrame(
                np.vstack(results[name]),
                index=signal.index,
                columns=signal.columns,
            )
            for name in results.columns
        }
        return multi_signal
