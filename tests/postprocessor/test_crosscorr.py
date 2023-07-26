import numpy as np
import pandas as pd
import pytest

from postprocessor.core.multisignal.crosscorr import (
    crosscorr,
    crosscorrParameters,
)


def generate_sinusoids_df(
    time_axis,
    num_replicates,
):
    t = time_axis
    ts = np.tile(t, num_replicates).reshape((num_replicates, len(t)))
    s = 3 * np.sin(
        2 * np.pi * ts + 2 * np.pi * np.random.rand(num_replicates, 1)
    )
    s_df = pd.DataFrame(s)
    return s_df


@pytest.mark.parametrize("time_axis", [np.linspace(0, 4, 200)])
@pytest.mark.parametrize("num_replicates", [333])
def test_crosscorr(
    time_axis,
    num_replicates,
):
    """Tests croscorr.

    Tests whether a crosscorr runner can be initialised with default
    parameters and runs without errors.
    """
    dummy_signal1 = generate_sinusoids_df(time_axis, num_replicates)
    dummy_signal2 = generate_sinusoids_df(time_axis, num_replicates)
    crosscorr_runner = crosscorr(crosscorrParameters.default())
    _ = crosscorr_runner.run(dummy_signal1, dummy_signal2)
