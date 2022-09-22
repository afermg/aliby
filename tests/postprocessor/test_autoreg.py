import numpy as np
import pandas as pd
import pytest

from postprocessor.core.processes.autoreg import autoreg, autoregParameters


def generate_sinusoids_df(
    time_axis,
    list_freqs,
):
    """Generate sinusoids and put them in a dataframe.

    Parameters ---------- time_axis : array_like     Time axis.
    list_freqs : list     List of frequencies for the sinusoids
    Examples -------- generate_sinusoids_df([0,1,2,3,4], [1,2,3])
    produces a dataframe containing 3 rows.  The first row has a
    sinusoid of frequency 1, the second has frequency 2, and the third
    has frequency 3. The time axis goes from 0 to 5.
    """
    sinusoids = np.array(
        [np.sin((2 * np.pi * freq) * time_axis) for freq in list_freqs]
    )
    return pd.DataFrame(sinusoids)


@pytest.mark.parametrize("time_axis", [np.arange(0, 10, 0.01)])
@pytest.mark.parametrize("list_freqs", [[1, 2, 3]])
def test_autoreg(
    time_axis,
    list_freqs,
):
    """Tests autoreg.

    Tests whether an autoreg runner can be initialised with default
    parameters and runs without errors.
    """
    dummy_signal = generate_sinusoids_df(time_axis, list_freqs)
    autoreg_runner = autoreg(autoregParameters.default())
    # freqs_df, power_df, order_df = autoreg_runner.run(dummy_signal)
    _, _, _ = autoreg_runner.run(dummy_signal)
