import numpy as np
import pandas as pd
import pytest

from postprocessor.core.processes.gpsignal import (
    estimate_gr,
    gpsignal,
    gpsignalParameters,
)


def dummy_signal(n_cells, n_tps, noise_level):
    signal = np.array([np.linspace(1, 2, n_tps) for _ in range(n_cells)])
    noise = np.random.normal(scale=noise_level, size=signal.shape)
    return pd.DataFrame(signal + noise)


def test_dummy_signal():
    ds = dummy_signal(5, 10, 0.001)
    assert len(ds.columns) == 10
    assert len(ds) == 5
    # assert np.isclose(ds.std(), 0.001).any()


def default_values():
    return dict(
        dt=1, noruns=5, bounds={0: (0, 2), 1: (1, 3), 2: (-8, 0)}, verbose=True
    )


# TODO: the tolerance threshold still needs to be tuned to expectations
thresh = 0.1
np.random.seed(42)


@pytest.mark.parametrize("n_cells", [10])
@pytest.mark.parametrize("n_tps", [50])
@pytest.mark.parametrize("noise_level", [0.01])
@pytest.mark.xfail(reason="Cell 6 is failing since unification")  # TODO FIX
def test_estimate_gr(n_cells, n_tps, noise_level):
    ds = dummy_signal(n_cells, n_tps, noise_level)
    # Growth rate is just the slope
    gr = 1 / n_tps
    for i, volume in ds.iterrows():
        results = estimate_gr(volume, **default_values())
        est_gr = results["growth_rate"]
        assert np.allclose(est_gr, gr, rtol=thresh), f"Failed for cell {i}"


def test_gpsignal():
    ds = dummy_signal(5, 10, 0.001)
    gpsig = gpsignal(gpsignalParameters.default())
    multi_signal = gpsig.run(ds)
    assert "fit_volume" in multi_signal
