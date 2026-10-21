import numpy as np

from src.metrics import mape_nonzero, psi, wape


def test_mape_ignores_zero_days():
    actual = [100, 0, 200]
    predicted = [110, 50, 180]
    # errors on the two trading days: 10% and 10%
    assert round(mape_nonzero(actual, predicted), 6) == 10.0


def test_wape_counts_zero_days():
    actual = [100, 0, 100]
    predicted = [100, 50, 100]
    assert wape(actual, predicted) == 25.0


def test_perfect_forecast_has_zero_error():
    actual = [10, 20, 30]
    assert mape_nonzero(actual, actual) == 0
    assert wape(actual, actual) == 0


def test_psi_is_zero_for_identical_data():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) < 1e-6


def test_psi_detects_a_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    shifted = rng.normal(1.5, 1, 2000)
    assert psi(ref, shifted) > 0.2