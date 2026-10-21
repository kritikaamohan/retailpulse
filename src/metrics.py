"""Metric helpers used in the forecasting notebook and for drift checks."""
import numpy as np


def mape_nonzero(actual, predicted):
    """MAPE (%) computed only on days where actual revenue is above zero."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual > 0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100)


def wape(actual, predicted):
    """Weighted absolute percentage error (%), valid on days with zero revenue."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.abs(actual - predicted).sum() / actual.sum() * 100)


def psi(reference, current, bins=10):
    """Population Stability Index of one feature (reference vs current)."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))