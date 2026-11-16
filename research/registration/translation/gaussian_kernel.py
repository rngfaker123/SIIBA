"""Separable discrete Gaussian used by the archived shift/blur controls."""
import numpy as np


def kernel(sigma, radius):
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    values = np.exp(-positions * positions / (2 * sigma * sigma))
    return values / values.sum()
