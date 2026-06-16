"""Baseline: all bars at maximum cross-section (heavy but safe)."""
import numpy as np
def design_truss(n_bars):
    return np.full(n_bars, 35.0)  # max area
