"""Baseline: uniformly spaced design points."""
import numpy as np
def select_designs(k, d_lo, d_hi, n_params):
    return np.linspace(d_lo, d_hi, k)
