"""Baseline: constant population size (no bottleneck/expansion)."""
import numpy as np
def fit_demography(observed_sfs, n_sample):
    """Return [N_anc, N_bot, N_cur, T_start, T_end] — all in relative units."""
    return np.array([1.0, 1.0, 1.0, 0.1, 0.05])  # constant size
