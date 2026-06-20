"""Baseline: zero control amplitudes (free ZZ evolution → not CNOT)."""
import numpy as np
def design_pulse(n_steps, n_controls, dim):
    return np.zeros((n_steps, n_controls))
