"""Baseline: uniform weights (all ones) — PSLL = -13.3 dB."""
import numpy as np
def design_array(n_elements, d_lambda, mainlobe_width_deg):
    return np.ones(n_elements, dtype=complex)
