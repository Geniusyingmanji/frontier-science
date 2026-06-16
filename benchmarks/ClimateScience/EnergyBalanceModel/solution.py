"""Baseline: textbook North & Coakley (1979) parameters."""
import numpy as np
def calibrate_ebm(T_obs):
    # [A, B, D, alpha_ice, alpha_ocean, T_ice, S_mult]
    return np.array([203.3, 2.09, 0.44, 0.62, 0.30, -10.0, 1.0])
