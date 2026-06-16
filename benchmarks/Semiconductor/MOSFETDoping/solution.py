"""Baseline: uniform doping (poor electrostatic control)."""
import numpy as np
def design_doping(n_points, channel_length):
    return np.full(n_points, 1e17)  # uniform 1e17 /m^3
