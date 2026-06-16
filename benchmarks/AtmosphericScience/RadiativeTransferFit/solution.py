"""Baseline: isothermal profile at 250K."""
import numpy as np
def retrieve_profile(observed_radiances, n_layers, n_channels):
    return np.full(n_layers, 250.0)
