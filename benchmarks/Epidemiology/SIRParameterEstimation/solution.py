"""Baseline: fixed default parameters."""
import numpy as np

def estimate_sir(t_obs, I_obs, N_pop):
    return (0.3, 0.1, N_pop - I_obs[0], I_obs[0], 0.0)
