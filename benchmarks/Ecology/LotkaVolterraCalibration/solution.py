"""Baseline: return fixed default parameters (poor fit)."""
import numpy as np

def calibrate(t_obs, x_obs, y_obs):
    return (1.0, 0.5, 0.5, 1.0, x_obs[0], y_obs[0])
