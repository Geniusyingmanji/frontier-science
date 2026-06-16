"""Baseline: 20% perturbed HH parameters (poor fit to target trace)."""
import numpy as np

def fit_hh():
    """Return HH parameters as dict. Goal: match the hidden target voltage trace."""
    return {
        "g_Na": 144.0,   # true: 120 (20% high)
        "g_K": 43.2,     # true: 36 (20% high)
        "g_L": 0.36,     # true: 0.3 (20% high)
        "E_Na": 60.0,    # true: 50 (shifted)
        "E_K": -92.4,    # true: -77 (shifted)
        "E_L": -65.3,    # true: -54.4 (shifted)
        "Cm": 1.2,       # true: 1.0 (20% high)
        "I_ext": 12.0,   # true: 10 (20% high)
    }
