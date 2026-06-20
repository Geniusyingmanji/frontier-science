"""Baseline: standard k-epsilon constants."""
import numpy as np
def calibrate_rans():
    return {"C_mu": 0.09, "C_e1": 1.44, "C_e2": 1.92, "sigma_k": 1.0, "sigma_e": 1.3}
