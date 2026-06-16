"""Baseline: guessed Arrhenius parameters (poor fit)."""
import numpy as np
def fit_kinetics(data, temperatures, t_eval):
    return {"A1": 1e10, "E1": 70000.0, "A2": 1e10, "E2": 70000.0}
