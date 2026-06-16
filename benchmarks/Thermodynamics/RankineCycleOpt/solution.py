"""Baseline: simple Rankine cycle (no reheat, moderate conditions)."""
import numpy as np
def optimize_rankine():
    # [P_boiler_MPa, T_superheat_C, P_condenser_kPa, reheat_fraction]
    return np.array([10.0, 500.0, 10.0, 0.0])
