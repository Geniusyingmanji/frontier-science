"""Baseline: Stokes flow (Re=0 approximation — linear, no inertia)."""
import numpy as np

def solve_cavity(Re, N):
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    p = np.zeros((N, N))
    # Top wall boundary condition
    u[-1, :] = 1.0
    return u, v, p
