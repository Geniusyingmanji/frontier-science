"""Baseline: random normalized state."""
import numpy as np

def find_ground_state(H, n_qubits, n_layers):
    dim = 2 ** n_qubits
    psi = np.random.default_rng(0).standard_normal(dim) + 0j
    psi /= np.linalg.norm(psi)
    return psi
