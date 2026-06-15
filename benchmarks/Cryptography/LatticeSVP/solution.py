"""Baseline: return the shortest row of the basis (no reduction)."""
import numpy as np

def find_short_vector(B):
    B = np.array(B, dtype=float)
    norms = np.linalg.norm(B, axis=1)
    return B[np.argmin(norms)].astype(int)
