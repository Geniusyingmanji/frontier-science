"""Baseline: all sources at center (poor coverage of target field)."""
import numpy as np
def place_sources(nx, ny, n_sources):
    return {
        "positions": np.full((n_sources, 2), 0.5),
        "strengths": np.ones(n_sources),
    }
