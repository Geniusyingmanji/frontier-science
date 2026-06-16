"""Baseline: linearly spaced resonators (mediocre broadband coverage)."""
import numpy as np
def design_absorber(n_resonators, freq_range):
    return {
        "cavity_depths_mm": np.linspace(10, 80, n_resonators),
        "neck_lengths_mm": np.full(n_resonators, 5.0),
        "neck_radii_mm": np.full(n_resonators, 2.0),
        "cavity_radii_mm": np.full(n_resonators, 15.0),
    }
