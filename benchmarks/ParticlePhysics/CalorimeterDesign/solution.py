"""Baseline: uniform layer structure (2mm Pb + 4mm scint × 30)."""
import numpy as np
def design_calorimeter(n_layers, max_total_length):
    return {
        "passive_thicknesses_mm": np.full(n_layers, 2.0),
        "active_thicknesses_mm": np.full(n_layers, 4.0),
    }
