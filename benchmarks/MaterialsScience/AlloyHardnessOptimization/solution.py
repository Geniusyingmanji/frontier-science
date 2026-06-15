"""Baseline: random equimolar compositions."""
import numpy as np

def design_alloy(n_elements, element_names, predict_hardness, n_candidates):
    rng = np.random.default_rng(0)
    comps = rng.dirichlet(np.ones(n_elements), n_candidates)
    return comps
