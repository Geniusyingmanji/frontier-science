"""Baseline: equal generation share (ignores costs and constraints)."""
import numpy as np
def solve_opf(n_bus, n_gen, demand, gen_pmax, gen_pmin, cost_a, cost_b, lines, line_limits):
    total_demand = sum(demand)
    return np.full(n_gen, total_demand / n_gen)
