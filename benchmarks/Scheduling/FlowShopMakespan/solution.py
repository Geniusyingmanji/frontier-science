"""Baseline: natural order (job 0, 1, 2, ... n-1) — typically poor."""
import numpy as np
def schedule_flowshop(processing_times, n_jobs, n_machines):
    return list(range(n_jobs))
