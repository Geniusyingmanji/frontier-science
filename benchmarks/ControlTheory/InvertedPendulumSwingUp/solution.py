"""Baseline: simple bang-bang (pushes right always)."""
import numpy as np

def swing_up_controller(state, t, dt):
    return 5.0  # constant push
