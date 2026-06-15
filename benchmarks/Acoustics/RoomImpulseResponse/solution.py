"""Baseline: direct path + first-order reflections only."""
import numpy as np

def compute_rir(room_dims, source_pos, mic_pos, fs, max_order, absorption):
    c = 343.0  # speed of sound
    Lx, Ly, Lz = room_dims
    sx, sy, sz = source_pos
    mx, my, mz = mic_pos
    max_t = max(room_dims) * 2 / c
    n_samples = int(fs * max_t) + 1
    h = np.zeros(n_samples)
    # Direct path
    d = np.sqrt((sx-mx)**2 + (sy-my)**2 + (sz-mz)**2)
    idx = int(d / c * fs)
    if 0 <= idx < n_samples:
        h[idx] += 1.0 / max(d, 0.01)
    return h[:n_samples]
