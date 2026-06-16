"""Baseline: identity precoder (no beamforming)."""
import numpy as np
def design_precoder(H, n_tx, n_rx, snr_db):
    return np.eye(n_tx, n_rx, dtype=complex)
