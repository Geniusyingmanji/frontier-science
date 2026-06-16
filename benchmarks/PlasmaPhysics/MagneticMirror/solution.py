"""Baseline: uniform coil spacing and uniform low current (weak mirror)."""
import numpy as np
def design_mirror(n_coils, z_range, min_sep, i_min, i_max):
    positions = np.linspace(z_range[0] + 0.3, z_range[1] - 0.3, n_coils)
    currents = np.full(n_coils, i_min)  # minimum current everywhere
    return {"coil_positions_z": positions, "coil_currents": currents}
