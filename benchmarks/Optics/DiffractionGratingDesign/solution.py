import numpy as np
def design_grating(wavelength, period, n_substrate, target_order, n_grooves):
    return np.full(n_grooves, wavelength / (2 * n_substrate))
