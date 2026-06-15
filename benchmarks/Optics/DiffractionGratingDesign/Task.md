# DiffractionGratingDesign — optimize grating groove profile for maximum efficiency

## Scientific background
Diffraction gratings disperse light by wavelength. The groove profile (depth, shape, blaze angle)
determines efficiency into the desired order. Rigorous coupled-wave analysis (RCWA) computes
diffraction efficiency. Optimizing the profile for a target wavelength range and diffraction order
is critical for spectrometers, telecom, and laser systems.

## Your task
```python
def design_grating(wavelength, period, n_substrate, target_order, n_grooves):
    """Return groove depths (n_grooves,) defining the grating profile.
    Maximize 1st-order diffraction efficiency at the given wavelength."""
```

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
