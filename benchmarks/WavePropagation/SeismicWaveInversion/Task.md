# SeismicWaveInversion — reconstruct subsurface velocity from seismic waveforms

## Scientific background
Full Waveform Inversion (FWI) reconstructs subsurface velocity by matching synthetic
seismograms (computed via the 2D acoustic wave equation) to observed data. The forward
model — finite-difference time-domain (FDTD) wave propagation on an 80×60 grid — is
computationally expensive and creates a highly non-convex misfit landscape with cycle-
skipping (wrong local minima from phase ambiguity). FWI is the gold standard in
exploration geophysics and requires multi-scale, frequency-continuation strategies.

Reference: Virieux & Operto, Geophysics 74, WCC1 (2009); Tarantola, Geophysics 49, 1259 (1984).

## Your task
```python
def design_velocity(nx, nz):
    \"\"\"Return a velocity model (nz, nx) in m/s range [1000, 5000].
    Goal: produce synthetic seismograms that match the hidden observed data.\"\"\"
```

The evaluator runs FDTD wave propagation through your velocity model with 3 sources and
20 receivers, comparing the resulting seismograms to observations via waveform misfit.

## Scoring
`score = clip((misfit_homogeneous - misfit_found) / misfit_homogeneous, 0, 1)`
Homogeneous 2000 m/s → score 0. True velocity model → score 1.0.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
