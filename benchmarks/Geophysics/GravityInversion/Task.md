# GravityInversion — recover subsurface density from surface gravity measurements

## Scientific background
Gravity inversion reconstructs subsurface density anomalies from surface gravity data.
The forward problem is linear (g = G·ρ) but the inverse is severely ill-posed: the kernel's
singular values decay rapidly and infinitely many density distributions explain the same
surface signal. Intelligent regularization (Tikhonov, L1, total variation) is required.
This underpins mineral exploration, basin modeling, and geodynamics.

Reference: Li & Oldenburg, Geophysics 63, 109 (1998); Talwani et al. 1959.

## Your task
```python
def invert_gravity(G_kernel, g_obs, nx, nz):
    \"\"\"Given the (n_obs, nx*nz) forward kernel and observed gravity (n_obs,),
    return density anomaly vector of shape (nx*nz,).\"\"\"
```

## Scoring
60% data fit (χ² reduction) + 40% structural similarity to true model.
Zero-model baseline = score 0.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
