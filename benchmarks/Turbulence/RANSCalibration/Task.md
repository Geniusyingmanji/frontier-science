# RANSCalibration — calibrate turbulence model constants to match DNS data

## Scientific background
The k-ε turbulence model uses 5 empirical constants (C_μ, C_ε1, C_ε2, σ_k, σ_ε) calibrated
to match high-fidelity DNS data. The coupled nonlinear k-ε ODEs for a boundary layer are
stiff and the constants interact non-linearly: small changes in C_ε2 dramatically affect the
dissipation rate, which feeds back into k through production. Calibrating simultaneously
against velocity AND turbulent kinetic energy profiles is the standard challenge in
computational turbulence modeling.

Reference: Launder & Sharma, Letters in Heat & Mass Transfer 1, 131 (1974); Wilcox 2006.

## Your task
```python
def calibrate_rans():
    \"\"\"Return dict with C_mu, C_e1, C_e2, sigma_k, sigma_e constants.\"\"\"
```

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
