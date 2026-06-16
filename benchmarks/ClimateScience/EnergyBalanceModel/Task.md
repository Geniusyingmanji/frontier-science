# EnergyBalanceModel — calibrate a 1D climate model to observed temperatures

## Scientific background
The Budyko-Sellers 1D Energy Balance Model (EBM) represents latitude-dependent climate:
incoming solar S(x)·(1-α) is balanced by outgoing longwave A+BT and poleward heat transport
D·∇²T. With ice-albedo feedback (α jumps at the ice line), the model has bistable equilibria
(ice-free Earth vs snowball). Calibrating the 7 parameters to match observed zonal-mean
temperatures is an underdetermined inverse problem with physical degeneracies.

Reference: North, J. Atmos. Sci. 32, 2033 (1975); Budyko 1969.

## Your task
```python
def calibrate_ebm(T_obs):
    \"\"\"Given observed zonal-mean temperatures (45,) from equator to pole,
    return parameter vector [A, B, D, alpha_ice, alpha_ocean, T_ice, S_multiplier].\"\"\"
```

## Scoring
`score = clip((RMSE_baseline - RMSE_found) / RMSE_baseline, 0, 1)` vs textbook params.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
