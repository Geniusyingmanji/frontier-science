# RadiativeTransferFit — retrieve atmospheric temperature from satellite radiances

## Scientific background
Satellite remote sensing retrieves atmospheric temperature profiles by fitting modeled
radiances to observed spectra. The forward model computes thermal IR emission through a
multi-layer atmosphere; the inverse problem (retrieval) is ill-posed and requires
regularization. Each spectral channel "sees" a different depth due to its optical
properties, providing vertical information.

Reference: Rodgers, Inverse Methods for Atmospheric Sounding (World Scientific, 2000).

## Your task
```python
def retrieve_profile(observed_radiances, n_layers, n_channels):
    \"\"\"Given 10 observed TOA radiances, return temperature profile (20,) in Kelvin.\"\"\"
```

## Scoring
RMSE of retrieved profile vs true profile, normalized against isothermal baseline.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
