# SeismicInversion — recover subsurface velocity from seismic data

## Scientific background

Full Waveform Inversion (FWI) recovers the subsurface velocity model from seismic
recordings. Given travel-time picks from a 1D layered model, the inverse problem is to
recover the velocity profile. This underpins oil/gas exploration and earthquake seismology.

## Your task

```python
def invert_seismic(travel_times, source_positions, receiver_positions, n_layers):
    """Given observed travel times and geometry, return velocity profile (n_layers,)."""
```

## Scoring

R² of predicted travel times (using recovered velocities) vs. observations.

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
