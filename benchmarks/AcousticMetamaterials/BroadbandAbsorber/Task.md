# BroadbandAbsorber — design acoustic metamaterial for broadband sound absorption

## Scientific background
Acoustic metamaterial absorbers use arrays of sub-wavelength Helmholtz resonators to
achieve broadband absorption. Each resonator has a narrow bandwidth; achieving α > 0.9
over 200-2000 Hz with total depth < 120mm requires carefully tuning the geometric
parameters (cavity depths, neck dimensions) so their impedances produce near-perfect
absorption across the target band. This is the state-of-the-art in noise control.

Reference: Jiménez et al., APL 109, 121902 (2016); Li & Assouar, APL 108, 063502 (2016).

## Your task
```python
def design_absorber(n_resonators, freq_range):
    \"\"\"Return dict with 'cavity_depths_mm' (8,), 'neck_lengths_mm' (8,),
    'neck_radii_mm' (8,), 'cavity_radii_mm' (8,). All in mm.
    Max total depth 120mm. Goal: maximize mean absorption 200-2000 Hz.\"\"\"
```

## Scoring
Mean absorption coefficient improvement over linearly-spaced baseline, toward SoTA (α≈0.92).

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
