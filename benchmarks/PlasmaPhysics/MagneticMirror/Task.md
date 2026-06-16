# MagneticMirror — optimize coil configuration for plasma confinement

## Scientific background
Magnetic mirrors confine charged particles by reflecting them from regions of high field
at the ends. The confinement quality depends on the mirror ratio R = B_max/B_min (higher R
traps more particles) and the field curvature at the midplane (minimum-B is needed for MHD
stability). Optimizing N coil positions and currents to maximize the confined fraction while
maintaining stability is a core problem in fusion plasma physics.

Reference: Chen, Intro to Plasma Physics (2016) Ch. 8; Post, Nucl. Fusion 27, 1579 (1987).

## Your task
```python
def design_mirror(n_coils, z_range, min_sep, i_min, i_max):
    \"\"\"Return dict with 'coil_positions_z' (8,) and 'coil_currents' (8,).
    Coils are circular loops at radius 0.5m. Maximize confinement quality.\"\"\"
```

## Scoring
Combined: confined_fraction × stability_factor × uniformity. Simple 2-coil mirror → 0;
optimized multi-coil minimum-B configuration → 1.0.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
