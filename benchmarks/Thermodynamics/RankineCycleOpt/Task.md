# RankineCycleOpt — optimize steam power cycle parameters

## Scientific background
The Rankine cycle powers most of the world's electricity. Optimizing boiler pressure,
superheat temperature, condenser pressure, and reheat strategy to maximize thermal
efficiency (subject to metallurgical and moisture constraints) is classic thermodynamic
engineering. Modern ultra-supercritical plants reach η ≈ 46-47%.

Reference: Çengel & Boles, Thermodynamics (9th ed, 2019) Ch. 10.

## Your task
```python
def optimize_rankine():
    \"\"\"Return [P_boiler_MPa (5-30), T_superheat_C (400-620),
    P_condenser_kPa (3-15), reheat_fraction (0-1)].\"\"\"
```

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
