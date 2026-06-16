# MOSFETDoping — optimize doping profile for maximum on/off ratio

## Scientific background
In a MOSFET, the doping profile N(x) along the channel determines the electrostatic
potential barrier that controls current flow. Optimizing this profile to maximize the
Ion/Ioff current ratio (minimizing leakage while maintaining drive current) is the core
challenge of sub-100nm transistor design. The problem involves solving the 1D Poisson
equation self-consistently with carrier statistics.

Reference: Sze & Ng, Physics of Semiconductor Devices (Wiley, 2007).

## Your task
```python
def design_doping(n_points, channel_length):
    \"\"\"Return doping concentration profile (n_points,) in /m^3 range [1e15, 1e20].\"\"\"
```

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
