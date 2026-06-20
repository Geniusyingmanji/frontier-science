# OceanCurrentInversion — recover current field from drifter trajectories

## Your task
```python
def invert_currents(obs_traj, init_pos, nx, ny, domain):
    \"\"\"Given observed drifter trajectories (15, 51, 2) and initial positions,
    return dict with 'u' and 'v' fields (20, 20) m/s.\"\"\"
```
## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
