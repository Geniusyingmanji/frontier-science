# LidDrivenCavity — solve 2D Navier-Stokes in a square cavity

## Scientific background

The lid-driven cavity flow is the canonical benchmark for incompressible Navier-Stokes
solvers: a square domain with a moving top wall (u=1, v=0) and stationary other walls.
At Re=100-1000, vortex structures form. Ghia et al. (1982) tabulated reference velocity
profiles that remain the gold standard for CFD validation.

## Your task

```python
def solve_cavity(Re, N):
    """Solve the 2D lid-driven cavity at Reynolds number Re on an NxN grid.
    Return (u, v, p) where u,v are (N,N) velocity fields and p is (N,N) pressure."""
```

## Scoring

Mean L2 error of the centerline velocity profiles (u along vertical, v along horizontal)
compared to Ghia et al. tabulated values. Scored as gap-closed from Stokes (Re=0) baseline.

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU, < 30s. Do not read `verification/`.
