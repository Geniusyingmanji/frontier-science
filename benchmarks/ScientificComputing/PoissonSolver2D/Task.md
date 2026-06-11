# PoissonSolver2D — design a more accurate Poisson solver

## Scientific background

The Poisson equation `−∇²u = f` is the workhorse of computational physics (electrostatics,
diffusion, incompressible flow projection, ...). The quality of a numerical solver — its
discretization order, linear solve, and conditioning — directly sets the accuracy/cost
tradeoff of large simulations. Improving the stencil and solve is a long-standing topic in
scientific computing.

## Problem

Solve `−∇²u = f` on the unit square `(0,1)²` with homogeneous Dirichlet boundary conditions
`u = 0`. The right-hand side is the fixed multi-mode field

```
f(x,y) = Σ_(k,m,c) c · ((kπ)² + (mπ)²) · sin(kπx) · sin(mπy)
         with (k,m,c) ∈ {(1,1,1.0), (2,3,0.5), (3,1,0.3), (4,2,0.2)}
```

(The exact solution is held out by the evaluator. Note `u = f/(2π²)` is **not** the solution
here — the modes have different eigenvalues.)

## Your task

Edit **`solution.py`** so it defines:

```python
def solve_poisson(n: int) -> "np.ndarray":
    """Return the (n, n) array of u at the interior grid points
    x_i = i*h, i=1..n, with h = 1/(n+1). Boundary values are 0."""
```

The evaluator calls `solve_poisson(49)` and measures the relative L2 error against the exact
solution on that grid.

## Scoring

With `E` the relative L2 error, `E_baseline` the weak Jacobi baseline error, and `E_ref` a
4th-order reference error:

```
combined_score = clip( (log10(E_baseline) − log10(E)) / (log10(E_baseline) − log10(E_ref)), 0, 1 )
```

So the initial Jacobi solver scores ~0, a converged 2nd-order direct solve scores ~0.48, and
a 4th-order (or spectral) solver reaches 1.0.

## Rules

- Only edit `solution.py`; keep the `solve_poisson(n)` signature and `(n, n)` output.
- Deterministic, CPU only, seconds-scale, no network. `numpy` and `scipy` only.
- Do not read anything under `verification/` or `frontier_eval/`.
