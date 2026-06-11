"""Initial baseline for PoissonSolver2D (weak but valid).

Solves the discrete Poisson system with 50 Jacobi sweeps — nowhere near converged, so the
error is large. Edit this file to do better: a direct sparse solve of the 5-point system,
a higher-order (e.g. 9-point Mehrstellen) stencil, multigrid, or a spectral method.
"""

import numpy as np

MODES = [(1, 1, 1.0), (2, 3, 0.5), (3, 1, 0.3), (4, 2, 0.2)]   # given by the task (the RHS f)


def _f(n):
    h = 1.0 / (n + 1)
    xs = (np.arange(1, n + 1)) * h
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    f = np.zeros_like(X)
    for k, m, c in MODES:
        f += c * ((k * np.pi) ** 2 + (m * np.pi) ** 2) * np.sin(k * np.pi * X) * np.sin(m * np.pi * Y)
    return f


def solve_poisson(n: int) -> np.ndarray:
    h = 1.0 / (n + 1)
    u = np.zeros((n + 2, n + 2))
    rhs = np.zeros((n + 2, n + 2))
    rhs[1:-1, 1:-1] = _f(n) * h * h
    for _ in range(50):  # far from converged
        u[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                                + rhs[1:-1, 1:-1])
    return u[1:-1, 1:-1]
