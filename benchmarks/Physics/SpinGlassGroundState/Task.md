# SpinGlassGroundState — find low-energy states of an Ising spin glass

## Scientific background

The Sherrington–Kirkpatrick (SK) model is the canonical mean-field spin glass: `N` Ising
spins `s_i ∈ {−1,+1}` with symmetric Gaussian couplings `J_ij ~ N(0,1)/√N`. The energy is

```
E(s) = −∑_{i<j} J_ij s_i s_j = −½ sᵀ J s
```

Finding the ground state (minimum energy) is NP-hard in general and underlies models of
disordered magnets, combinatorial optimization (Max-Cut), and the physics of glasses.

## Your task

Edit **`solution.py`** so it defines:

```python
def solve(n: int, J: "np.ndarray") -> "np.ndarray":
    """Given N and the (N, N) symmetric coupling matrix J, return a length-N
    array of spins in {-1, +1} with energy as low as possible."""
```

The evaluator builds several fixed instances, calls `solve(n, J)`, and compares your
configuration's energy to the exact ground state (precomputed by full enumeration).

## Scoring

Per instance, with `E_ref` the energy of the all-`+1` configuration and `E_min` the exact
ground state:

```
score = clip( (E_ref − E_found) / (E_ref − E_min), 0, 1 )
```

`combined_score` is the mean over instances. A trivial guess scores ~0; reaching the exact
ground state scores 1.0.

## Rules

- Only edit `solution.py`. Keep the `solve(n, J)` signature and the ±1 output contract.
- Deterministic is preferred; keep each instance under a few seconds. CPU only, no network.
- `numpy` and `scipy` only. Do not read anything under `verification/` or `frontier_eval/`.
