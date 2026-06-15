# LatticeSVP — find short lattice vectors

## Scientific background

The Shortest Vector Problem (SVP) on integer lattices underpins post-quantum cryptography
(lattice-based schemes like Kyber, Dilithium). Given a basis B of a lattice L, find a
nonzero vector v ∈ L with minimum Euclidean norm. This is NP-hard to approximate within
certain factors. LLL reduction gives polynomial-time 2^(n/2) approximation.

## Your task

```python
def find_short_vector(B):
    """Given integer basis matrix B (n×n), return a nonzero lattice vector v (n,)
    with small norm. v must be an integer combination of B's rows."""
```

## Scoring

Approximation ratio = ||v_found|| / ||v_shortest_known||. Closer to 1.0 is better.

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
