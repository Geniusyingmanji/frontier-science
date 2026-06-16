# HartreeFockSCF — optimize molecular orbital coefficients for H2

## Scientific background
The Hartree-Fock (HF) method finds the best single-determinant wavefunction by optimizing
molecular orbital coefficients to minimize the electronic energy. For H2 in the STO-3G basis,
this is a 2-parameter problem on the Stiefel manifold (orthonormality constraint). The energy
is a quartic function of coefficients via two-electron integrals. SCF convergence issues
(oscillation, wrong solution) are well-documented challenges.

Reference: Szabo & Ostlund, Modern Quantum Chemistry (1996) Ch. 3.

## Your task
```python
def optimize_orbitals(H_core, S, ERI, V_nn):
    \"\"\"Given one-electron Hamiltonian (2,2), overlap (2,2), two-electron integrals (2,2,2,2),
    and nuclear repulsion, return orbital coefficients (2,) for 1 occupied MO.
    Lower HF energy = higher score. Variational principle guarantees E >= E_HF_exact.\"\"\"
```

## Scoring
`score = clip((E_baseline - E_found) / (E_baseline - E_HF_exact), 0, 1)`
Equal-mix orbital → score 0. Exact HF solution E = -1.1167 Ha → score 1.0.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
