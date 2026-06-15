# VariationalEigensolver — optimize a parameterized quantum circuit ansatz

## Scientific background

The Variational Quantum Eigensolver (VQE) finds the ground-state energy of a Hamiltonian
by optimizing parameters of a quantum circuit ansatz. It is a leading candidate for quantum
advantage in chemistry and materials. Here we simulate classically: given a Hamiltonian matrix
H, find the state |ψ(θ)⟩ that minimizes ⟨ψ|H|ψ⟩ using a hardware-efficient ansatz.

## Your task

```python
def find_ground_state(H, n_qubits, n_layers):
    """Given Hamiltonian H (2^n × 2^n), return state vector psi (2^n,) complex.
    |psi| should be normalized. Minimize <psi|H|psi>."""
```

## Scoring

`score = clip((E_random - E_found) / (E_random - E_exact), 0, 1)` where E_exact is the
true ground-state energy. Mean over instances.

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
