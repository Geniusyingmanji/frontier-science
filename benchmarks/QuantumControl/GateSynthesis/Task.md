# GateSynthesis — design control pulses to implement a quantum gate

## Scientific background
Quantum gate synthesis finds time-dependent control fields that steer a quantum system to
implement a target unitary transformation (here: CNOT gate on 2 qubits). The system has a
fixed ZZ coupling and 4 control channels (X, Y on each qubit). The control landscape has
many local traps, and the matrix exponential propagation creates a highly nonlinear map from
pulse amplitudes to gate fidelity. This is the core problem in quantum optimal control (GRAPE).

Reference: Khaneja et al., J. Magn. Reson. 172, 296 (2005); Glaser et al., EPJ D 69, 279 (2015).

## Your task
```python
def design_pulse(n_steps, n_controls, dim):
    \"\"\"Return (n_steps=50, n_controls=4) array of control amplitudes in [-5, 5].
    Goal: maximize gate fidelity with CNOT target.\"\"\"
```
## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
