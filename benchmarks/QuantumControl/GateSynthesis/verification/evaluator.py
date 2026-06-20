"""Oracle: quantum gate synthesis — find pulse sequence implementing a target unitary.

Agent designs a time-dependent Hamiltonian control sequence. Oracle propagates the time-
dependent Schrödinger equation and compares the resulting unitary to a target gate.
Gate fidelity = |Tr(U_target† U_achieved)|² / d² where d = dimension.
"""
import numpy as np
from scipy.linalg import expm

D = 4  # 2-qubit system
N_STEPS = 50
DT = 0.1  # time step

# Pauli matrices
I2 = np.eye(2, dtype=complex)
SX = np.array([[0,1],[1,0]], dtype=complex)
SY = np.array([[0,-1j],[1j,0]], dtype=complex)
SZ = np.array([[1,0],[0,-1]], dtype=complex)

# System Hamiltonian (fixed): ZZ coupling + X drive
H_ZZ = np.kron(SZ, SZ)
H_X1 = np.kron(SX, I2)
H_X2 = np.kron(I2, SX)
H_Y1 = np.kron(SY, I2)
H_Y2 = np.kron(I2, SY)
CONTROLS = [H_X1, H_X2, H_Y1, H_Y2]  # 4 control channels
N_CONTROLS = len(CONTROLS)

# Target gate: CNOT
_CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

def _propagate(amplitudes):
    """Time-ordered propagation under piecewise-constant Hamiltonian."""
    U = np.eye(D, dtype=complex)
    for t in range(N_STEPS):
        H = H_ZZ * 0.1  # fixed coupling
        for c in range(N_CONTROLS):
            H += amplitudes[t, c] * CONTROLS[c]
        U = expm(-1j * H * DT) @ U
    return U

def _gate_fidelity(U_target, U_achieved):
    """Average gate fidelity."""
    overlap = np.abs(np.trace(U_target.conj().T @ U_achieved))**2 / D**2
    return float(overlap)

def evaluate(design_pulse):
    try:
        amplitudes = np.asarray(design_pulse(N_STEPS, N_CONTROLS, D), dtype=float)
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if amplitudes.shape != (N_STEPS, N_CONTROLS):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": f"shape {amplitudes.shape} != ({N_STEPS},{N_CONTROLS})", "feasibility_rate": 0.0}
    amplitudes = np.clip(amplitudes, -5.0, 5.0)
    U = _propagate(amplitudes)
    fidelity = _gate_fidelity(_CNOT, U)
    # Baseline: zero control (free evolution under ZZ coupling)
    U_free = _propagate(np.zeros((N_STEPS, N_CONTROLS)))
    fid_base = _gate_fidelity(_CNOT, U_free)
    # Score
    score = max(0.0, min(1.0, (fidelity - fid_base) / (1.0 - fid_base))) if fid_base < 1.0 else 0.0
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "gate_fidelity": round(fidelity, 6), "baseline_fidelity": round(fid_base, 6)}
