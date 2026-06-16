"""Oracle: magnetic mirror confinement optimization.

Agent places N coils along z-axis to create a mirror field. Oracle computes on-axis B(z)
from Biot-Savart (analytical for circular loops), then evaluates confinement quality:
mirror ratio, confined fraction, and minimum-B stability (field-line curvature at midplane).
"""
import numpy as np

MU0 = 4e-7 * np.pi
COIL_RADIUS = 0.5  # meters (fixed)
N_COILS = 8
Z_RANGE = [-2.0, 2.0]  # meters
MIN_SEPARATION = 0.15  # meters between coil centers
I_MIN, I_MAX = 0.1, 10.0  # MA

def _bz_on_axis(z, coil_z, coil_I):
    """On-axis B_z from N circular loops: B = mu0 I a^2 / (2(a^2 + (z-z0)^2)^(3/2))."""
    a = COIL_RADIUS
    Bz = np.zeros_like(z)
    for zc, Ic in zip(coil_z, coil_I):
        r2 = a**2 + (z - zc)**2
        Bz += MU0 * Ic * 1e6 * a**2 / (2 * r2**1.5)  # Ic in MA
    return Bz

def evaluate(design_mirror):
    try:
        result = design_mirror(N_COILS, Z_RANGE, MIN_SEPARATION, I_MIN, I_MAX)
        coil_z = np.asarray(result["coil_positions_z"], dtype=float)
        coil_I = np.asarray(result["coil_currents"], dtype=float)
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if coil_z.shape != (N_COILS,) or coil_I.shape != (N_COILS,):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "bad shape", "feasibility_rate": 0.0}
    # Constraint: coils within range and separated
    coil_z = np.clip(coil_z, Z_RANGE[0], Z_RANGE[1])
    coil_I = np.clip(coil_I, I_MIN, I_MAX)
    sorted_idx = np.argsort(coil_z)
    coil_z_s = coil_z[sorted_idx]
    for i in range(len(coil_z_s) - 1):
        if coil_z_s[i+1] - coil_z_s[i] < MIN_SEPARATION * 0.9:
            return {"combined_score": 0.0, "valid": 0.0, "error_message": "coils too close", "feasibility_rate": 0.0}
    # Compute field profile
    z = np.linspace(Z_RANGE[0], Z_RANGE[1], 500)
    Bz = _bz_on_axis(z, coil_z, coil_I)
    if np.any(Bz <= 0):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "B <= 0 on axis", "feasibility_rate": 0.0}
    B_max = np.max(Bz)
    B_min = np.min(Bz)
    if B_min < 1e-6:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "B_min too small", "feasibility_rate": 0.0}
    # Mirror ratio and confined fraction
    R = B_max / B_min
    f_confined = 1.0 - 1.0 / R  # isotropic pitch-angle distribution
    # Minimum-B stability: check d²B/dz² > 0 at the B_min location (good curvature)
    i_min = np.argmin(Bz)
    if 2 <= i_min <= len(Bz) - 3:
        dz = z[1] - z[0]
        d2B = (Bz[i_min+1] - 2*Bz[i_min] + Bz[i_min-1]) / dz**2
        stability = 1.0 if d2B > 0 else 0.3  # penalty for bad curvature
    else:
        stability = 0.5  # min at edge, ambiguous
    # Uniformity in central region
    center_mask = (z > -0.5) & (z < 0.5)
    B_center = Bz[center_mask]
    uniformity = 1.0 - np.std(B_center) / np.mean(B_center) if len(B_center) > 0 else 0.0
    uniformity = max(0.0, uniformity)
    # Combined score
    raw = f_confined * stability * (0.7 + 0.3 * uniformity)
    # Normalize: simple 2-coil mirror gives ~0.35; good design ~0.82
    baseline = 0.35; sota = 0.82
    score = max(0.0, min(1.0, (raw - baseline) / (sota - baseline)))
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "mirror_ratio": round(R, 3), "f_confined": round(f_confined, 4),
            "stability": round(stability, 2), "raw_quality": round(raw, 4)}
