"""Oracle: fit overlapping Lorentzian peaks to a synthetic NMR spectrum.

Agent decomposes a spectrum into individual peaks. The challenge: peaks overlap heavily,
the number of peaks is unknown, and noise + baseline drift confound the fit.
"""
import numpy as np

N_POINTS = 512
N_PEAKS_TRUE = 8  # hidden

def _lorentzian(x, center, width, amplitude):
    return amplitude * width**2 / ((x - center)**2 + width**2)

def _generate_spectrum(seed=42):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, N_POINTS)  # ppm
    centers = rng.uniform(1, 9, N_PEAKS_TRUE)
    widths = rng.uniform(0.02, 0.15, N_PEAKS_TRUE)
    amplitudes = rng.uniform(0.5, 3.0, N_PEAKS_TRUE)
    spectrum = np.zeros(N_POINTS)
    for c, w, a in zip(centers, widths, amplitudes):
        spectrum += _lorentzian(x, c, w, a)
    # Add baseline drift + noise
    baseline = 0.1 * np.sin(2*np.pi*x/10) + 0.05
    noise = rng.normal(0, 0.05, N_POINTS)
    return x, spectrum + baseline + noise, centers, widths, amplitudes

_X, _SPEC, _C_TRUE, _W_TRUE, _A_TRUE = _generate_spectrum()

def evaluate(fit_spectrum):
    try:
        result = fit_spectrum(_X.copy(), _SPEC.copy())
        centers = np.asarray(result["centers"], dtype=float)
        widths = np.asarray(result["widths"], dtype=float)
        amplitudes = np.asarray(result["amplitudes"], dtype=float)
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    n_peaks = len(centers)
    if n_peaks < 1 or n_peaks > 20 or len(widths) != n_peaks or len(amplitudes) != n_peaks:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "inconsistent peak arrays", "feasibility_rate": 0.0}
    # Reconstruct spectrum
    recon = np.zeros(N_POINTS)
    for c, w, a in zip(centers, widths, amplitudes):
        recon += _lorentzian(_X, c, w, max(a, 0))
    # Residual
    residual = float(np.sqrt(np.mean((_SPEC - recon)**2)))
    spec_power = float(np.sqrt(np.mean(_SPEC**2)))
    # Baseline: single broad peak at center
    recon_base = _lorentzian(_X, 5.0, 2.0, 1.0)
    residual_base = float(np.sqrt(np.mean((_SPEC - recon_base)**2)))
    score = max(0.0, min(1.0, (residual_base - residual) / residual_base)) if residual_base > 1e-6 else 0.0
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "residual_rms": round(residual, 6), "n_peaks_found": n_peaks}
