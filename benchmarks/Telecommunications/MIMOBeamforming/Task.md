# MIMOBeamforming — design precoding matrix to maximize spectral efficiency

## Scientific background
In MIMO wireless communications, a transmitter with N_tx antennas sends data to a receiver
with N_rx antennas through a fading channel H. The precoding matrix W shapes the transmitted
signal to maximize spectral efficiency (bits/s/Hz). With a total power constraint, SVD
precoding is optimal, but per-antenna constraints (practical for massive MIMO) make the
problem non-trivially constrained. This is the core of 5G/6G beamforming.

Reference: Tse & Viswanath, Fundamentals of Wireless Communication (Cambridge, 2005).

## Your task
```python
def design_precoder(H, n_tx, n_rx, snr_db):
    \"\"\"Given channel matrix H (n_rx, n_tx) complex, return precoding matrix W (n_tx, n_streams).
    Power: Tr(WW^H) will be normalized to n_tx. Maximize sum-rate.\"\"\"
```

## Scoring
Rate improvement over identity precoder, normalized vs SVD-optimal.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
