# RoomImpulseResponse — simulate room acoustics via image-source method

## Scientific background

The Room Impulse Response (RIR) characterizes how sound propagates in an enclosed space,
accounting for wall reflections, absorption, and path delays. The image-source method models
reflections by "mirroring" the source across walls. Accurate RIR simulation is critical for
speech processing, VR audio, and architectural acoustics.

## Your task

```python
def compute_rir(room_dims, source_pos, mic_pos, fs, max_order, absorption):
    """Return the impulse response h (1D array, length ~ fs * room_size/c).
    room_dims: (Lx, Ly, Lz) in meters. absorption: wall absorption coeff [0,1]."""
```

The evaluator compares your RIR against a high-order reference RIR (order 50) via SNR.

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
