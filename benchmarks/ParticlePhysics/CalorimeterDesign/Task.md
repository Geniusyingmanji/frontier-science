# CalorimeterDesign — optimize sampling calorimeter for energy resolution

## Scientific background
Electromagnetic calorimeters in particle physics measure photon/electron energies via
shower development. A sampling calorimeter alternates dense absorber (Pb) and active
detector (scintillator) layers. The energy resolution σ/E depends on sampling fraction,
layer geometry, and shower containment. Optimizing the layer structure (possibly non-uniform)
to minimize the stochastic resolution term is critical for detector design (ATLAS, CMS).

Reference: Wigmans, Calorimetry (Oxford, 2017); Fabjan & Gianotti, Rev. Mod. Phys. 75, 1243 (2003).

## Your task
```python
def design_calorimeter(n_layers, max_total_length):
    \"\"\"Return dict with 'passive_thicknesses_mm' (30,) in [0.5, 5.0] and
    'active_thicknesses_mm' (30,) in [1.0, 10.0]. Total length <= 500 mm.
    Minimize energy resolution σ/E at 10 GeV.\"\"\"
```

## Scoring
`score = clip((σ_uniform - σ_found) / (σ_uniform - σ_optimal), 0, 1)`
Uniform (2mm Pb + 4mm scint): σ ≈ 3.8%. Optimized graded design: σ ≈ 1.6%.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
