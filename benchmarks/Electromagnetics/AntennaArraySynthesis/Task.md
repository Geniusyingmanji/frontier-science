# AntennaArraySynthesis — design array weights to minimize sidelobe level

## Scientific background
A uniform linear antenna array of N elements with spacing d produces a radiation pattern
whose shape depends on the complex excitation weights. Minimizing the peak sidelobe level
(PSLL) while maintaining a specified mainlobe width is the classic array synthesis problem.
The Dolph-Chebyshev weights achieve the theoretical minimum PSLL for a given mainlobe width
(Chebyshev equiripple), but discovering them requires domain knowledge.

Reference: Dolph, Proc. IRE 34, 335 (1946); Van Trees, Optimum Array Processing (2002).

## Your task
```python
def design_array(n_elements, d_lambda, mainlobe_width_deg):
    \"\"\"Return complex excitation weights (n_elements,) for a ULA.
    Goal: minimize peak sidelobe level outside the mainlobe.\"\"\"
```

## Scoring
`score = clip((PSLL_uniform - PSLL_found) / (PSLL_uniform - PSLL_chebyshev), 0, 1)`
Lower PSLL (more negative dB) = better. Uniform → -13.3 dB, Chebyshev → -30 to -40 dB.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
