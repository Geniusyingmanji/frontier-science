# ReactionMechanismFitting — fit Arrhenius kinetic parameters from concentration data

## Scientific background
A consecutive reaction A→B→C with Arrhenius kinetics k_i = A_i·exp(-E_i/RT) is observed
at multiple temperatures. The agent fits the 4 parameters (A₁, E₁, A₂, E₂) to noisy
concentration profiles. This is the prototypical hard inverse problem in chemical kinetics:
6+ orders of magnitude in A, strong A/E correlation ("compensation effect"), and stiff ODEs
near the Arrhenius knee create a rugged landscape.

Reference: Turányi & Tomlin, Analysis of Kinetic Reaction Mechanisms (Springer 2014).

## Your task
```python
def fit_kinetics(data, temperatures, t_eval):
    \"\"\"data: dict {T_kelvin: (n_times, 3) concentration array [A, B, C]}.
    Return dict with keys A1, E1, A2, E2 (pre-exponentials s^-1, activation energies J/mol).\"\"\"
```

## Scoring
R² of predicted concentrations vs observations, normalized vs baseline wrong-params fit.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
