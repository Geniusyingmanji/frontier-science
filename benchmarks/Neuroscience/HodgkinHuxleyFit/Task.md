# HodgkinHuxleyFit — fit neuron model parameters to a voltage trace

## Scientific background

The Hodgkin-Huxley (1952) model describes action potential generation via 4 coupled ODEs
governing membrane voltage and ion-channel gating (Na⁺ activation m, inactivation h; K⁺
activation n). With 8 core parameters (conductances, reversal potentials, capacitance,
stimulus current), fitting the model to experimental voltage traces is a classic nonlinear
inverse problem. The landscape is highly multimodal — compensatory parameter changes produce
similar waveforms (e.g. faster activation + more inactivation). This is the foundational
model for computational neuroscience.

Reference: Hodgkin & Huxley, J. Physiol. 117, 500 (1952); Buhry et al., Neurocomputing 81, 53 (2012).

## Your task

```python
def fit_hh():
    \"\"\"Return a dict with keys: g_Na, g_K, g_L, E_Na, E_K, E_L, Cm, I_ext.
    The evaluator simulates the HH model with your parameters and scores how well
    the voltage trace matches a hidden target (generated from the true parameters).\"\"\"
```

The target trace: 50ms simulation, step-current injection from 5-35ms. The closer your
simulated trace matches the target (in RMSE), the higher your score.

## Scoring

```
score = clip( (RMSE_baseline - RMSE_found) / RMSE_baseline, 0, 1 )
```
Baseline: 20% perturbation of true parameters. True parameters → RMSE=0, score=1.0.

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
