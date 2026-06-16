# DemographicSFS — infer population history from the Site Frequency Spectrum

## Scientific background
The Site Frequency Spectrum (SFS) counts how many polymorphic sites have a derived allele
at each frequency 1/n, 2/n, ..., (n-1)/n in a sample. Under neutrality, the SFS shape is
determined by population size history: bottlenecks produce excess rare variants, expansions
produce excess intermediate-frequency variants. Fitting a parameterized demographic model
(3-epoch: ancestral → bottleneck → expansion) to the observed SFS by numerically solving
the diffusion equation is the core inference problem in tools like ∂a∂i and moments.

Reference: Gutenkunst et al., PLoS Genetics 5, e1000695 (2009); Kimura 1955.

## Your task
```python
def fit_demography(observed_sfs, n_sample):
    \"\"\"Given the observed SFS (n_sample-1,) and sample size n, return
    [N_anc, N_bot, N_cur, T_bot_start, T_bot_end] — relative population sizes
    and times in 2N_anc generations. All positive.\"\"\"
```

## Scoring
Poisson log-likelihood of model-predicted SFS vs observed. Normalized between constant-N
baseline and true generating parameters.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
