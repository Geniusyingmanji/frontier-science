# HeatExchangerDesign — optimize a counter-flow heat exchanger

## Scientific background
Heat exchangers transfer thermal energy between fluids. The effectiveness-NTU method relates
geometry (tube length, diameter, number of passes) to thermal effectiveness. Optimizing the
geometry for maximum heat transfer per unit volume is central to HVAC, power plants, and
chemical engineering.

## Your task
```python
def design_exchanger(T_hot_in, T_cold_in, m_hot, m_cold, cp_hot, cp_cold, U, max_area):
    """Return (area, n_passes) that maximize effectiveness = Q / Q_max."""
```

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
