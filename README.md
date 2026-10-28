# Forecasting Volatility in Calm and Turbulent Markets

**Does a different forecasting model help when the market moves into a high-volatility period?**

This project forecasts how much the NIFTY 50 index will fluctuate over the **next five trading days**. I built a hidden Markov model from scratch to infer calm and volatile market states, then trained a separate random forest for each state.

The combined model reduced forecast RMSE by **11.69%** compared with assuming the next five days will have the same volatility as the previous five. However, a **single random forest trained across both states** performed almost as well. The extra benefit from separating regimes is small and disappears in the non-overlapping evaluation.

[**Results and controls**](docs/RESULTS.md) · [**Narrated notebook**](HMM_Forward_5Day_Realized_Volatility.ipynb) · [**Runnable analysis**](scripts/run_analysis.py)

## What I built

- A two-state hidden Markov model with forward–backward inference and Baum–Welch estimation, implemented from scratch.
- State probabilities based only on information available at each forecast date.
- A genuine **future-volatility target**, followed by state-specific forecasting models and useful controls.
- A chronological evaluation that excludes unfinished training outcomes at the test boundary and checks the effect of overlapping forecast windows.

## Results in context

| Forecasting approach | Daily-test RMSE | Non-overlapping-test RMSE |
|---|---:|---:|
| **Hidden states + separate random forests** | **9.2569** | 8.6131 |
| One random forest for all market states | 9.2787 | **8.6061** |
| Use the previous five days' volatility | 10.4825 | 9.4629 |
| Exponentially weighted historical volatility | 10.4758 | 10.1721 |

RMSE gives larger forecast mistakes more weight; lower is better. Errors are in **annualised volatility percentage points**, not trading returns. The daily evaluation contains 1,140 forecasts; evaluating every fifth origin leaves 228 non-overlapping observations.

The regime model improves daily RMSE by only **0.23%** over the single-forest control, and their ordering reverses in the non-overlapping check. The predictors are useful relative to the simple historical baseline; a decisive advantage from the hidden-state layer is not established.

![Forecast errors for the models and baselines](docs/assets/forecast_rmse_comparison.png)

## How the analysis works

The first 70% of the timeline is the training block. A two-state lognormal HMM is fitted there, then causal filtered probabilities identify the state using observations available at each date. Random forests predict the next five days' realised volatility. Training labels must finish before the test block starts.

Adjacent daily forecasts share future returns, so the additional every-fifth-origin check removes target-window overlap. [Detailed tables, units and assumptions](docs/RESULTS.md).

## Run it

Python 3.10+; run from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/run_analysis.py
```

The script uses the included `nifty50_prices.csv` snapshot and writes tables to [outputs_forward_5day_volatility](outputs_forward_5day_volatility/). The notebook provides the narrated workflow. Its Yahoo download path is a fallback; future downloads need not reproduce the same data vintage.

## Scope

This is a volatility-forecasting study, not a trading backtest. One index and one static chronological split do not establish market-wide robustness. Repeated walk-forward estimation and other assets would be needed to test that broader claim.

**Author:** Fakhruddin Hakim Hussain · [Portfolio](https://github.com/Fakhruddin12)
