# Forward-volatility results

The portfolio refresh reran the exported analysis against the included price snapshot with Python 3.12, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3 and scikit-learn 1.8.0. Small differences from the archived notebook are attributable to the execution environment; the interpretation is unchanged.

## Daily test

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| HMM + random forests | 9.2569 | 5.9135 | 0.4318 |
| Global random forest | 9.2787 | 5.9168 | 0.4291 |
| EWMA | 10.4758 | 6.5230 | 0.2723 |
| Historical five-day volatility | 10.4825 | 6.9731 | 0.2714 |

RMSE improves 11.69% against the historical baseline, but only about 0.23% against the global forest. Volatility errors are in annualised percentage points.

## Non-overlapping check

Evaluating every fifth forecast origin leaves 228 observations. Global-forest RMSE is 8.6061 and HMM-plus-forest RMSE is 8.6131. The ordering reverses: regime separation is not decisively better. Historical-baseline RMSE is 9.4629.

[Daily results](../outputs_forward_5day_volatility/forecast_results.csv) · [Non-overlapping results](../outputs_forward_5day_volatility/non_overlapping_results.csv) · [Predictions](../outputs_forward_5day_volatility/test_predictions.csv)

## Evaluation controls

The HMM is fit only on the first 70% of the timeline, with causal forward-filtered state probabilities. Forest training labels must end strictly before the test boundary. The target contains five future returns; adjacent daily targets overlap, motivating the separate weekly check. The script retains assertions for transition normalisation, filtering and train/test label boundaries.

## Limits

One index, one chronological split and a static HMM fit are studied. The small difference between forests is not evidence of a statistically established regime advantage. These are volatility forecasts, not a backtested trading strategy.

[Archived narrative and notebook results](ORIGINAL_RESEARCH_NOTES.md)
