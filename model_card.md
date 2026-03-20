# Model Card

## Purpose

The model stack estimates the probability that a trader, desk, or daily activity window exhibits Barings-like rogue trading and operational risk behavior.

## Included components

- `logistic_regression`: interpretable fraud classifier.
- `random_forest`: nonlinear supervised classifier.
- `isolation_forest`: unsupervised anomaly detector for unusual operational patterns.
- `sequence_autoencoder`: TensorFlow LSTM autoencoder when available, otherwise PCA reconstruction over rolling sequences.
- `ensemble`: weighted aggregation and calibration to `low`, `medium`, `high`, and `critical`.

## Inputs

Derived operational features such as hidden-loss ratio, secret account usage, PnL/cash mismatch, exposure growth, funding spikes, reconciliation breaks, and sign-off weakness.

## Outputs

- `risk_score` in `[0, 1]`
- `risk_band`
- top contributing factors
- narrative reason code

## Limitations

- Synthetic labels are scenario-driven and do not represent real internal Barings ledgers.
- Sequence modeling falls back to PCA when TensorFlow is unavailable.
- Explanations are exact SHAP values only when `shap` is installed.

