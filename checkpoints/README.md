# C-CASE Checkpoints

This directory contains model and preprocessing checkpoints required
for reproducing the C-CASE experiments.

The final experiments use:

- K-Means clustering with K = 3
- Random state = 42
- Chronological 80:20 development/test split
- 10-observation historical lookback
- 1-observation (5-minute) forecasting horizon

The preprocessing checkpoint contains the fitted scaler,
K-Means model, feature configuration, cluster assignments,
and train/test information.

Large binary checkpoint files are not committed unless required.
They can be regenerated using the preprocessing script.
