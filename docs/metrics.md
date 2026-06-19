# Metrics

DepthLens Pro reports two categories of metrics: prediction-only diagnostics and ground-truth metrics.

## Prediction-only diagnostics

Prediction-only diagnostics do not require a GT file. They summarize properties of the predicted depth plane such as range, mean, standard deviation, and runtime/cache timing. These are useful for debugging and comparison, but they are not accuracy scores.

## Ground-truth metrics

GT metrics require a user-provided depth map. The evaluator masks invalid pixels, resizes with nearest-neighbour behavior where appropriate, applies median-scale alignment, and computes standard depth-estimation metrics.

Let `d` be ground truth, `p` be prediction, and `N` be the valid-pixel count.

| Metric | Formula | Interpretation |
|---|---|---|
| Abs Rel | `mean(abs(d - p) / d)` | Lower is better |
| Sq Rel | `mean((d - p)^2 / d)` | Lower is better; penalizes larger errors |
| RMSE | `sqrt(mean((d - p)^2))` | Lower is better |
| Log RMSE | `sqrt(mean((log(d) - log(p))^2))` | Lower is better; scale-aware in log space |
| δ thresholds | `% where max(d/p, p/d) < 1.25^k` for `k = 1, 2, 3` | Higher is better |

## Median-scale alignment

Monocular models predict relative depth, so the evaluator aligns prediction scale to GT with a median ratio before computing GT metrics. This keeps evaluation focused on relative depth structure rather than an arbitrary model-output scale.
