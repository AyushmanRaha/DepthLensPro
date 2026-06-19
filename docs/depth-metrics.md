# Understanding Depth Metrics

[← Back to README](../README.md)

### Prediction-Only Metrics (no GT required)

These metrics are computed from the normalised depth plane alone. They measure the internal richness and structure of the prediction, not its accuracy against a reference.

| Metric | What it measures | Good values |
|---|---|---|
| **Entropy** | Shannon entropy of the 256-bin depth histogram | Higher = more uniformly distributed depth values |
| **Dynamic range** | log₂(max/min non-zero depth) in bits | Higher = wider depth variation captured |
| **Coverage** | Fraction of histogram bins with ≥1% of peak count | Higher = depth values spread across full range |
| **Edge density** | Fraction of pixels with gradient magnitude > mean+std | Higher = more structural depth edges |
| **SSIM (proxy)** | Structural similarity between predicted depth and greyscale RGB input | Not a benchmark metric; correlates depth structure with image edges |
| **SILog (proxy)** | Log-depth dispersion of the prediction itself | Not true SILog; use for relative comparison only |

### GT Metrics (requires ground truth upload)

These are standard monocular depth estimation benchmark metrics used in papers like Eigen et al. and the MiDaS evaluation suite.

| Metric | Formula | Interpretation |
|---|---|---|
| **Abs Rel** | `mean(|pred − gt| / gt)` | Primary quality metric; lower is better |
| **Sq Rel** | `mean((pred − gt)² / gt)` | Penalises large errors more heavily; lower is better |
| **GT RMSE** | `sqrt(mean((pred − gt)²))` | Root mean squared error; lower is better |
| **GT Log RMSE** | `sqrt(mean((log pred − log gt)²))` | Less sensitive to scale outliers; lower is better |
| **δ < 1.25** | `mean(max(pred/gt, gt/pred) < 1.25)` | Percentage of pixels within 25% of GT; higher is better |
| **δ < 1.25²** | Same with threshold 1.5625 | Looser accuracy; higher is better |
| **δ < 1.25³** | Same with threshold 1.953 | Loosest threshold; higher is better |

<div align="right"><sub><a href="../README.md#depthlens-pro">⬆ back to README</a></sub></div>

---
