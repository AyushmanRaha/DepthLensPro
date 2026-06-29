# Models, Colormaps & Metrics

[← Back to README](../README.md)

### Supported Models

| Canonical ID | Display name | Architecture | Input size | ONNX file | Recommended use |
|---|---|---|---:|---|---|
| `midas_small` | MiDaS Small | MiDaS small / EfficientNet-Lite | 256×256 | `midas_small.onnx` | Fast previews, CPU-friendly runs, webcam |
| `dpt_hybrid` | DPT Hybrid | DPT Hybrid / ViT-Hybrid | 384×384 | `dpt_hybrid.onnx` | Balanced quality/speed when a GPU or fast CPU is available |
| `dpt_large` | DPT Large | DPT Large / ViT-Large | 384×384 | `dpt_large.onnx` | Highest-detail option; GPU-preferred for practical latency |

Model names are normalised to canonical IDs automatically. Standard PyTorch use does not require ONNX files; ONNX Runtime is used only when requested or automatic selection finds a valid local graph. All of these resolve to `midas_small`:

```
MiDaS_small  /  MiDaS Small  /  midas_small  /  midas-small  /  MiDaS-Small
```

---

### Which Model Should I Use?

| Goal | Recommended model |
|---|---|
| Fastest result | MiDaS Small |
| Webcam / real-time preview | MiDaS Small |
| Balanced visual quality | DPT Hybrid |
| Maximum edge detail | DPT Large |
| CPU-only machine | MiDaS Small |
| CUDA or MPS GPU available | DPT Hybrid or DPT Large |
| Benchmarking ONNX acceleration | MiDaS Small (most ONNX-export-friendly) |

---

### Colormaps

Supported colormaps:

```
inferno · plasma · viridis · magma · jet · hot · bone · turbo
```

| Colormap | Use when |
|---|---|
| `inferno` | You want high contrast; safe default for most visualisations |
| `viridis` | You want perceptually uniform, colorblind-safer output |
| `plasma` | You want a bright, warm presentation style |
| `magma` | You want a softer dark-to-light depth map |
| `turbo` | You want strong colour separation across the full depth range |
| `jet` | You need a classic rainbow map for compatibility |
| `hot` | You want heat-map-style output |
| `bone` | You want a subtle greyscale-adjacent map |

---

### Metrics Modes

| Mode | Description | Latency impact |
|---|---|---|
| `none` | Return output images only — no metric computation | Minimal |
| `fast` | Lightweight prediction statistics (min, max, mean, std, entropy, histogram) | ~1–3 ms |
| `full` | Full statistics plus proxy diagnostics (SSIM, SILog, PSNR, gradient, edge density) | ~5–15 ms |

### Metric Groups

| Group | Examples | Requires GT? |
|---|---|---|
| Prediction stats | min, max, mean, std, median, histogram, entropy, coverage | No |
| Proxy metrics | SSIM (vs input), SILog, PSNR, gradient error, edge density, MAE/RMSE vs predicted mean | No |
| Ground-truth metrics | Abs Rel, Sq Rel, GT RMSE, log RMSE, δ < 1.25 / 1.25² / 1.25³ | Yes |
| Reported unavailable | GT SSIM, GT PSNR, ordinal error, surface normal error, LPIPS | Depends; not yet implemented |

<div align="right"><sub><a href="../README.md#depthlens-pro">⬆ back to README</a></sub></div>

---
