import math

import cv2
import numpy as np


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = (0.01) ** 2, (0.03) ** 2

    def blur(x):
        return cv2.GaussianBlur(x.astype(np.float32), (11, 11), 1.5).astype(np.float64)

    mu_a, mu_b = blur(a), blur(b)
    sig_a = blur(a**2) - mu_a**2
    sig_b = blur(b**2) - mu_b**2
    sig_ab = blur(a * b) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
    return float(np.mean(numerator / (denominator + 1e-10)))


def compute_metrics(depth: np.ndarray, img_bgr: np.ndarray) -> dict:
    depth64 = depth.astype(np.float64)
    flat = depth64.flatten()

    d_min, d_max = float(flat.min()), float(flat.max())
    d_mean, d_std = float(flat.mean()), float(flat.std())
    d_med = float(np.median(flat))

    nz = flat[flat > 1e-6]
    dyn = float(np.log2(nz.max() / nz.min())) if len(nz) > 0 else 0.0

    hist, _ = np.histogram(flat, bins=256, range=(0.0, 1.0))
    hp = hist / hist.sum()
    entropy = float(-np.sum(hp[hp > 0] * np.log2(hp[hp > 0])))
    coverage = float((hp >= hp.max() * 0.01).mean())

    d32 = depth64.astype(np.float32)
    gx = cv2.Sobel(d32, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(d32, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.sqrt(gx**2 + gy**2)
    grad_mean, grad_std = float(gmag.mean()), float(gmag.std())
    grad_error = grad_mean
    edge_thresh = gmag.mean() + gmag.std()
    edge_density = float((gmag > edge_thresh).mean())

    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    ssim_val = _ssim(gray_img, depth64)

    pseudo = np.full_like(depth64, d_mean)
    eps = 1e-6
    mae = float(np.abs(depth64 - pseudo).mean())
    rmse = float(np.sqrt(np.mean((depth64 - pseudo) ** 2)))
    log_rmse = float(np.sqrt(np.mean((np.log(depth64 + eps) - np.log(pseudo + eps)) ** 2)))

    log_d = np.log(depth64 + eps)
    silog = float(np.sqrt(np.mean((log_d - log_d.mean()) ** 2)) * 100)

    mse_v = float(np.mean((depth64 - d_mean) ** 2))
    psnr = float(10 * math.log10(1.0 / (mse_v + 1e-10))) if mse_v > 1e-10 else 99.0

    h32, edges = np.histogram(flat, bins=32, range=(0.0, 1.0))

    return {
        "min": round(d_min, 4),
        "max": round(d_max, 4),
        "mean": round(d_mean, 4),
        "std": round(d_std, 4),
        "median": round(d_med, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "log_rmse": round(log_rmse, 4),
        "silog": round(silog, 2),
        "psnr": round(psnr, 2),
        "dynamic_range": round(dyn, 2),
        "entropy": round(entropy, 3),
        "coverage": round(coverage, 4),
        "ssim": round(ssim_val, 4),
        "gradient_mean": round(grad_mean, 4),
        "gradient_std": round(grad_std, 4),
        "gradient_error": round(grad_error, 4),
        "edge_density": round(edge_density, 4),
        "abs_rel": None,
        "sq_rel": None,
        "delta_1": None,
        "delta_2": None,
        "delta_3": None,
        "lpips": None,
        "ordinal_error": None,
        "surface_normal_error": None,
        "histogram": {
            "counts": h32.tolist(),
            "bin_edges": [round(e, 3) for e in edges.tolist()],
        },
    }
