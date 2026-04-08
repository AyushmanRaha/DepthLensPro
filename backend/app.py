"""
DepthLens Pro — Backend API v3.0
FastAPI + PyTorch MiDaS · GPU/CPU selection · Full MDE metric suite
"""

import time, base64, logging, hashlib, math
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("depthlens")

# ── Constants ─────────────────────────────────────────────────────────────────
SUPPORTED_MODELS = {
    "MiDaS_small": {
        "label": "Small",
        "description": "~30 ms · EfficientNet-Lite · CPU-friendly",
        "compute": "CPU or GPU",
    },
    "DPT_Hybrid": {
        "label": "Hybrid",
        "description": "~120 ms · ViT-Hybrid · GPU recommended",
        "compute": "GPU recommended",
    },
    "DPT_Large": {
        "label": "Large",
        "description": "~400 ms · ViT-Large · GPU required for speed",
        "compute": "GPU required",
    },
}

COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma":  cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma":   cv2.COLORMAP_MAGMA,
    "jet":     cv2.COLORMAP_JET,
    "hot":     cv2.COLORMAP_HOT,
    "bone":    cv2.COLORMAP_BONE,
    "turbo":   cv2.COLORMAP_TURBO,
}

MAX_DIM      = 2048
MAX_SIZE_MB  = 20
CACHE: dict  = {}
MODELS: dict = {}
TRANSFORMS: dict = {}


# ── Device helpers ─────────────────────────────────────────────────────────────
def _available_devices() -> dict:
    devs = {"cpu": {"name": "CPU (System)", "type": "cpu", "available": True}}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devs[f"cuda:{i}"] = {
                "name":      p.name,
                "type":      "cuda",
                "index":     i,
                "memory_gb": round(p.total_memory / 1024**3, 1),
                "available": True,
            }
    return devs


def _resolve(requested: str) -> torch.device:
    avail = _available_devices()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested not in avail:
        raise ValueError(f"Device '{requested}' unavailable. Options: {list(avail)}")
    return torch.device(requested)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    devs = _available_devices()
    log.info(f"🚀 DepthLens Pro — devices: {list(devs)}")
    try:
        _load_model("MiDaS_small", "cpu")
        log.info("✅ MiDaS_small pre-loaded on CPU")
    except Exception as e:
        log.warning(f"⚠️  Pre-load skipped: {e}")
    yield
    log.info("🛑 Shutting down")
    MODELS.clear(); TRANSFORMS.clear(); CACHE.clear()


app = FastAPI(title="DepthLens Pro API", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── Model loading ──────────────────────────────────────────────────────────────
def _load_model(model_name: str, device_str: str):
    key = f"{model_name}:{device_str}"
    if key in MODELS:
        return MODELS[key], TRANSFORMS[model_name]
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name!r}")

    device = _resolve(device_str)
    log.info(f"Loading '{model_name}' → {device} …")
    model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
    model.to(device).eval()

    if model_name not in TRANSFORMS:
        mt = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        TRANSFORMS[model_name] = (
            mt.small_transform if model_name == "MiDaS_small" else mt.dpt_transform
        )

    MODELS[key] = (model, device)
    log.info(f"✅ '{model_name}' ready on {device}")
    return (model, device), TRANSFORMS[model_name]


# ── Image / inference helpers ─────────────────────────────────────────────────
def _decode(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image — corrupt or unsupported format")
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        s = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _infer(img_bgr: np.ndarray, model_name: str, device_str: str) -> np.ndarray:
    (model, device), transform = _load_model(model_name, device_str)
    rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    batch = transform(rgb).to(device)
    with torch.no_grad():
        pred = model(batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=img_bgr.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze()
    d = pred.cpu().numpy().astype(np.float32)
    lo, hi = d.min(), d.max()
    return (d - lo) / (hi - lo + 1e-8)


def _colorize(depth: np.ndarray, cmap: str) -> np.ndarray:
    u8 = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, COLORMAPS.get(cmap, cv2.COLORMAP_INFERNO))


def _b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _fhash(raw: bytes, model: str, cmap: str, dev: str) -> str:
    return hashlib.sha1(f"{model}:{cmap}:{dev}:{hashlib.md5(raw).hexdigest()}".encode()).hexdigest()


# ── MDE Metric Suite ──────────────────────────────────────────────────────────
def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    C1, C2 = (0.01)**2, (0.03)**2
    def blur(x): return cv2.GaussianBlur(x.astype(np.float32), (11, 11), 1.5).astype(np.float64)
    mu_a, mu_b = blur(a), blur(b)
    sig_a = blur(a**2) - mu_a**2
    sig_b = blur(b**2) - mu_b**2
    sig_ab = blur(a*b) - mu_a*mu_b
    num = (2*mu_a*mu_b + C1) * (2*sig_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sig_a + sig_b + C2)
    return float(np.mean(num / (den + 1e-10)))


def _compute_metrics(depth: np.ndarray, img_bgr: np.ndarray) -> dict:
    D  = depth.astype(np.float64)
    flat = D.flatten()

    # Basic stats
    d_min, d_max = float(flat.min()), float(flat.max())
    d_mean = float(flat.mean())
    d_std  = float(flat.std())
    d_med  = float(np.median(flat))

    # Dynamic range
    nz  = flat[flat > 1e-6]
    dyn = float(np.log2(nz.max() / nz.min())) if len(nz) > 0 else 0.0

    # Entropy
    hist, _ = np.histogram(flat, bins=256, range=(0.0, 1.0))
    hp      = hist / hist.sum()
    entropy = float(-np.sum(hp[hp > 0] * np.log2(hp[hp > 0])))

    # Coverage
    coverage = float((hp >= hp.max() * 0.01).mean())

    # Gradient / Edge
    D32 = D.astype(np.float32)
    gx  = cv2.Sobel(D32, cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(D32, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.sqrt(gx**2 + gy**2)
    grad_mean   = float(gmag.mean())
    grad_std    = float(gmag.std())
    grad_error  = grad_mean          # proxy: high = more edge detail
    edge_thresh = gmag.mean() + gmag.std()
    edge_density = float((gmag > edge_thresh).mean())

    # SSIM vs. grayscale input
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    ssim_val = _ssim(gray_img, D)

    # Error vs. mean (intra-map, no GT needed)
    pseudo = np.full_like(D, d_mean)
    eps    = 1e-6
    mae    = float(np.abs(D - pseudo).mean())
    rmse   = float(np.sqrt(np.mean((D - pseudo)**2)))
    log_rmse = float(np.sqrt(np.mean((np.log(D + eps) - np.log(pseudo + eps))**2)))

    # SILog
    log_d = np.log(D + eps)
    ld_mean = log_d.mean()
    silog = float(np.sqrt(np.mean((log_d - ld_mean)**2)) * 100)

    # PSNR
    mse_v = float(np.mean((D - d_mean)**2))
    psnr  = float(10 * math.log10(1.0 / (mse_v + 1e-10))) if mse_v > 1e-10 else 99.0

    # Histogram
    h32, edges = np.histogram(flat, bins=32, range=(0.0, 1.0))

    return {
        "min":      round(d_min,  4),
        "max":      round(d_max,  4),
        "mean":     round(d_mean, 4),
        "std":      round(d_std,  4),
        "median":   round(d_med,  4),

        "mae":      round(mae,      4),
        "rmse":     round(rmse,     4),
        "log_rmse": round(log_rmse, 4),
        "silog":    round(silog,    2),
        "psnr":     round(psnr,     2),

        "dynamic_range":  round(dyn,          2),
        "entropy":        round(entropy,       3),
        "coverage":       round(coverage,      4),

        "ssim":           round(ssim_val,      4),
        "gradient_mean":  round(grad_mean,     4),
        "gradient_std":   round(grad_std,      4),
        "gradient_error": round(grad_error,    4),
        "edge_density":   round(edge_density,  4),

        # GT-required — not computable without ground truth
        "abs_rel":             None,
        "sq_rel":              None,
        "delta_1":             None,
        "delta_2":             None,
        "delta_3":             None,
        "lpips":               None,
        "ordinal_error":       None,
        "surface_normal_error": None,

        "histogram": {
            "counts":    h32.tolist(),
            "bin_edges": [round(e, 3) for e in edges.tolist()],
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"service": "DepthLens Pro API", "version": "3.0.0"}


@app.get("/health")
async def health():
    devs = _available_devices()
    return {
        "status":          "ok",
        "primary_device":  "cuda:0" if torch.cuda.is_available() else "cpu",
        "devices":         devs,
        "loaded_models":   list(MODELS.keys()),
        "cache_entries":   len(CACHE),
        "torch_version":   torch.__version__,
        "cuda_available":  torch.cuda.is_available(),
    }


@app.get("/devices")
async def list_devices():
    return {"devices": _available_devices()}


@app.get("/models")
async def list_models():
    return {"models": [{"id": k, **v} for k, v in SUPPORTED_MODELS.items()]}


@app.get("/colormaps")
async def list_colormaps():
    return {"colormaps": list(COLORMAPS.keys())}


@app.post("/estimate")
async def estimate(
    file:     UploadFile = File(...),
    model:    str        = Form("MiDaS_small"),
    colormap: str        = Form("inferno"),
    device:   str        = Form("auto"),
):
    if model not in SUPPORTED_MODELS:
        raise HTTPException(422, f"Unknown model '{model}'")
    if colormap not in COLORMAPS:
        raise HTTPException(422, f"Unknown colormap '{colormap}'")

    avail = list(_available_devices().keys()) + ["auto"]
    if device not in avail:
        raise HTTPException(422, f"Device '{device}' unavailable. Options: {avail}")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Expected an image file")

    raw = await file.read()
    if len(raw) / 1024**2 > MAX_SIZE_MB:
        raise HTTPException(413, f"File exceeds {MAX_SIZE_MB} MB limit")

    resolved = str(_resolve(device))
    ck = _fhash(raw, model, colormap, resolved)
    if ck in CACHE:
        log.info(f"Cache hit: {file.filename!r}")
        return JSONResponse({**CACHE[ck], "cached": True})

    try:
        img = _decode(raw)
    except ValueError as e:
        raise HTTPException(422, str(e))

    t0 = time.perf_counter()
    try:
        depth = _infer(img, model, resolved)
    except Exception as e:
        log.exception("Inference failed")
        raise HTTPException(500, f"Inference error: {e}")
    lat = round((time.perf_counter() - t0) * 1000, 1)

    col  = _colorize(depth, colormap)
    gray = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mets = _compute_metrics(depth, img)

    result = {
        "depth_map":   _b64(col),
        "grayscale":   _b64(gray),
        "metrics":     mets,
        "latency_ms":  lat,
        "model":       model,
        "colormap":    colormap,
        "device_used": resolved,
        "resolution":  {"width": img.shape[1], "height": img.shape[0]},
        "filename":    file.filename,
        "cached":      False,
    }
    CACHE[ck] = result
    log.info(f"✅ {file.filename!r} | {model} | {resolved} | {lat} ms")
    return JSONResponse(result)


@app.post("/batch")
async def batch(
    files:    list[UploadFile] = File(...),
    model:    str              = Form("MiDaS_small"),
    colormap: str              = Form("inferno"),
    device:   str              = Form("auto"),
):
    if len(files) > 10:
        raise HTTPException(422, "Batch limit: 10 images")
    resolved = str(_resolve(device))
    results, errors = [], []
    for f in files:
        try:
            raw = await f.read()
            if len(raw) / 1024**2 > MAX_SIZE_MB:
                raise ValueError("File too large")
            ck = _fhash(raw, model, colormap, resolved)
            if ck in CACHE:
                results.append({**CACHE[ck], "cached": True}); continue
            img   = _decode(raw)
            t0    = time.perf_counter()
            depth = _infer(img, model, resolved)
            lat   = round((time.perf_counter() - t0) * 1000, 1)
            col   = _colorize(depth, colormap)
            gray  = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            res   = {
                "depth_map": _b64(col), "grayscale": _b64(gray),
                "metrics": _compute_metrics(depth, img),
                "latency_ms": lat, "model": model, "colormap": colormap,
                "device_used": resolved,
                "resolution": {"width": img.shape[1], "height": img.shape[0]},
                "filename": f.filename, "cached": False,
            }
            CACHE[ck] = res; results.append(res)
        except Exception as e:
            errors.append({"filename": f.filename, "error": str(e)})
    return JSONResponse({"results": results, "errors": errors,
                         "total": len(files), "succeeded": len(results), "failed": len(errors)})


@app.delete("/cache")
async def clear_cache():
    n = len(CACHE); CACHE.clear()
    return {"cleared": n}


@app.exception_handler(Exception)
async def _err(req: Request, exc: Exception):
    log.exception(f"Unhandled: {req.url}")
    return JSONResponse(500, {"detail": "Internal server error"})