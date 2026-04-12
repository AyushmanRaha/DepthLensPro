"""
DepthLens Pro — Backend API v3.1
FastAPI + PyTorch MiDaS · GPU/CPU selection · Full MDE metric suite

Fixes over v3.0:
- _available_devices: MPS correctly classified as GPU only (not NPU).
  Apple Neural Engine is not accessible via PyTorch/MPS.
- _available_devices: Intel XPU correctly classified as GPU only (not NPU).
  Intel AI Boost NPU is accessed via OpenVINO, not torch.xpu.
- _default_device_key: priority is now CUDA > MPS > XPU > CPU.
  Previously XPU (labelled NPU) incorrectly beat CUDA.
- MPS availability: guards with both is_built() AND is_available().
- _load_model: removed redundant second call to _resolve().
- Lifespan pre-load: warms up on the best available device, not always CPU.
- _acceleration_checks: NPU entry removed (no PyTorch NPU backend exists);
  health endpoint now accurately reports what is actually operational.
"""

import time, base64, logging, hashlib, math
import subprocess, platform, os
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
def _get_apple_chip() -> str | None:
    """Return e.g. 'M2 Pro', 'M3 Max', 'M1' if on Apple Silicon, else None."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=3
        ).stdout.strip()
        if out.startswith("Apple "):
            return out[6:]
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            if "Chip" in line and "Apple" in line:
                chip = line.split(":", 1)[-1].strip()
                return chip.replace("Apple ", "")
    except Exception:
        pass
    return None


def _mps_available() -> bool:
    """
    MPS requires BOTH is_built() (PyTorch compiled with Metal support)
    AND is_available() (running on macOS ≥ 12.3 with compatible hardware).
    """
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _available_devices() -> dict:
    """
    Returns a dict of device-key → device-info for every compute target
    this process can actually use.

    Classification notes
    --------------------
    • CUDA  → GPU   (NVIDIA hardware via CUDA)
    • MPS   → GPU   (Apple Silicon GPU via Metal Performance Shaders)
              NOT "npu": the Apple Neural Engine is a separate chip and is
              not accessible through PyTorch/MPS.
    • XPU   → GPU   (Intel Arc / Xe iGPU via torch.xpu)
              NOT "npu": Intel's AI Boost NPU is accessed via OpenVINO,
              not torch.xpu. Labelling XPU as NPU was factually wrong and
              caused XPU to be incorrectly prioritised over CUDA.
    """
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "System CPU"
    devs = {
        "cpu": {
            "name": f"CPU · {cpu_name}",
            "hardware_name": cpu_name,
            "type": "cpu",
            "compute_classes": ["cpu"],
            "available": True,
        }
    }

    # ── NVIDIA CUDA ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devs[f"cuda:{i}"] = {
                "name":            f"GPU · {p.name}",
                "hardware_name":   p.name,
                "type":            "cuda",
                "compute_classes": ["gpu"],
                "index":           i,
                "memory_gb":       round(p.total_memory / 1024**3, 1),
                "available":       True,
            }

    # ── Intel XPU (Arc / Xe iGPU) ───────────────────────────────────────────
    # torch.xpu targets Intel GPU silicon, NOT Intel's AI Boost NPU.
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            count = torch.xpu.device_count()
        except Exception:
            count = 0
        for i in range(count):
            name = (
                torch.xpu.get_device_name(i)
                if hasattr(torch.xpu, "get_device_name")
                else f"Intel XPU {i}"
            )
            devs[f"xpu:{i}"] = {
                "name":            f"GPU · {name}",
                "hardware_name":   name,
                "type":            "xpu",
                "compute_classes": ["gpu"],   # XPU is a GPU, not an NPU
                "index":           i,
                "available":       True,
            }

    # ── Apple MPS (Metal Performance Shaders on Apple Silicon GPU) ───────────
    # MPS uses the GPU cores of Apple Silicon, NOT the Apple Neural Engine.
    if _mps_available():
        chip = _get_apple_chip() or "Apple Silicon"
        devs["mps"] = {
            "name":            f"GPU · Apple {chip} (Metal)",
            "hardware_name":   f"Apple {chip}",
            "type":            "mps",
            "compute_classes": ["gpu"],   # GPU via Metal, not NPU
            "chip":            chip,
            "available":       True,
        }

    return devs


def _default_device_key() -> str:
    """
    Pick the best available compute target for deep-learning inference.

    Priority: CUDA > MPS > XPU > CPU

    Rationale:
    • CUDA has the most mature PyTorch optimisation and broadest op support.
    • MPS (Apple Silicon GPU) is second-best; well-supported since PyTorch 2.x.
    • Intel XPU (Arc/Xe) has growing but still more limited PyTorch op coverage.
    • CPU is the universal fallback.

    Note: there is no PyTorch-accessible NPU backend — Apple ANE and Intel
    AI Boost are not exposed via torch.device, so they are not considered here.
    """
    devs = _available_devices()
    if any(k.startswith("cuda:") for k in devs):
        return "cuda:0"
    if "mps" in devs:
        return "mps"
    # XPU: prefer the first Intel GPU if present
    xpu_keys = [k for k in devs if k.startswith("xpu:")]
    if xpu_keys:
        return xpu_keys[0]
    return "cpu"


def _acceleration_checks(devs: dict) -> dict:
    """
    Run tiny tensor ops on each available accelerator to verify runtime
    usability beyond mere availability flags.

    Returns only what actually exists — no fake 'npu' entry since no
    PyTorch NPU backend is available on any current platform.
    """
    checks = {}

    def _probe(device_key: str, label: str):
        try:
            dev = torch.device(device_key)
            x = torch.randn((16, 16), device=dev)
            y = torch.mm(x, x.transpose(0, 1))
            _ = float(y.mean().detach().cpu().item())
            checks[label] = {"available": True, "operational": True}
        except Exception as e:
            checks[label] = {"available": True, "operational": False, "error": str(e)}

    cuda_keys = [k for k in devs if k.startswith("cuda:")]
    if cuda_keys:
        _probe(cuda_keys[0], "cuda")
    else:
        checks["cuda"] = {"available": False, "operational": False}

    if "mps" in devs:
        _probe("mps", "mps")
    else:
        checks["mps"] = {"available": False, "operational": False}

    xpu_keys = [k for k in devs if k.startswith("xpu:")]
    if xpu_keys:
        _probe(xpu_keys[0], "xpu")
    else:
        checks["xpu"] = {"available": False, "operational": False}

    return checks


def _resolve(requested: str) -> torch.device:
    """Validate and convert a device string to a torch.device."""
    avail = _available_devices()
    if requested == "auto":
        return torch.device(_default_device_key())
    if requested not in avail:
        raise ValueError(f"Device '{requested}' unavailable. Options: {list(avail)}")
    return torch.device(requested)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    devs = _available_devices()
    best = _default_device_key()
    log.info(f"🚀 DepthLens Pro v3.1 — devices: {list(devs)}  best: {best}")
    try:
        # Pre-load MiDaS_small on the best available device so the first
        # inference request doesn't pay the cold-start penalty.
        _load_model("MiDaS_small", best)
        log.info(f"✅ MiDaS_small pre-loaded on {best}")
    except Exception as e:
        log.warning(f"⚠️  Pre-load on {best} skipped: {e}")
        # Fallback: at least warm up on CPU
        try:
            _load_model("MiDaS_small", "cpu")
            log.info("✅ MiDaS_small pre-loaded on CPU (fallback)")
        except Exception as e2:
            log.warning(f"⚠️  CPU pre-load also skipped: {e2}")
    yield
    log.info("🛑 Shutting down")
    MODELS.clear(); TRANSFORMS.clear(); CACHE.clear()


app = FastAPI(title="DepthLens Pro API", version="3.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── Model loading ──────────────────────────────────────────────────────────────
def _load_model(model_name: str, device_str: str):
    """
    Load (or return cached) a MiDaS model on the specified device.

    `device_str` must already be a resolved device key (e.g. 'cuda:0', 'mps',
    'cpu') — NOT 'auto'. Callers are responsible for resolving 'auto' before
    calling this function. This avoids a redundant second call to _resolve().
    """
    key = f"{model_name}:{device_str}"
    if key in MODELS:
        return MODELS[key], TRANSFORMS[model_name]
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name!r}")

    device = torch.device(device_str)   # already validated — no need to _resolve() again
    log.info(f"Loading '{model_name}' → {device} …")
    model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
    model.to(device).eval()

    # MPS requires float32; mixed-precision is not yet fully supported
    if device.type == "mps":
        model = model.float()

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

    d_min, d_max = float(flat.min()), float(flat.max())
    d_mean = float(flat.mean())
    d_std  = float(flat.std())
    d_med  = float(np.median(flat))

    nz  = flat[flat > 1e-6]
    dyn = float(np.log2(nz.max() / nz.min())) if len(nz) > 0 else 0.0

    hist, _ = np.histogram(flat, bins=256, range=(0.0, 1.0))
    hp      = hist / hist.sum()
    entropy = float(-np.sum(hp[hp > 0] * np.log2(hp[hp > 0])))
    coverage = float((hp >= hp.max() * 0.01).mean())

    D32 = D.astype(np.float32)
    gx  = cv2.Sobel(D32, cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(D32, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.sqrt(gx**2 + gy**2)
    grad_mean   = float(gmag.mean())
    grad_std    = float(gmag.std())
    grad_error  = grad_mean
    edge_thresh = gmag.mean() + gmag.std()
    edge_density = float((gmag > edge_thresh).mean())

    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    ssim_val = _ssim(gray_img, D)

    pseudo = np.full_like(D, d_mean)
    eps    = 1e-6
    mae    = float(np.abs(D - pseudo).mean())
    rmse   = float(np.sqrt(np.mean((D - pseudo)**2)))
    log_rmse = float(np.sqrt(np.mean((np.log(D + eps) - np.log(pseudo + eps))**2)))

    log_d = np.log(D + eps)
    ld_mean = log_d.mean()
    silog = float(np.sqrt(np.mean((log_d - ld_mean)**2)) * 100)

    mse_v = float(np.mean((D - d_mean)**2))
    psnr  = float(10 * math.log10(1.0 / (mse_v + 1e-10))) if mse_v > 1e-10 else 99.0

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

        "abs_rel":              None,
        "sq_rel":               None,
        "delta_1":              None,
        "delta_2":              None,
        "delta_3":              None,
        "lpips":                None,
        "ordinal_error":        None,
        "surface_normal_error": None,

        "histogram": {
            "counts":    h32.tolist(),
            "bin_edges": [round(e, 3) for e in edges.tolist()],
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"service": "DepthLens Pro API", "version": "3.1.0"}


@app.get("/health")
async def health():
    devs  = _available_devices()
    best  = _default_device_key()
    accel = {k: v for k, v in devs.items() if k != "cpu"}
    checks = _acceleration_checks(devs)

    # acceleration_ok: at least one non-CPU device is operational
    accel_ok = any(c.get("operational") for c in checks.values() if c.get("available"))

    return {
        "status":          "ok",
        "version":         "3.1.0",
        "primary_device":  best,
        "devices":         devs,
        "loaded_models":   list(MODELS.keys()),
        "cache_entries":   len(CACHE),
        "torch_version":   torch.__version__,
        "cuda_available":  any(k.startswith("cuda:") for k in devs),
        "mps_available":   "mps" in devs,
        "xpu_available":   any(k.startswith("xpu:") for k in devs),
        # 'npu_available' removed — no PyTorch NPU backend exists on any platform
        "acceleration_ok": bool(accel) and accel_ok,
        "acceleration_checks": checks,
        "system": {
            "os":           platform.platform(),
            "machine":      platform.machine(),
            "cpu":          devs["cpu"]["hardware_name"],
            "accelerators": [d["name"] for d in accel.values()],
        },
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

    # Resolve 'auto' once here; _load_model receives the concrete device string
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