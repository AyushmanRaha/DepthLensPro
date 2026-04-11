import time

import cv2
from fastapi import HTTPException, UploadFile

from .config import COLORMAPS, MAX_BATCH_FILES, MAX_SIZE_MB, SUPPORTED_MODELS
from .devices import available_devices, resolve_device
from .image_ops import colorize_depth, decode_image, encode_png_b64, make_cache_key
from .metrics import compute_metrics
from .models import infer_depth
from .runtime import CACHE


def validate_request(model: str, colormap: str, device: str) -> str:
    if model not in SUPPORTED_MODELS:
        raise HTTPException(422, f"Unknown model '{model}'")
    if colormap not in COLORMAPS:
        raise HTTPException(422, f"Unknown colormap '{colormap}'")

    available = list(available_devices().keys()) + ["auto"]
    if device not in available:
        raise HTTPException(422, f"Device '{device}' unavailable. Options: {available}")
    return str(resolve_device(device))


async def process_single(file: UploadFile, model: str, colormap: str, device: str) -> dict:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Expected an image file")

    raw = await file.read()
    if len(raw) / 1024**2 > MAX_SIZE_MB:
        raise HTTPException(413, f"File exceeds {MAX_SIZE_MB} MB limit")

    resolved = validate_request(model, colormap, device)
    cache_key = make_cache_key(raw, model, colormap, resolved)
    if cache_key in CACHE:
        return {**CACHE[cache_key], "cached": True}

    try:
        img = decode_image(raw)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    t0 = time.perf_counter()
    try:
        depth = infer_depth(img, model, resolved)
    except Exception as exc:
        raise HTTPException(500, f"Inference error: {exc}") from exc
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    colorized = colorize_depth(depth, colormap)
    gray = cv2.cvtColor((depth * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
    result = {
        "depth_map": encode_png_b64(colorized),
        "grayscale": encode_png_b64(gray),
        "metrics": compute_metrics(depth, img),
        "latency_ms": latency_ms,
        "model": model,
        "colormap": colormap,
        "device_used": resolved,
        "resolution": {"width": img.shape[1], "height": img.shape[0]},
        "filename": file.filename,
        "cached": False,
    }
    CACHE[cache_key] = result
    return result


async def process_batch(files: list[UploadFile], model: str, colormap: str, device: str) -> dict:
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(422, f"Batch limit: {MAX_BATCH_FILES} images")

    resolved = validate_request(model, colormap, device)
    results, errors = [], []

    for file in files:
        try:
            raw = await file.read()
            if len(raw) / 1024**2 > MAX_SIZE_MB:
                raise ValueError("File too large")

            cache_key = make_cache_key(raw, model, colormap, resolved)
            if cache_key in CACHE:
                results.append({**CACHE[cache_key], "cached": True})
                continue

            img = decode_image(raw)
            t0 = time.perf_counter()
            depth = infer_depth(img, model, resolved)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            colorized = colorize_depth(depth, colormap)
            gray = cv2.cvtColor((depth * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
            result = {
                "depth_map": encode_png_b64(colorized),
                "grayscale": encode_png_b64(gray),
                "metrics": compute_metrics(depth, img),
                "latency_ms": latency_ms,
                "model": model,
                "colormap": colormap,
                "device_used": resolved,
                "resolution": {"width": img.shape[1], "height": img.shape[0]},
                "filename": file.filename,
                "cached": False,
            }
            CACHE[cache_key] = result
            results.append(result)
        except Exception as exc:
            errors.append({"filename": file.filename, "error": str(exc)})

    return {
        "results": results,
        "errors": errors,
        "total": len(files),
        "succeeded": len(results),
        "failed": len(errors),
    }
