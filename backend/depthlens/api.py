import platform

import torch
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from .config import APP_VERSION, COLORMAPS, SUPPORTED_MODELS
from .devices import available_devices, default_device_key
from .runtime import CACHE, MODELS
from .service import process_batch, process_single

router = APIRouter()


@router.get("/")
async def root():
    return {"service": "DepthLens Pro API", "version": APP_VERSION}


@router.get("/health")
async def health():
    devices = available_devices()
    return {
        "status": "ok",
        "primary_device": default_device_key(),
        "devices": devices,
        "loaded_models": list(MODELS.keys()),
        "cache_entries": len(CACHE),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "system": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "cpu": devices["cpu"]["hardware_name"],
            "accelerators": [d["name"] for key, d in devices.items() if key != "cpu"],
        },
    }


@router.get("/devices")
async def list_devices():
    return {"devices": available_devices()}


@router.get("/models")
async def list_models():
    return {"models": [{"id": key, **value} for key, value in SUPPORTED_MODELS.items()]}


@router.get("/colormaps")
async def list_colormaps():
    return {"colormaps": list(COLORMAPS.keys())}


@router.post("/estimate")
async def estimate(
    file: UploadFile = File(...),
    model: str = Form("MiDaS_small"),
    colormap: str = Form("inferno"),
    device: str = Form("auto"),
):
    return JSONResponse(await process_single(file, model, colormap, device))


@router.post("/batch")
async def batch(
    files: list[UploadFile] = File(...),
    model: str = Form("MiDaS_small"),
    colormap: str = Form("inferno"),
    device: str = Form("auto"),
):
    return JSONResponse(await process_batch(files, model, colormap, device))


@router.delete("/cache")
async def clear_cache():
    size = len(CACHE)
    CACHE.clear()
    return {"cleared": size}
