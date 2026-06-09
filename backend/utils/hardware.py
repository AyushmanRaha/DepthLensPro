"""Hardware detection and torch/ONNX device selection helpers."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any, Sequence

import torch

DeviceMap = dict[str, dict[str, Any]]


def _get_apple_chip() -> str | None:
    """Resolve the Apple Silicon chip name when available."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout.strip()
        if out.startswith("Apple "):
            return out[6:]
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
        for line in out.splitlines():
            if "Chip" in line and "Apple" in line:
                chip = line.split(":", 1)[-1].strip()
                return chip.replace("Apple ", "")
    except Exception:
        pass
    return None


def _mps_available() -> bool:
    """Validate that PyTorch can execute Metal Performance Shaders."""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_built() and mps_backend.is_available())


def _xpu_available() -> bool:
    """Return whether the optional Intel XPU backend is currently available."""
    xpu_backend = getattr(torch, "xpu", None)
    return bool(xpu_backend is not None and xpu_backend.is_available())


def _available_devices() -> DeviceMap:
    """Discover compute targets the current PyTorch runtime can use."""
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "System CPU"
    devs: DeviceMap = {
        "cpu": {
            "name": f"CPU · {cpu_name}",
            "hardware_name": cpu_name,
            "type": "cpu",
            "compute_classes": ["cpu"],
            "available": True,
        }
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devs[f"cuda:{i}"] = {
                "name": f"GPU · {props.name}",
                "hardware_name": props.name,
                "type": "cuda",
                "compute_classes": ["gpu"],
                "index": i,
                "memory_gb": round(props.total_memory / 1024**3, 1),
                "available": True,
            }

    if _xpu_available():
        xpu_backend = torch.xpu
        try:
            count = xpu_backend.device_count()
        except Exception:
            count = 0
        for i in range(count):
            name = (
                xpu_backend.get_device_name(i)
                if hasattr(xpu_backend, "get_device_name")
                else f"Intel XPU {i}"
            )
            devs[f"xpu:{i}"] = {
                "name": f"GPU · {name}",
                "hardware_name": name,
                "type": "xpu",
                "compute_classes": ["gpu"],
                "index": i,
                "available": True,
            }

    if _mps_available():
        chip = _get_apple_chip() or "Apple Silicon"
        devs["mps"] = {
            "name": f"GPU · Apple {chip} (Metal)",
            "hardware_name": f"Apple {chip}",
            "type": "mps",
            "compute_classes": ["gpu"],
            "chip": chip,
            "available": True,
        }

    return devs


def _default_device_key() -> str:
    """Choose the fastest supported inference device."""
    devs = _available_devices()
    if any(k.startswith("cuda:") for k in devs):
        return "cuda:0"
    if "mps" in devs:
        return "mps"
    xpu_keys = [k for k in devs if k.startswith("xpu:")]
    if xpu_keys:
        return xpu_keys[0]
    return "cpu"


def _resolve(requested: str) -> torch.device:
    """Resolve a requested device key to a torch.device."""
    avail = _available_devices()
    if requested == "auto":
        return torch.device(_default_device_key())
    if requested not in avail:
        raise ValueError(f"Device '{requested}' unavailable. Options: {list(avail)}")
    return torch.device(requested)


def _onnx_provider_candidates(device: str) -> list[str]:
    """Map a PyTorch-style device string to ordered ONNX Runtime providers.

    ONNX Runtime does not understand PyTorch device names like ``mps`` or
    ``cuda:0``. This helper translates those names into execution-provider
    names and always appends CPU as the safe final fallback.
    """

    requested = (device or "auto").lower()
    if requested == "auto":
        requested = _default_device_key()

    providers: list[str]
    if requested.startswith("cuda"):
        providers = ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
    elif requested in {"mps", "metal", "coreml", "ane"}:
        providers = ["CoreMLExecutionProvider"]
    elif requested.startswith("xpu") or requested.startswith("intel"):
        providers = ["OpenVINOExecutionProvider", "DnnlExecutionProvider"]
    elif requested.startswith("rocm") or requested.startswith("hip"):
        providers = ["ROCMExecutionProvider", "MIGraphXExecutionProvider"]
    elif requested.startswith("dml") or requested.startswith("directml"):
        providers = ["DmlExecutionProvider"]
    else:
        providers = []

    providers.append("CPUExecutionProvider")
    return list(dict.fromkeys(providers))


def _onnx_providers_for_device(device: str, available: Sequence[str]) -> list[str]:
    """Return ONNX Runtime providers supported by this runtime for a PyTorch device."""

    available_set = set(available)
    candidates = _onnx_provider_candidates(device)
    selected = [provider for provider in candidates if provider in available_set]
    if "CPUExecutionProvider" in available_set and "CPUExecutionProvider" not in selected:
        selected.append("CPUExecutionProvider")
    return selected or list(available)


def _acceleration_checks(devs: DeviceMap) -> dict[str, dict[str, Any]]:
    """Verify that discovered accelerators can execute tensor operations."""
    checks: dict[str, dict[str, Any]] = {}

    def _probe(device_key: str, label: str) -> None:
        try:
            dev = torch.device(device_key)
            x = torch.randn((16, 16), device=dev)
            y = torch.mm(x, x.transpose(0, 1))
            _ = float(y.mean().detach().cpu().item())
            checks[label] = {"available": True, "operational": True}
        except Exception as exc:
            checks[label] = {"available": True, "operational": False, "error": str(exc)}

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
