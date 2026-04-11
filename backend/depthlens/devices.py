import os
import platform
import subprocess

import torch


def get_apple_chip() -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=3,
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
        ).stdout
        for line in out.splitlines():
            if "Chip" in line and "Apple" in line:
                chip = line.split(":", 1)[-1].strip()
                return chip.replace("Apple ", "")
    except Exception:
        pass
    return None


def available_devices() -> dict:
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "System CPU"
    devices = {
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
            devices[f"cuda:{i}"] = {
                "name": f"GPU · {props.name}",
                "hardware_name": props.name,
                "type": "cuda",
                "compute_classes": ["gpu"],
                "index": i,
                "memory_gb": round(props.total_memory / 1024**3, 1),
                "available": True,
            }
    if torch.backends.mps.is_available():
        chip = get_apple_chip() or "Apple Silicon"
        devices["mps"] = {
            "name": f"GPU/NPU · Apple {chip}",
            "hardware_name": f"Apple {chip}",
            "type": "mps",
            "compute_classes": ["gpu", "npu"],
            "chip": chip,
            "available": True,
        }
    return devices


def default_device_key() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_device(requested: str) -> torch.device:
    devices = available_devices()
    if requested == "auto":
        return torch.device(default_device_key())
    if requested not in devices:
        raise ValueError(f"Device '{requested}' unavailable. Options: {list(devices)}")
    if requested == "mps":
        return torch.device("mps")
    return torch.device(requested)
