"""Model asset manifest, validation, and status helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from backend.model_registry import MODEL_REGISTRY, REPO_ROOT, get_model_spec, resolve_onnx_path

MANIFEST_PATH = REPO_ROOT / "models" / "manifest.json"
PYTORCH_CACHE_DIR = Path(os.getenv("DEPTHLENS_TORCH_CACHE_DIR", REPO_ROOT / "models" / "pytorch"))
MIN_ONNX_BYTES = 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "assets": [], "missing": True, "path": os.fspath(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("path", os.fspath(path))
            return data
    except Exception as exc:
        return {
            "schema_version": 1,
            "assets": [],
            "error": f"manifest_invalid: {exc}",
            "path": os.fspath(path),
        }
    return {"schema_version": 1, "assets": [], "error": "manifest_invalid", "path": os.fspath(path)}


def _manifest_asset_map(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for item in manifest.get("assets") or []:
        if isinstance(item, dict):
            out[(str(item.get("model_id")), str(item.get("engine")))] = item
    return out


def validate_onnx_asset(
    model_id: str, *, deep: bool = False, device: str = "cpu"
) -> dict[str, Any]:
    spec = get_model_spec(model_id)
    resolved = resolve_onnx_path(model_id)
    path_value = resolved.get("onnx_path")
    expected = resolved.get("expected_path")
    status: dict[str, Any] = {
        "model_id": spec.model_id,
        "display_name": spec.display_name,
        "engine": "onnx",
        "required": True,
        "filename": spec.onnx_filename,
        "path": path_value,
        "expected_path": expected,
        "input_shape": [1, 3, *spec.input_size],
        "exists": False,
        "valid": False,
        "validation_status": "missing",
    }
    if not path_value:
        status["error_code"] = "MODEL_FILE_MISSING"
        return status
    path = Path(str(path_value))
    status["exists"] = path.is_file()
    if not path.is_file():
        status["error_code"] = "MODEL_FILE_MISSING"
        return status
    size = path.stat().st_size
    status["size_bytes"] = size
    status["sha256"] = sha256_file(path)
    if size < MIN_ONNX_BYTES:
        status.update({"validation_status": "too_small", "error_code": "MODEL_FILE_TOO_SMALL"})
        return status
    status["validation_status"] = "file_validated"
    status["valid"] = True
    if deep:
        try:
            onnx = __import__("onnx")
            model = onnx.load(os.fspath(path), load_external_data=True)
            onnx.checker.check_model(model)
            status["onnx_checker"] = "ok"
            from backend.services.onnx_diagnostics import create_onnx_session

            session = create_onnx_session(model_id, device, model_path=path)
            status["runtime"] = session
            status["valid"] = bool(session.get("ok"))
            status["validation_status"] = (
                "runtime_validated" if session.get("ok") else "runtime_invalid"
            )
            if not session.get("ok"):
                status["error_code"] = session.get("error_code") or "ONNX_PROVIDER_UNAVAILABLE"
        except Exception as exc:
            status.update(
                {
                    "valid": False,
                    "validation_status": "deep_validation_failed",
                    "error_code": "MODEL_CHECKSUM_FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return status


def pytorch_asset_status(model_id: str) -> dict[str, Any]:
    spec = get_model_spec(model_id)
    cache_dir = PYTORCH_CACHE_DIR.expanduser()
    candidates = [cache_dir / f"{spec.model_id}.pt", cache_dir / f"{spec.pytorch_model_name}.pt"]
    existing = next((p for p in candidates if p.is_file() and p.stat().st_size > 0), None)
    return {
        "model_id": spec.model_id,
        "display_name": spec.display_name,
        "engine": "pytorch",
        "required": True,
        "path": os.fspath(existing) if existing else None,
        "expected_paths": [os.fspath(p) for p in candidates],
        "exists": existing is not None,
        "valid": existing is not None,
        "size_bytes": existing.stat().st_size if existing else None,
        "sha256": sha256_file(existing) if existing else None,
        "validation_status": "file_present" if existing else "missing_or_torchhub_cache_only",
        "note": "Setup pre-caches Torch Hub weights/transforms. Direct .pt files may be absent when Torch Hub cache is used.",
    }


def model_status(*, deep: bool = False, device: str = "cpu") -> dict[str, Any]:
    manifest = load_manifest()
    assets = []
    onnx_ok = True
    pytorch_ok = True
    for model_id in MODEL_REGISTRY:
        onnx = validate_onnx_asset(model_id, deep=deep, device=device)
        pt = pytorch_asset_status(model_id)
        assets.extend([onnx, pt])
        onnx_ok = onnx_ok and bool(onnx.get("valid"))
        pytorch_ok = pytorch_ok and bool(pt.get("valid"))
    return {
        "status": "ok" if onnx_ok else "setup_incomplete",
        "schema_version": 1,
        "manifest": manifest,
        "manifest_path": os.fspath(MANIFEST_PATH),
        "models_dir": os.fspath(REPO_ROOT / "models"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.system(),
        "arch": platform.machine(),
        "onnx_all_valid": onnx_ok,
        "pytorch_all_visible": pytorch_ok,
        "assets": assets,
        "models": {
            model_id: {
                "onnx": next(
                    a for a in assets if a["model_id"] == model_id and a["engine"] == "onnx"
                ),
                "pytorch": next(
                    a for a in assets if a["model_id"] == model_id and a["engine"] == "pytorch"
                ),
            }
            for model_id in MODEL_REGISTRY
        },
    }


def write_manifest(path: Path = MANIFEST_PATH, *, deep: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = model_status(deep=deep)
    manifest = {
        "schema_version": 1,
        "generated_at": status["generated_at"],
        "platform": status["platform"],
        "arch": status["arch"],
        "assets": status["assets"],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
