"""Export and validate MiDaS Torch Hub weights as static ONNX files."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.depth_models import MIDAS_REPO, onnx_model_path
from backend.model_registry import (
    MODEL_REGISTRY,
    get_model_spec,
    normalize_model_id,
    onnx_output_path,
)
from backend.services.onnx_diagnostics import create_onnx_session

OPTIONAL_ONNX_MODELS = {"dpt_hybrid", "dpt_large"}


def _torch_hub_load(*args: Any, **kwargs: Any) -> Any:
    hub_load = cast(Callable[..., Any], torch.hub.load)
    return hub_load(*args, **kwargs)


def parse_model_list(value: str) -> list[str]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if raw == ["all"]:
        return sorted(MODEL_REGISTRY)
    invalid = [item for item in raw if item not in MODEL_REGISTRY]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported model(s): {', '.join(invalid)}")
    return raw


def configure_certificates() -> dict[str, str | None]:
    """Point Python HTTPS clients at certifi when available."""
    try:
        import certifi

        bundle = certifi.where()
    except Exception:
        bundle = None
    if bundle:
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
        os.environ.setdefault("CURL_CA_BUNDLE", bundle)
    os.environ.setdefault("TORCH_HOME", os.fspath(ROOT / "models" / "torch-cache"))
    return {"certifi_bundle": bundle, "torch_home": os.environ.get("TORCH_HOME")}


def _dummy_input(shape: tuple[int, int, int, int]) -> Any:
    try:
        import numpy as np

        return np.zeros(shape, dtype=np.float32)
    except Exception:
        return None


def _validate_onnx(path: Path, model_id: str) -> tuple[bool, str | None, dict[str, Any]]:
    spec = get_model_spec(model_id)
    shape = (1, 3, spec.input_size[0], spec.input_size[1])
    detail: dict[str, Any] = {
        "input_shape": list(shape),
        "file_size": path.stat().st_size if path.exists() else 0,
    }
    if not path.is_file() or path.stat().st_size <= 0:
        return False, "exported ONNX file is missing or empty", detail | {"status": "empty"}
    try:
        onnx = __import__("onnx")
        exported = onnx.load(os.fspath(path), load_external_data=True)
        onnx.checker.check_model(exported)
        detail["checker"] = "ok"
    except ImportError:
        detail["checker"] = "skipped_onnx_not_installed"
    except Exception as exc:
        return (
            False,
            f"onnx.checker failed: {type(exc).__name__}: {exc}",
            detail | {"status": "invalid_checker"},
        )
    session = create_onnx_session(model_id, "cpu", model_path=path)
    if not session.get("ok"):
        return (
            False,
            str(session.get("technical_detail") or session.get("message")),
            detail
            | {
                "status": "invalid_session",
                "session": {k: v for k, v in session.items() if k != "session"},
            },
        )
    try:
        ort_session = session["session"]
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        dummy = _dummy_input(shape)
        if dummy is not None:
            outputs = ort_session.run([output_name], {input_name: dummy})
            detail["dummy_outputs"] = [list(getattr(out, "shape", [])) for out in outputs]
        detail["providers"] = session.get("providers_used", [])
    except Exception as exc:
        return (
            False,
            f"dummy inference failed: {type(exc).__name__}: {exc}",
            detail | {"status": "invalid_dummy_inference"},
        )
    return True, None, detail | {"status": "available"}


def _quarantine(path: Path, reason: str) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = path.with_suffix(path.suffix + f".{stamp}.failed")
    try:
        os.replace(path, dest)
        (dest.with_suffix(dest.suffix + ".txt")).write_text(reason, encoding="utf-8")
        return dest
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass
        return None


def export_model_to_onnx(
    model_id: str,
    force: bool = False,
    input_shape: tuple[int, int, int, int] | None = None,
    output_dir: Path | None = None,
    opset: int = 17,
) -> dict[str, Any]:
    configure_certificates()
    canonical = normalize_model_id(model_id)
    spec = get_model_spec(canonical)
    path = onnx_output_path(canonical, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = input_shape or (1, 3, spec.input_size[0], spec.input_size[1])

    if path.is_file() and path.stat().st_size > 0 and not force:
        ok, error, validation = _validate_onnx(path, canonical)
        if ok:
            return {
                "ok": True,
                "model_id": canonical,
                "input_shape": list(shape),
                "onnx_path": os.fspath(path),
                "reused_existing": True,
                "export_strategy": "cached_valid_onnx",
                "opset": opset,
                "validation": validation,
                "file_size": path.stat().st_size,
                "error_code": None,
                "message": "Using cached ONNX model.",
                "technical_detail": None,
            }
        quarantined = _quarantine(path, error or "invalid cached ONNX")
        return {
            "ok": False,
            "model_id": canonical,
            "input_shape": list(shape),
            "onnx_path": os.fspath(path),
            "quarantined_path": os.fspath(quarantined) if quarantined else None,
            "reused_existing": False,
            "export_strategy": None,
            "opset": opset,
            "validation": validation,
            "error_code": "ONNX_VALIDATION_FAILED",
            "message": "Cached ONNX failed validation and was removed/quarantined.",
            "technical_detail": error,
        }

    model = _torch_hub_load(
        MIDAS_REPO,
        spec.pytorch_model_name,
        trust_repo=True,
        skip_validation=True,
    )
    model.eval().cpu()
    dummy = torch.zeros(shape, dtype=torch.float32)
    errors: list[str] = []
    signature = inspect.signature(torch.onnx.export)
    strategies = ["legacy_torch_onnx_export", "dynamo_export"]
    for strategy in strategies:
        tmp_path: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp.onnx"
            )
            os.close(fd)
            tmp_path = Path(tmp_name)
            kwargs: dict[str, Any] = {
                "export_params": True,
                "opset_version": opset,
                "do_constant_folding": True,
                "input_names": ["input"],
                "output_names": ["output"],
            }
            if "dynamo" in signature.parameters:
                kwargs["dynamo"] = strategy == "dynamo_export"
            elif strategy == "dynamo_export":
                continue
            if "external_data" in signature.parameters:
                kwargs["external_data"] = False
            with torch.no_grad():
                torch.onnx.export(model, (dummy,), os.fspath(tmp_path), **kwargs)
            ok, error, validation = _validate_onnx(tmp_path, canonical)
            if ok:
                os.replace(tmp_path, path)
                tmp_path = None
                return {
                    "ok": True,
                    "model_id": canonical,
                    "input_shape": list(shape),
                    "onnx_path": os.fspath(path),
                    "reused_existing": False,
                    "export_strategy": strategy,
                    "opset": opset,
                    "validation": validation,
                    "file_size": path.stat().st_size,
                    "runtime_providers": validation.get("providers", []),
                    "error_code": None,
                    "message": f"Exported {spec.display_name} with {strategy}.",
                    "technical_detail": None,
                }
            errors.append(f"{strategy}: validation failed: {error}")
            _quarantine(tmp_path, error or "validation failed")
            tmp_path = None
        except Exception as exc:
            errors.append(f"{strategy}: {type(exc).__name__}: {exc}")
        finally:
            if tmp_path is not None:
                _quarantine(tmp_path, "temporary export failed")
    return {
        "ok": False,
        "model_id": canonical,
        "input_shape": list(shape),
        "onnx_path": os.fspath(path),
        "reused_existing": False,
        "export_strategy": None,
        "opset": opset,
        "error_code": "ONNX_EXPORT_FAILED",
        "optional": canonical in OPTIONAL_ONNX_MODELS,
        "message": "ONNX export failed; PyTorch inference remains available.",
        "technical_detail": "\n".join(errors),
    }


def validate_model(model_id: str, output_dir: Path | None = None) -> dict[str, Any]:
    canonical = normalize_model_id(model_id)
    path = onnx_output_path(canonical, output_dir)
    ok, error, detail = _validate_onnx(path, canonical)
    return {
        "ok": ok,
        "model_id": canonical,
        "onnx_path": os.fspath(path),
        "validation": detail,
        "error_code": None if ok else str(detail.get("status", "invalid")).upper(),
        "message": "ONNX validation passed." if ok else (error or "ONNX validation failed"),
        "technical_detail": error,
    }


def export_model(model_name: str, output_dir: Path, opset: int = 17) -> Path:
    result = export_model_to_onnx(model_name, output_dir=output_dir, opset=opset)
    if not result.get("ok"):
        raise RuntimeError(result.get("technical_detail") or result.get("message"))
    return Path(str(result["onnx_path"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DepthLens Pro MiDaS models to ONNX.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="midas_small")
    parser.add_argument(
        "--models",
        type=parse_model_list,
        help="Comma-separated model IDs to export/validate, or all.",
    )
    parser.add_argument("--all", action="store_true", help="Export every supported MiDaS model.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=onnx_model_path("midas_small").parent,
        help="Directory for generated .onnx files.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if a valid cached ONNX file exists."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing ONNX files without exporting.",
    )
    parser.add_argument(
        "--strict",
        "--require-all",
        dest="strict",
        action="store_true",
        help="Fail if any selected model is missing/invalid, including optional DPT models.",
    )
    parser.add_argument(
        "--allow-optional-dpt-failure",
        action="store_true",
        default=True,
        help="Allow DPT Hybrid/Large failures while requiring MiDaS Small.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = sorted(MODEL_REGISTRY) if args.all else (args.models if args.models else [args.model])
    failures = []
    for index, model_name in enumerate(models, 1):
        action = "validating" if args.validate_only else "exporting"
        print(f"[ONNX] [{index}/{len(models)}] {action} {model_name}...", flush=True)
        result = (
            validate_model(model_name, args.output_dir)
            if args.validate_only
            else export_model_to_onnx(
                model_name, force=args.force, output_dir=args.output_dir, opset=args.opset
            )
        )
        print(json.dumps(result, indent=2, default=str), flush=True)
        if not result.get("ok"):
            optional = result.get("model_id") in OPTIONAL_ONNX_MODELS and not args.strict
            if not optional:
                failures.append(result)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
