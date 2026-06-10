"""Export MiDaS Torch Hub weights to static ONNX Runtime graph files.

Usage:
    python backend/scripts/export_onnx.py --model midas_small
    python backend/scripts/export_onnx.py --all
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.depth_models import MIDAS_REPO, onnx_model_path  # noqa: E402
from backend.model_registry import (  # noqa: E402
    MODEL_REGISTRY,
    get_model_spec,
    normalize_model_id,
    onnx_output_path,
)
from backend.services.onnx_diagnostics import create_onnx_session  # noqa: E402


def _validate_onnx(path: Path, model_id: str) -> tuple[bool, str | None]:
    if not path.is_file() or path.stat().st_size <= 0:
        return False, "exported ONNX file is missing or empty"
    try:
        onnx = __import__("onnx")
        exported = onnx.load(os.fspath(path))
        onnx.checker.check_model(exported)
    except ImportError:
        pass
    except Exception as exc:
        return False, f"onnx.checker failed: {type(exc).__name__}: {exc}"
    session = create_onnx_session(model_id, "cpu", model_path=path)
    if not session.get("ok"):
        return False, str(session.get("technical_detail") or session.get("message"))
    return True, None


def export_model_to_onnx(
    model_id: str,
    force: bool = False,
    input_shape: tuple[int, int, int, int] | None = None,
    output_dir: Path | None = None,
    opset: int = 17,
) -> dict[str, Any]:
    """Export a model with static shapes and validate the result before success."""

    canonical = normalize_model_id(model_id)
    spec = get_model_spec(canonical)
    path = onnx_output_path(canonical, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = input_shape or (1, 3, spec.input_size[0], spec.input_size[1])

    if path.is_file() and path.stat().st_size > 0 and not force:
        ok, error = _validate_onnx(path, canonical)
        if ok:
            return {
                "ok": True,
                "model_id": canonical,
                "onnx_path": os.fspath(path),
                "reused_existing": True,
                "export_strategy": "cached_valid_onnx",
                "error_code": None,
                "message": "Using cached ONNX model.",
                "technical_detail": None,
            }
        path.unlink(missing_ok=True)

    model = torch.hub.load(MIDAS_REPO, spec.pytorch_model_name, trust_repo=True)  # type: ignore[no-untyped-call]
    model.eval().cpu()
    dummy = torch.zeros(shape, dtype=torch.float32)
    errors: list[str] = []

    strategies = ["legacy_torch_onnx_export", "dynamo_export"]
    for strategy in strategies:
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp.onnx", delete=False
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
            kwargs: dict[str, Any] = {
                "export_params": True,
                "opset_version": opset,
                "do_constant_folding": True,
                "input_names": ["input"],
                "output_names": ["output"],
            }
            signature = inspect.signature(torch.onnx.export)
            if "dynamo" in signature.parameters:
                kwargs["dynamo"] = strategy == "dynamo_export"
            elif strategy == "dynamo_export":
                continue
            with torch.no_grad():
                torch.onnx.export(model, (dummy,), os.fspath(tmp_path), **kwargs)
            ok, error = _validate_onnx(tmp_path, canonical)
            if ok:
                os.replace(tmp_path, path)
                tmp_path = None
                return {
                    "ok": True,
                    "model_id": canonical,
                    "onnx_path": os.fspath(path),
                    "reused_existing": False,
                    "export_strategy": strategy,
                    "error_code": None,
                    "message": f"Exported {spec.display_name} with {strategy}.",
                    "technical_detail": None,
                }
            errors.append(f"{strategy}: validation failed: {error}")
        except Exception as exc:
            errors.append(f"{strategy}: {type(exc).__name__}: {exc}")
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    return {
        "ok": False,
        "model_id": canonical,
        "onnx_path": os.fspath(path),
        "reused_existing": False,
        "export_strategy": None,
        "error_code": "ONNX_EXPORT_FAILED",
        "message": "ONNX export failed; PyTorch inference remains available.",
        "technical_detail": "\n".join(errors),
    }


def export_model(model_name: str, output_dir: Path, opset: int = 17) -> Path:
    """Compatibility wrapper returning the exported path or raising on failure."""

    result = export_model_to_onnx(model_name, output_dir=output_dir, opset=opset)
    if not result.get("ok"):
        raise RuntimeError(result.get("technical_detail") or result.get("message"))
    return Path(str(result["onnx_path"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DepthLens Pro MiDaS models to ONNX.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="midas_small")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = sorted(MODEL_REGISTRY) if args.all else [args.model]
    for model_name in models:
        result = export_model_to_onnx(
            model_name, force=args.force, output_dir=args.output_dir, opset=args.opset
        )
        if result.get("ok"):
            print(f"Exported {model_name} → {result['onnx_path']} ({result['export_strategy']})")
        else:
            print(
                f"Failed {model_name}: {result['message']}\n{result.get('technical_detail')}",
                file=sys.stderr,
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
