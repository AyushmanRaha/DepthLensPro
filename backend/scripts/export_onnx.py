"""Export MiDaS Torch Hub weights to dynamic ONNX Runtime graph files.

Usage:
    python backend/scripts/export_onnx.py --model MiDaS_small
    python backend/scripts/export_onnx.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.depth_models import MIDAS_REPO, ONNX_INPUT_SIZE, onnx_model_path  # noqa: E402
from backend.services.inference import SUPPORTED_MODELS  # noqa: E402


def export_model(model_name: str, output_dir: Path, opset: int = 17) -> Path:
    """Load MiDaS from Torch Hub and export a dynamic BCHW ONNX graph."""

    output_path = onnx_model_path(model_name, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = torch.hub.load(MIDAS_REPO, model_name, trust_repo=True)  # type: ignore[no-untyped-call]
    model.eval().cpu()

    dummy = torch.zeros((1, 3, ONNX_INPUT_SIZE[0], ONNX_INPUT_SIZE[1]), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (dummy,),
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["depth"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "depth": {0: "batch", 1: "height", 2: "width"},
            },
        )

    onnx = __import__("onnx")
    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DepthLens Pro MiDaS models to ONNX.")
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), default="MiDaS_small")
    parser.add_argument("--all", action="store_true", help="Export every supported MiDaS model.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=onnx_model_path("MiDaS_small").parent,
        help="Directory for generated .onnx files.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = sorted(SUPPORTED_MODELS) if args.all else [args.model]
    for model_name in models:
        output_path = export_model(model_name, args.output_dir, args.opset)
        print(f"Exported {model_name} → {output_path}")


if __name__ == "__main__":
    main()
