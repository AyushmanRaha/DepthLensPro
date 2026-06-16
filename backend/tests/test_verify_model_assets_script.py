from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_model_assets.py"


def test_verify_model_assets_fails_without_assets(tmp_path: Path) -> None:
    env = os.environ.copy(); env.update({"TORCH_HOME": str(tmp_path / "cache"), "DEPTHLENS_ONNX_DIR": str(tmp_path / "onnx"), "DEPTHLENS_DISABLE_MODEL_DOWNLOADS": "1"})
    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 1
    assert "FIX:" in proc.stdout


def test_verify_model_assets_passes_with_fake_cache(tmp_path: Path) -> None:
    repo = tmp_path / "cache" / "hub" / "intel-isl_MiDaS_master"
    (repo / "midas").mkdir(parents=True)
    (repo / "hubconf.py").write_text("# fake")
    env = os.environ.copy(); env.update({"TORCH_HOME": str(tmp_path / "cache"), "DEPTHLENS_ONNX_DIR": str(tmp_path / "onnx"), "DEPTHLENS_DISABLE_MODEL_DOWNLOADS": "1"})
    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 0
