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
    ckpt = tmp_path / "cache" / "hub" / "checkpoints"
    ckpt.mkdir(parents=True)
    for name in ("midas_v21_small_256.pt", "dpt_hybrid_384.pt", "dpt_large_384.pt"):
        (ckpt / name).write_bytes(b"fake")
    env = os.environ.copy(); env.update({"TORCH_HOME": str(tmp_path / "cache"), "DEPTHLENS_ONNX_DIR": str(tmp_path / "onnx"), "DEPTHLENS_DISABLE_MODEL_DOWNLOADS": "1"})
    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 0

MANAGER = ROOT / "scripts" / "manage_model_assets.py"


def _fake_cache(tmp_path: Path, models: list[str]) -> Path:
    repo = tmp_path / "cache" / "hub" / "intel-isl_MiDaS_master"
    (repo / "midas").mkdir(parents=True)
    (repo / "hubconf.py").write_text("# fake")
    ckpt = tmp_path / "cache" / "hub" / "checkpoints"
    ckpt.mkdir(parents=True)
    names = {
        "midas_small": "midas_v21_small_256.pt",
        "dpt_hybrid": "dpt_hybrid_384.pt",
        "dpt_large": "dpt_large_384.pt",
    }
    for model in models:
        (ckpt / names[model]).write_bytes(b"fake")
    return tmp_path / "cache"


def test_manage_model_assets_verify_all_passes_with_fake_cache(tmp_path: Path) -> None:
    cache = _fake_cache(tmp_path, ["midas_small", "dpt_hybrid", "dpt_large"])
    env = os.environ.copy(); env.update({"TORCH_HOME": str(cache), "DEPTHLENS_ONNX_DIR": str(tmp_path / "onnx")})
    proc = subprocess.run([sys.executable, str(MANAGER), "verify", "--models", "all", "--json"], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 0
    assert '"ready": true' in proc.stdout


def test_manage_model_assets_verify_all_fails_when_one_missing(tmp_path: Path) -> None:
    cache = _fake_cache(tmp_path, ["midas_small", "dpt_hybrid"])
    env = os.environ.copy(); env.update({"TORCH_HOME": str(cache), "DEPTHLENS_ONNX_DIR": str(tmp_path / "onnx")})
    proc = subprocess.run([sys.executable, str(MANAGER), "verify", "--models", "all"], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 1
    assert "DPT_Large" in proc.stdout


def test_manage_model_assets_small_does_not_require_compare_models(tmp_path: Path) -> None:
    cache = _fake_cache(tmp_path, ["midas_small"])
    env = os.environ.copy(); env.update({"TORCH_HOME": str(cache), "DEPTHLENS_ONNX_DIR": str(tmp_path / "onnx")})
    proc = subprocess.run([sys.executable, str(MANAGER), "verify", "--models", "small"], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 0
