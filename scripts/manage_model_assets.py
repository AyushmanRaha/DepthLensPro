#!/usr/bin/env python3
"""Install and verify DepthLens Pro PyTorch MiDaS model assets."""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = {
    "midas_small": {"hub": "MiDaS_small", "checkpoint": "midas_v21_small_256.pt"},
    "dpt_hybrid": {"hub": "DPT_Hybrid", "checkpoint": "dpt_hybrid_384.pt"},
    "dpt_large": {"hub": "DPT_Large", "checkpoint": "dpt_large_384.pt"},
}
GROUPS = {"small": ["midas_small"], "compare": list(MODELS), "all": list(MODELS)}


def log(msg: str) -> None:
    print(msg, flush=True)


def torch_home() -> Path:
    return Path(os.environ.get("TORCH_HOME", ROOT / "models" / "torch-cache")).expanduser().resolve()


def checkpoint_dir(home: Path) -> Path:
    return home / "hub" / "checkpoints"


def repo_candidates(home: Path) -> list[Path]:
    hub = home / "hub"
    out = [hub / "intel-isl_MiDaS_master", hub / "intel-isl_MiDaS_main"]
    if hub.exists():
        out.extend(sorted(p for p in hub.glob("intel-isl_MiDaS*") if p.is_dir()))
    return list(dict.fromkeys(out))


def repo_cached(home: Path) -> tuple[bool, str | None]:
    for candidate in repo_candidates(home):
        if (candidate / "hubconf.py").is_file() and ((candidate / "midas").is_dir() or (candidate / "transforms.py").is_file()):
            return True, str(candidate)
    return False, None


def model_list(value: str) -> list[str]:
    if value in GROUPS:
        return GROUPS[value]
    names = []
    for part in value.split(','):
        key = part.strip().lower().replace('-', '_')
        aliases = {v["hub"].lower().replace('-', '_'): k for k, v in MODELS.items()}
        key = aliases.get(key, key)
        if key not in MODELS:
            raise SystemExit(f"Unknown --models value '{part}'. Use small, compare, all, or one of {', '.join(MODELS)}")
        names.append(key)
    return list(dict.fromkeys(names))


def verify(models: list[str], *, json_mode: bool = False) -> dict[str, Any]:
    home = torch_home(); repo_ok, repo_path = repo_cached(home)
    ckpt = checkpoint_dir(home)
    per = {}
    for model in models:
        expected = ckpt / MODELS[model]["checkpoint"]
        per[MODELS[model]["hub"]] = {"ready": repo_ok and expected.is_file() and expected.stat().st_size > 0, "checkpoint": str(expected), "checkpoint_exists": expected.is_file(), "repo_cached": repo_ok}
    ready = all(item["ready"] for item in per.values())
    payload = {"ready": ready, "torch_home": str(home), "hub_dir": str(home / "hub"), "checkpoint_dir": str(ckpt), "midas_repo_cached": repo_ok, "midas_repo_path": repo_path, "models": per, "requested_models": [MODELS[m]["hub"] for m in models]}
    if json_mode:
        log(json.dumps(payload, indent=2))
    else:
        log("Checking Python runtime")
        log(f"Python: {sys.executable}")
        log(f"TORCH_HOME: {home}")
        log("Checking Torch Hub cache")
        log(f"MiDaS repo cached: {repo_ok} ({repo_path or 'missing'})")
        log("Verifying cached assets")
        for name, item in per.items():
            log(f"  {name}: {'ready' if item['ready'] else 'missing'} checkpoint={item['checkpoint']}")
        if not ready:
            missing = [name for name, item in per.items() if not item["ready"]]
            log("ERROR: Missing PyTorch MiDaS assets: " + ", ".join(missing))
            log("FIX: python scripts/manage_model_assets.py install --models all")
    return payload


def child_code(model: str, home: str, force: bool) -> str:
    hub_name = MODELS[model]["hub"]
    force_reload = "True" if force else "False"
    return f"""
import os, sys, time
os.environ['TORCH_HOME'] = {home!r}
print('Checking torch import', flush=True)
import torch
print('Installing MiDaS repo/transforms', flush=True)
t0=time.perf_counter()
torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True, force_reload={force_reload})
print('Installing {hub_name}', flush=True)
torch.hub.load('intel-isl/MiDaS', {hub_name!r}, trust_repo=True, force_reload={force_reload})
print('DONE {hub_name} elapsed={{:.1f}}s'.format(time.perf_counter()-t0), flush=True)
"""


def run_step(model: str, timeout: int, force: bool) -> None:
    home = torch_home(); home.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TORCH_HOME": str(home), "PYTHONUNBUFFERED": "1"}
    started = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-u", "-c", child_code(model, str(home), force)], cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill(); out, _ = proc.communicate()
        log(out or "")
        raise RuntimeError(f"Timed out installing {MODELS[model]['hub']} after {timeout}s. Network/GitHub may be blocked.")
    if out: print(out, end="", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed installing {MODELS[model]['hub']} (exit {proc.returncode}). Check network access to GitHub/Torch Hub.")
    log(f"Elapsed for {MODELS[model]['hub']}: {time.perf_counter()-started:.1f}s")


def install(models: list[str], retries: int, timeout: int, force: bool) -> int:
    log("Checking Python runtime")
    log(f"Python: {sys.executable}")
    log(f"TORCH_HOME: {torch_home()}")
    log("Checking torch import")
    try: __import__('torch')
    except Exception as exc:
        log(f"ERROR: torch import failed: {exc}"); log("Next: install backend requirements in the repo venv, then retry setup."); return 2
    log("Checking Torch Hub cache")
    torch_home().mkdir(parents=True, exist_ok=True); (torch_home()/"hub"/"checkpoints").mkdir(parents=True, exist_ok=True)
    for model in models:
        for attempt in range(1, retries + 2):
            log(f"Installing {MODELS[model]['hub']} (attempt {attempt}/{retries+1}, timeout {timeout}s)")
            try:
                run_step(model, timeout, force)
                break
            except Exception as exc:
                log(f"ERROR: {exc}")
                if attempt > retries:
                    log("Next: verify network/GitHub access or copy a pre-cached models/torch-cache directory, then run: python scripts/manage_model_assets.py verify --models all")
                    return 1
                time.sleep(min(5, attempt))
    return 0 if verify(models)["ready"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ("verify", "install"):
        sp = sub.add_parser(cmd); sp.add_argument("--models", default="all"); sp.add_argument("--json", action="store_true"); sp.add_argument("--offline", action="store_true"); sp.add_argument("--timeout-seconds", type=int, default=900); sp.add_argument("--retries", type=int, default=1); sp.add_argument("--force", action="store_true")
    sub.add_parser("paths")
    args = parser.parse_args()
    os.environ["TORCH_HOME"] = str(torch_home())
    if args.command == "paths":
        log(json.dumps({"repo_root": str(ROOT), "torch_home": str(torch_home()), "hub_dir": str(torch_home()/"hub"), "checkpoint_dir": str(checkpoint_dir(torch_home()))}, indent=2)); return 0
    models = model_list(args.models)
    if args.command == "verify":
        return 0 if verify(models, json_mode=args.json)["ready"] else 1
    return install(models, retries=args.retries, timeout=args.timeout_seconds, force=args.force)

if __name__ == "__main__":
    raise SystemExit(main())
