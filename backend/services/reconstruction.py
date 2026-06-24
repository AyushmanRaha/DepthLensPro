"""Lightweight point-cloud reconstruction helpers for DepthLens Pro."""

from __future__ import annotations

import base64
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from backend.services import inference

_ALLOWED_FORMATS = {"ply", "obj"}
_ALLOWED_SAMPLING = {"grid", "stride", "random"}
_ALLOWED_COORDINATES = {"z_up", "y_up"}


def normalize_reconstruction_format(fmt: str | None) -> str:
    """Normalize a requested point-cloud export format."""

    value = (fmt or "ply").strip().lower()
    if value not in _ALLOWED_FORMATS:
        raise ValueError("export_format must be one of: ply, obj")
    return value


def _clamp_int(value: int | None, default: int, minimum: int, maximum: int, name: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return max(minimum, min(maximum, parsed))


def _clamp_float(
    value: float | None, default: float, minimum: float, maximum: float, name: str
) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return max(minimum, min(maximum, parsed))


def clamp_reconstruction_options(
    *,
    max_points: int | None = 120000,
    preview_points: int | None = 5000,
    focal_scale: float | None = 1.2,
    depth_scale: float | None = 1.0,
    depth_near_percentile: float | None = 2.0,
    depth_far_percentile: float | None = 98.0,
    sampling: str | None = "grid",
    include_rgb: bool | None = True,
    coordinate_system: str | None = "y_up",
) -> dict[str, Any]:
    """Validate and normalize reconstruction options to safe MVP limits."""

    sampling_value = (sampling or "grid").strip().lower()
    if sampling_value not in _ALLOWED_SAMPLING:
        raise ValueError("sampling must be one of: grid, stride, random")
    coordinate_value = (coordinate_system or "y_up").strip().lower()
    if coordinate_value not in _ALLOWED_COORDINATES:
        raise ValueError("coordinate_system must be one of: y_up, z_up")

    near = _clamp_float(depth_near_percentile, 2.0, 0.0, 40.0, "depth_near_percentile")
    far = _clamp_float(depth_far_percentile, 98.0, 60.0, 100.0, "depth_far_percentile")
    if near >= far:
        raise ValueError("depth_near_percentile must be less than depth_far_percentile")

    return {
        "max_points": _clamp_int(max_points, 120000, 1000, 500000, "max_points"),
        "preview_points": _clamp_int(preview_points, 5000, 0, 20000, "preview_points"),
        "focal_scale": _clamp_float(focal_scale, 1.2, 0.25, 4.0, "focal_scale"),
        "depth_scale": _clamp_float(depth_scale, 1.0, 0.01, 100.0, "depth_scale"),
        "depth_near_percentile": near,
        "depth_far_percentile": far,
        "sampling": sampling_value,
        "include_rgb": bool(True if include_rgb is None else include_rgb),
        "coordinate_system": coordinate_value,
    }


def _sample_indices(
    ys: np.ndarray, xs: np.ndarray, total: int, max_points: int, sampling: str
) -> np.ndarray:
    if total <= max_points:
        return np.arange(total, dtype=np.int64)
    if sampling == "random":
        rng = np.random.default_rng(0)
        return np.sort(rng.choice(total, size=max_points, replace=False)).astype(np.int64)

    stride = max(1, int(math.ceil(math.sqrt(total / max_points))))
    selected = np.flatnonzero((ys % stride == 0) & (xs % stride == 0)).astype(np.int64)
    if selected.size >= max_points:
        grid_positions = np.linspace(0, selected.size - 1, num=max_points, dtype=np.int64)
        return selected[grid_positions]
    # Dense deterministic fallback if an offset grid undershoots because valid pixels are sparse.
    return np.linspace(0, total - 1, num=max_points, dtype=np.int64)


def depth_to_point_cloud(
    img_bgr: np.ndarray,
    depth: np.ndarray,
    *,
    max_points: int = 120000,
    focal_scale: float = 1.2,
    depth_scale: float = 1.0,
    depth_near_percentile: float = 2.0,
    depth_far_percentile: float = 98.0,
    sampling: str = "grid",
    include_rgb: bool = True,
    coordinate_system: str = "y_up",
) -> dict[str, Any]:
    """Project a normalized monocular depth map into an approximate colored point cloud."""

    options = clamp_reconstruction_options(
        max_points=max_points,
        preview_points=0,
        focal_scale=focal_scale,
        depth_scale=depth_scale,
        depth_near_percentile=depth_near_percentile,
        depth_far_percentile=depth_far_percentile,
        sampling=sampling,
        include_rgb=include_rgb,
        coordinate_system=coordinate_system,
    )
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim != 2:
        raise ValueError("depth must be a 2D array")
    img_arr = np.asarray(img_bgr)
    h, w = depth_arr.shape
    if img_arr.shape[:2] != (h, w):
        raise ValueError("image and depth dimensions must match")

    finite_depth = depth_arr[np.isfinite(depth_arr)]
    if finite_depth.size == 0:
        raise ValueError("depth map contains no finite values")
    lo = float(np.percentile(finite_depth, options["depth_near_percentile"]))
    hi = float(np.percentile(finite_depth, options["depth_far_percentile"]))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo <= 1e-8:
        raise ValueError("depth map has degenerate range for reconstruction")

    z_norm = np.clip((depth_arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    z = z_norm * float(options["depth_scale"])
    valid = np.isfinite(z) & np.isfinite(depth_arr)
    ys, xs = np.nonzero(valid)
    total = int(ys.size)
    if total == 0:
        raise ValueError("depth map produced no valid reconstruction points")

    sample_idx = _sample_indices(
        ys, xs, total, int(options["max_points"]), str(options["sampling"])
    )
    ys = ys[sample_idx]
    xs = xs[sample_idx]
    z_values = z[ys, xs].astype(np.float32, copy=False)

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    focal_px = float(options["focal_scale"]) * max(w, h)
    x_values = ((xs.astype(np.float32) - cx) * z_values / focal_px).astype(np.float32)
    y_values = ((ys.astype(np.float32) - cy) * z_values / focal_px).astype(np.float32)
    if options["coordinate_system"] == "y_up":
        y_values = -y_values
    points = np.column_stack((x_values, y_values, z_values)).astype(np.float32, copy=False)
    point_valid = np.isfinite(points).all(axis=1)
    points = points[point_valid]

    colors: np.ndarray | None = None
    if options["include_rgb"]:
        sampled_bgr = img_arr[ys, xs][point_valid]
        colors = sampled_bgr[:, ::-1].astype(np.uint8, copy=False)

    if points.size:
        bounds_min = [round(float(v), 6) for v in points.min(axis=0).tolist()]
        bounds_max = [round(float(v), 6) for v in points.max(axis=0).tolist()]
    else:
        bounds_min = [0.0, 0.0, 0.0]
        bounds_max = [0.0, 0.0, 0.0]

    return {
        "points": points,
        "colors": colors,
        "metadata": {
            "point_count": int(points.shape[0]),
            "source_width": int(w),
            "source_height": int(h),
            "max_points": int(options["max_points"]),
            "sampling": options["sampling"],
            "include_rgb": bool(options["include_rgb"]),
            "coordinate_system": options["coordinate_system"],
            "focal_px": round(float(focal_px), 6),
            "cx": round(float(cx), 6),
            "cy": round(float(cy), 6),
            "depth_scale": float(options["depth_scale"]),
            "depth_near_percentile": float(options["depth_near_percentile"]),
            "depth_far_percentile": float(options["depth_far_percentile"]),
            "depth_min": round(float(lo), 6),
            "depth_max": round(float(hi), 6),
            "bounds": {"min": bounds_min, "max": bounds_max},
        },
    }


def serialize_ply(points: np.ndarray, colors: np.ndarray | None = None) -> bytes:
    """Serialize points and optional uint8 RGB colors as ASCII PLY bytes."""

    pts = np.asarray(points, dtype=np.float32)
    cols = None if colors is None else np.asarray(colors, dtype=np.uint8)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if cols is not None and cols.shape != pts.shape:
        raise ValueError("colors must have shape (N, 3)")

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {pts.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if cols is not None:
        header.extend(["property uchar red", "property uchar green", "property uchar blue"])
    header.append("end_header")

    lines = header
    if cols is None:
        lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts)
    else:
        lines.extend(
            f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}"
            for (x, y, z), (r, g, b) in zip(pts, cols, strict=True)
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def serialize_obj(points: np.ndarray, colors: np.ndarray | None = None) -> bytes:
    """Serialize points and optional RGB vertex colors as Wavefront OBJ bytes."""

    pts = np.asarray(points, dtype=np.float32)
    cols = None if colors is None else np.asarray(colors, dtype=np.uint8)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if cols is not None and cols.shape != pts.shape:
        raise ValueError("colors must have shape (N, 3)")

    lines = ["# generated by DepthLens Pro", f"# point count: {pts.shape[0]}"]
    if cols is None:
        lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts)
    else:
        for (x, y, z), (r, g, b) in zip(pts, cols, strict=True):
            lines.append(
                f"v {x:.6f} {y:.6f} {z:.6f} {int(r) / 255:.6f} "
                f"{int(g) / 255:.6f} {int(b) / 255:.6f}"
            )
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_preview_points(
    points: np.ndarray, colors: np.ndarray | None, limit: int
) -> list[list[float | int]]:
    """Build a compact JSON-safe deterministic preview point list."""

    if limit <= 0:
        return []
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return []
    count = min(int(limit), int(pts.shape[0]))
    indices = np.linspace(0, pts.shape[0] - 1, num=count, dtype=np.int64)
    cols = None if colors is None else np.asarray(colors, dtype=np.uint8)
    preview: list[list[float | int]] = []
    for idx in indices:
        x, y, z = pts[idx]
        if cols is None:
            r, g, b = 255, 255, 255
        else:
            r, g, b = (int(c) for c in cols[idx])
        preview.append([round(float(x), 6), round(float(y), 6), round(float(z), 6), r, g, b])
    return preview


def _artifact_stem(filename: str | None) -> str:
    stem = Path(filename or "depthlens").stem.strip() or "depthlens"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return safe or "depthlens"


def reconstruct_point_cloud(
    *,
    raw: bytes,
    filename: str | None,
    model: str,
    device: str,
    colormap: str = "inferno",
    max_dim: int | None = None,
    export_format: str = "ply",
    max_points: int = 120000,
    preview_points: int = 5000,
    focal_scale: float = 1.2,
    depth_scale: float = 1.0,
    depth_near_percentile: float = 2.0,
    depth_far_percentile: float = 98.0,
    sampling: str = "grid",
    include_rgb: bool = True,
    coordinate_system: str = "y_up",
) -> dict[str, Any]:
    """Infer depth, reconstruct a point cloud, and return a JSON-safe export payload."""

    started = time.perf_counter()
    fmt = normalize_reconstruction_format(export_format)
    options = clamp_reconstruction_options(
        max_points=max_points,
        preview_points=preview_points,
        focal_scale=focal_scale,
        depth_scale=depth_scale,
        depth_near_percentile=depth_near_percentile,
        depth_far_percentile=depth_far_percentile,
        sampling=sampling,
        include_rgb=include_rgb,
        coordinate_system=coordinate_system,
    )
    inferred = inference.infer_depth_arrays(
        raw=raw,
        model=model,
        device=device,
        filename=filename,
        max_dim=max_dim,
    )
    cloud = depth_to_point_cloud(
        inferred["img_bgr"],
        inferred["depth"],
        max_points=options["max_points"],
        focal_scale=options["focal_scale"],
        depth_scale=options["depth_scale"],
        depth_near_percentile=options["depth_near_percentile"],
        depth_far_percentile=options["depth_far_percentile"],
        sampling=options["sampling"],
        include_rgb=options["include_rgb"],
        coordinate_system=options["coordinate_system"],
    )
    points = cloud["points"]
    colors = cloud["colors"]
    artifact = serialize_ply(points, colors) if fmt == "ply" else serialize_obj(points, colors)
    extension = ".ply" if fmt == "ply" else ".obj"
    artifact_mime = "model/ply" if fmt == "ply" else "model/obj"
    preview = build_preview_points(points, colors, int(options["preview_points"]))
    depth_preview = inference._b64(inference._colorize(inferred["depth"], colormap))

    return {
        "status": "ok",
        "filename": filename,
        "artifact_filename": f"{_artifact_stem(filename)}_point_cloud{extension}",
        "artifact_format": fmt,
        "artifact_mime": artifact_mime,
        "artifact_base64": base64.b64encode(artifact).decode("ascii"),
        "artifact_size_bytes": len(artifact),
        "preview": {
            "points": preview,
            "point_count": len(preview),
            "truncated": int(points.shape[0]) > len(preview),
        },
        "depth_map": depth_preview,
        "reconstruction": cloud["metadata"],
        "resolution": inferred["resolution"],
        "model": inferred["model_id"],
        "model_id": inferred["model_id"],
        "model_display_name": inferred["model_display_name"],
        "device_used": inferred["device_used"],
        "engine_used": inferred["engine_used"],
        "fallback_used": bool(inferred.get("fallback_used", False)),
        "depth_cached": bool(inferred.get("depth_cached", False)),
        "latency_ms": inferred["latency_ms"],
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "warnings": list(inferred.get("warnings") or []),
    }
