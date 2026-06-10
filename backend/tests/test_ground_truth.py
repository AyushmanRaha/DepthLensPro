from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
from PIL import Image

from backend.services import ground_truth as gt


def test_decode_png_ground_truth_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    arr = np.array([[1, 2], [0, 4]], dtype=np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    payload = gt.decode_ground_truth(buf.getvalue(), "depth.png")
    assert payload.depth.shape == (2, 2)
    assert payload.metadata["valid_pixel_count"] == 3
    assert payload.metadata["invalid_pixel_count"] == 1

    monkeypatch.setattr(gt, "_colorize", lambda values, colormap=0: "png-b64")
    pred = np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float32)
    result = gt.compute_ground_truth_metrics(pred, payload.depth, metadata=payload.metadata)
    metrics = result["metrics"]
    assert set(
        ["abs_rel", "sq_rel", "gt_mae", "gt_rmse", "gt_log_rmse", "delta_1", "delta_2", "delta_3"]
    ).issubset(metrics)
    assert result["metadata"]["scale_alignment"] == "median_scale"
    assert result["metadata"]["valid_pixel_count"] == 3


def test_decode_npy_rejects_multichannel() -> None:
    buf = io.BytesIO()
    np.save(buf, np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(gt.GroundTruthError, match="must be H×W"):
        gt.decode_ground_truth(buf.getvalue(), "depth.npy")


def test_shape_mismatch_resizes_gt_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gt, "_colorize", lambda values, colormap=0: "png-b64")

    def fake_resize(
        values: np.ndarray, size: tuple[int, int], interpolation: Any = None
    ) -> np.ndarray:
        return np.ones((size[1], size[0]), dtype=np.float32)

    monkeypatch.setattr(gt.cv2, "resize", fake_resize)
    pred = np.ones((4, 4), dtype=np.float32)
    labels = np.ones((2, 2), dtype=np.float32)
    result = gt.compute_ground_truth_metrics(pred, labels, metadata={"warnings": []})
    assert result["metadata"]["aligned_shape"] == [4, 4]
    assert "Resized GT" in result["metadata"]["warnings"][0]


def test_gt_scale_converts_millimeters_to_meters() -> None:
    buf = io.BytesIO()
    np.save(buf, np.array([[1000.0, 2000.0], [0.0, 4000.0]], dtype=np.float32))

    payload = gt.decode_ground_truth(buf.getvalue(), "depth.npy", scale=0.001)

    assert np.isclose(payload.depth[0, 0], 1.0)
    assert payload.metadata["scale_factor_input"] == 0.001
    assert "millimeters were converted to meters" in " ".join(payload.metadata["warnings"])


def test_median_alignment_rejects_near_zero_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gt, "_colorize", lambda values, colormap=0: "png-b64")
    pred = np.full((2, 2), 1e-7, dtype=np.float32)
    labels = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(gt.GroundTruthError, match="No overlapping|too small|near zero"):
        gt.compute_ground_truth_metrics(pred, labels, metadata={"warnings": []})


def test_resize_metadata_reports_method_and_invalid_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gt, "_colorize", lambda values, colormap=0: "png-b64")

    def fake_resize(
        values: np.ndarray, size: tuple[int, int], interpolation: Any = None
    ) -> np.ndarray:
        del interpolation
        return np.resize(values, (size[1], size[0])).astype(np.float32)

    monkeypatch.setattr(gt.cv2, "resize", fake_resize)
    pred = np.ones((4, 4), dtype=np.float32)
    labels = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)

    result = gt.compute_ground_truth_metrics(
        pred,
        labels,
        metadata={"warnings": [], "invalid_pixel_count": 1},
    )

    meta = result["metadata"]
    assert meta["resize_method"] == "nearest"
    assert meta["invalid_pixel_count_before_resize"] == 1
    assert "invalid_pixel_count_after_resize" in meta
