"""Tests for hardware fallback and device-selection helpers."""

from __future__ import annotations

from typing import Any

import backend.utils.hardware as hardware


def test_default_device_falls_back_to_cpu(monkeypatch: Any) -> None:
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hardware, "_mps_available", lambda: False)
    monkeypatch.setattr(hardware, "_xpu_available", lambda: False)

    assert hardware._default_device_key() == "cpu"
    assert set(hardware._available_devices()) == {"cpu"}


def test_default_device_prefers_cuda(monkeypatch: Any) -> None:
    class Props:
        name = "Test CUDA"
        total_memory = 8 * 1024**3

    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hardware.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(hardware.torch.cuda, "get_device_properties", lambda index: Props())
    monkeypatch.setattr(hardware, "_mps_available", lambda: False)
    monkeypatch.setattr(hardware, "_xpu_available", lambda: False)

    assert hardware._default_device_key() == "cuda:0"
    assert hardware._available_devices()["cuda:0"]["hardware_name"] == "Test CUDA"


def test_resolve_rejects_unavailable_device(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        hardware,
        "_available_devices",
        lambda: {
            "cpu": {
                "name": "CPU · test",
                "hardware_name": "test",
                "type": "cpu",
                "compute_classes": ["cpu"],
                "available": True,
            }
        },
    )

    try:
        hardware._resolve("cuda:0")
    except ValueError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("expected ValueError for unavailable device")


def test_onnx_provider_translation_maps_mps_to_coreml() -> None:
    providers = hardware._onnx_providers_for_device(
        "mps", ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    )

    assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_onnx_provider_translation_maps_cuda_index_to_cuda() -> None:
    providers = hardware._onnx_providers_for_device(
        "cuda:0", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_onnx_provider_translation_maps_xpu_to_openvino() -> None:
    providers = hardware._onnx_providers_for_device(
        "xpu:0", ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    )

    assert providers == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
