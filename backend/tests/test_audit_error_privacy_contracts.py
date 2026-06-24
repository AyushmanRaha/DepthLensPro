import asyncio
import json
import logging
import sys

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.api import routes
from backend.main import JsonLogFormatter, app

client = TestClient(app)


def assert_detail_envelope(response, status=422):
    assert response.status_code == status
    detail = response.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["error_code"]
    assert detail["message"]
    return detail


def test_route_level_validation_errors_are_structured():
    assert_detail_envelope(client.get("/benchmark?model=bad-model&iterations=0"))
    files = {"file": ("image.png", b"not-an-image", "image/png")}
    assert_detail_envelope(client.post("/estimate", files=files, data={"outputs": "bad-output"}))
    assert_detail_envelope(client.post("/compare", files=files, data={"metrics": "bad-mode"}))
    assert_detail_envelope(client.post("/detect", files=files, data={"threshold": "2"}))
    assert_detail_envelope(client.post("/reconstruct", files=files, data={"max_points": "0"}))
    assert_detail_envelope(
        client.post("/batch", files=[("files", (f"{i}.png", b"x", "image/png")) for i in range(11)])
    )


def test_compare_per_model_error_is_structured_and_legacy_compatible():
    err = routes._safe_compare_error(
        "MiDaS_small", RuntimeError("/home/alice/private/cat.png exploded")
    )
    assert err["error"] == err["message"]
    assert err["error_detail"]["error_code"]
    assert "cat.png" not in json.dumps(err)


@pytest.mark.asyncio
async def test_batch_item_timeout_is_structured():
    async def slow():
        raise asyncio.TimeoutError("/Users/bob/Desktop/secret.png")

    with pytest.raises(HTTPException) as raised:
        await routes._with_batch_item_timeout(slow())
    detail = raised.value.detail
    assert detail["error_code"] == "REQUEST_TIMEOUT"
    assert "secret.png" not in json.dumps(detail)


@pytest.mark.asyncio
async def test_route_timeout_codes_are_structured():
    async def slow():
        raise asyncio.TimeoutError("C:\\Users\\bob\\Desktop\\secret.png")

    with pytest.raises(HTTPException) as raised:
        await routes._with_route_timeout(slow(), "/estimate")
    assert raised.value.detail["error_code"] == "REQUEST_TIMEOUT"
    with pytest.raises(HTTPException) as reconstruct_raised:
        await routes._with_route_timeout(
            slow(),
            "/reconstruct",
            timeout_factory=routes.reconstruction_timeout_error,
            timeout_code="RECONSTRUCTION_TIMEOUT",
        )
    assert reconstruct_raised.value.detail["error_code"] == "RECONSTRUCTION_TIMEOUT"


def test_json_log_formatter_sanitizes_sensitive_fields():
    formatter = JsonLogFormatter()
    try:
        raise RuntimeError("failed /home/alice/Pictures/cat.png token=abcdef1234567890abcdef")
    except RuntimeError:
        exc_info = sys.exc_info()
    record = logging.LogRecord(
        "depthlens",
        logging.ERROR,
        "/Users/bob/Desktop/photo.jpg",
        12,
        "message C:\\Users\\bob\\Desktop\\secret.png cache_key=abcdef1234567890abcdef long %s",
        ("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVpBQkNERUZHSElKS0w=",),
        exc_info,
    )
    record.stack_info = "Stack at /home/alice/project/file.py with /Users/bob/Desktop/photo.jpg"
    record.cache_key = "abcdef1234567890abcdef1234567890"
    payload = json.loads(formatter.format(record))
    text = json.dumps(payload)
    for raw in [
        "/home/alice",
        "/Users/bob",
        "C:\\\\Users",
        "cat.png",
        "photo.jpg",
        "secret.png",
        "abcdef1234567890abcdef",
        "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVpBQkNERUZHSElKS0w=",
    ]:
        assert raw not in text
    assert payload["exception"]["type"] == "RuntimeError"
