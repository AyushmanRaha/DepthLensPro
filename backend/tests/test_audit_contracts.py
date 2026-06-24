from backend.api.errors import error_envelope
from backend.main import _cors_origins
from backend.services.observability import sanitize_message


def test_sanitize_message_redacts_paths_files_and_tokens():
    raw = (
        "/home/alice/photo.png C:\\Users\\alice\\secret.jpg "
        "cache:abc1234567890abcdef "
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/=="
    )
    out = sanitize_message(raw)
    assert "alice" not in out
    assert "photo.png" not in out
    assert "secret.jpg" not in out
    assert "[file]" in out or "[path]" in out


def test_error_envelope_shape():
    payload = error_envelope(
        "INVALID_CONTENT_TYPE", "Expected an image file", field="file", retryable=False
    )
    assert payload["error_code"] == "INVALID_CONTENT_TYPE"
    assert payload["message"] == "Expected an image file"
    assert payload["field"] == "file"
    assert "retryable" not in payload


def test_default_cors_is_local_not_wildcard():
    origins = _cors_origins()
    assert "*" not in origins
    assert "null" in origins
    assert any(origin.startswith("http://localhost") for origin in origins)
