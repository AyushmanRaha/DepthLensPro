from __future__ import annotations
from fastapi.testclient import TestClient
from backend.app import app
from backend.api import routes


def test_detect_route_structured_detector_unavailable(monkeypatch):
    class Boom(RuntimeError):
        error_code = 'DETECTOR_WEIGHTS_UNAVAILABLE'
    monkeypatch.setattr(routes, '_validated_device_or_422', lambda device: 'cpu')
    async def fail(**kwargs):
        raise Boom('weights missing')
    monkeypatch.setattr(routes, 'detect_objects_async', fail)
    res = TestClient(app).post('/api/detect', files={'file': ('a.jpg', b'xx', 'image/jpeg')})
    assert res.status_code == 503
    detail = res.json()['detail']
    assert detail['error_code'] == 'DETECTOR_WEIGHTS_UNAVAILABLE'
    assert 'weights missing' in detail['message']
    assert 'action' in detail
