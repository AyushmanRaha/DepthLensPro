from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_compare_engine_selector_contract():
    assert 'id="compareEngine"' in read("frontend/index.html")
    assert 'compareEngine:         $("#compareEngine")' in read("frontend/js/dom.js")
    compare = read("frontend/js/compare.js")
    assert 'el.compareEngine?.value || selEngine?.() || "auto"' in compare
    assert 'getInteractiveMaxDim(),"auto")' not in compare


def test_live_hysteresis_contract():
    settings = read("frontend/js/settings.js")
    api = read("frontend/js/api-client.js")
    assert "lastLiveOkAt" in settings
    assert "consecutiveLiveFailures" in settings
    assert "lastLiveError" in settings
    assert "Depth engine delayed" in api
    assert "inferenceReady = false" in api
    assert "consecutiveLiveFailures < 2" in api


def test_reconstruction_detection_timeout_contract():
    js = read("frontend/js/reconstruction.js")
    assert "60000" in js
    assert "20000" in js
    assert "cap.detecting" in js
    assert "maxDim: 512" in js
