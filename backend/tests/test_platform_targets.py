from __future__ import annotations
import json
from pathlib import Path


def test_root_scripts_include_cross_platform_x64_and_onnx() -> None:
    scripts = json.loads(Path('package.json').read_text())['scripts']
    for key in ['setup:mac','setup:win','setup:linux','setup:mac:onnx','setup:win:onnx','setup:linux:onnx','build:win:x64','build:linux:x64','launch:mac','launch:win','launch:linux']:
        assert key in scripts


def test_electron_build_targets_preserve_mac_arm_and_add_win_linux_x64() -> None:
    data = json.loads(Path('electron-app/package.json').read_text())
    assert data['build']['mac']['target'][0]['arch'] == ['arm64']
    assert set(data['build']['win']['target'][0]['arch']) == {'arm64','x64'}
    assert set(data['build']['linux']['target'][0]['arch']) == {'arm64','x64'}
    assert 'unsupported-arch.js --platform darwin --arch x64' in data['scripts']['build:mac:x64']
    assert 'build:win:x64:raw' in data['scripts']
    assert 'build:linux:x64:raw' in data['scripts']
