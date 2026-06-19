const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { computeDmgSizeBytes, bytesToBuilderSize, directorySize, preflight, shouldIgnore } = require('./scripts/package-size-preflight');

const MiB = 1024 ** 2;
const GiB = 1024 ** 3;
function tmp() { return fs.mkdtempSync(path.join(os.tmpdir(), 'depthlens-size-')); }
function file(p, bytes = 1) { fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, Buffer.alloc(bytes)); }
function minimalRepo(root) {
  file(path.join(root, 'electron-app', 'main.js'));
  file(path.join(root, 'backend', 'app.py'));
  file(path.join(root, 'frontend', 'index.html'));
  file(path.join(root, 'venv', 'bin', 'python3'));
  file(path.join(root, 'models', 'torch-cache', 'hub', 'intel-isl_MiDaS_master', 'hubconf.py'));
  fs.mkdirSync(path.join(root, 'models', 'onnx'), { recursive: true });
}

{
  const dmg = computeDmgSizeBytes(3 * GiB, { margin: 0.5, minimumBytes: 1 });
  assert(dmg >= Math.ceil((3 * GiB * 1.5 + 256 * MiB) / MiB) * MiB, 'DMG includes safety margin and filesystem overhead');
  assert.strictEqual(bytesToBuilderSize(1536 * MiB), '1536m');
}

{
  const root = tmp();
  file(path.join(root, 'keep.bin'), 10);
  file(path.join(root, '__pycache__', 'junk.pyc'), 1000);
  file(path.join(root, '.pytest_cache', 'node'), 1000);
  file(path.join(root, 'debug.log'), 1000);
  const result = directorySize(root);
  assert.strictEqual(result.bytes, 10, 'junk files are ignored by size accounting');
  assert(shouldIgnore('debug.log', 'debug.log'));
}

{
  const root = tmp();
  minimalRepo(root);
  const result = preflight({ repoRoot: root, platform: 'darwin', arch: 'arm64' });
  assert.strictEqual(result.ok, true, 'minimal complete repo passes preflight');
  assert(result.dmgBytes > result.payloadBytes, 'macOS preflight plans a DMG larger than payload before dmgbuild starts');
}

{
  const root = tmp();
  file(path.join(root, 'electron-app', 'main.js'));
  const result = preflight({ repoRoot: root, platform: 'linux', arch: 'x64' });
  assert.strictEqual(result.ok, false, 'missing required resources fail clearly');
  assert(result.missingRequired.some((p) => p.endsWith(path.join('backend'))));
}

{
  const root = tmp();
  minimalRepo(root);
  const result = preflight({ repoRoot: root, platform: 'darwin', arch: 'arm64', onnx: 'require-all', models: 'all' });
  assert.strictEqual(result.ok, false, 'missing required ONNX model files fail preflight clearly');
  assert(result.missingRequired.some((p) => p.endsWith(path.join('models', 'onnx', 'dpt_large.onnx'))));
}

console.log('package size preflight tests passed');
