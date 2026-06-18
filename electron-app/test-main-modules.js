const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { getAppRoot, getResourcePath } = require('./src/main/paths');
const { getPythonCandidates } = require('./src/main/python-resolver');
const { sanitizeSettings, readPersistedSettings, writePersistedSettings } = require('./src/main/settings-store');
const { missingResourceEntries, createStartupDetails, parsePidFromText, isLikelyInstalledAppPath, createBackendLifecycle } = require('./src/main/backend-lifecycle');

function fakeApp(userData, isPackaged = false) {
  return { isPackaged, getPath(name) { assert.strictEqual(name, 'userData'); return userData; }, getVersion() { return 'test'; } };
}

const devRoot = getAppRoot({ app: { isPackaged: false }, dirname: path.join(__dirname, 'nested') });
assert.strictEqual(devRoot, __dirname);
assert.strictEqual(getResourcePath({ app: { isPackaged: false }, dirname: path.join(__dirname, 'nested') }, 'frontend', 'index.html'), path.join(__dirname, 'frontend', 'index.html'));

const winCandidates = getPythonCandidates({ root: 'C:\\DepthLensPro', isDev: true, platform: 'win32', resourcesPath: 'C:\\DepthLensPro\\resources' });
assert(winCandidates.includes('py'));
assert(winCandidates.some((candidate) => candidate.endsWith(path.join('venv', 'Scripts', 'python.exe'))));
const posixCandidates = getPythonCandidates({ root: '/repo', isDev: false, platform: 'linux', resourcesPath: '/resources' });
assert(!posixCandidates.includes('python3'));
assert(posixCandidates.includes('/repo/venv/bin/python3'));

const sanitized = sanitizeSettings({ targetFps: 99, webcamFrameMaxDimension: 512, smoothingPreference: 2, selectedModel: 'X', unknown: true });
assert.strictEqual(sanitized.targetFps, 2);
assert.strictEqual(sanitized.webcamFrameMaxDimension, 512);
assert.strictEqual(sanitized.smoothingPreference, 0.95);
assert.strictEqual(sanitized.selectedModel, 'X');
assert.strictEqual(sanitized.unknown, undefined);

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'depthlens-settings-'));
const app = fakeApp(tmp);
const saved = writePersistedSettings(app, { targetFps: 5, selectedDevice: 'cpu' });
assert.strictEqual(saved.targetFps, 5);
assert.strictEqual(readPersistedSettings(app).selectedDevice, 'cpu');
fs.writeFileSync(path.join(tmp, 'settings.json'), '{bad json', 'utf8');
const recovered = readPersistedSettings(app, { warn() {} });
assert.strictEqual(recovered.recoveredFromCorruption, true);
assert(fs.readdirSync(tmp).some((name) => name.startsWith('settings.json.corrupt-')));

const detailRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'depthlens-details-'));
fs.mkdirSync(path.join(detailRoot, 'backend'), { recursive: true });
fs.writeFileSync(path.join(detailRoot, 'backend', 'app.py'), '');
fs.mkdirSync(path.join(detailRoot, 'models', 'onnx'), { recursive: true });
const details = createStartupDetails({ app: fakeApp(tmp, false), backendUrl: 'http://127.0.0.1:8765', pythonPath: 'python3', backendDir: path.join(detailRoot, 'backend'), cwd: detailRoot, command: 'python3 -m uvicorn', logPath: '/tmp/log' });
const missing = missingResourceEntries(details).map(([label]) => label);
assert(missing.includes('frontend/'));
assert(!missing.includes('models/'));
assert.strictEqual(parsePidFromText('users:(pid=1234,fd=7)'), 1234);
assert.strictEqual(parsePidFromText('PID: 5678'), 5678);
assert.strictEqual(isLikelyInstalledAppPath('/Applications/DepthLens Pro.app/Contents/MacOS/DepthLens Pro', 'darwin'), true);

const lifecycleState = { backendOwnedByElectron: true, backendPid: 999999, shutdownInProgress: true };
const lifecycle = createBackendLifecycle({
  app: fakeApp(tmp, false),
  log: { info() {}, warn() {}, error() {} },
  getAppRoot: () => detailRoot,
  getResourcePath: (...parts) => path.join(detailRoot, ...parts),
  getPythonPath: () => 'python3',
  pidStore: { readStoredBackendPid: () => null, readStoredBackendMetadata: () => null, writeBackendPidFiles() {}, removeBackendPidFiles() {} },
  BACKEND_HOST: '127.0.0.1',
  getBackendPort: () => 8765,
  setBackendPort() {},
  getBackendUrl: () => 'http://127.0.0.1:8765',
  setBackendUrl() {},
  getState: () => lifecycleState,
  setState: (patch) => Object.assign(lifecycleState, patch),
  logPath: () => '/tmp/log',
});
lifecycle.rememberBackendOutput('stdout', 'x'.repeat(12000));
lifecycle.rememberBackendOutput('stderr', 'last-line');
assert(lifecycle.backendOutputExcerpt().length <= 8000);
assert(lifecycle.backendOutputExcerpt().includes('last-line'));
lifecycle.shutdownOwnedBackend().then((ok) => assert.strictEqual(ok, true));

console.log('main process module tests passed');
