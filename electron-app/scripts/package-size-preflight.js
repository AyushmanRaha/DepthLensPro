#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;
const DEFAULT_MARGIN = 0.35;
const DEFAULT_MIN_DMG_BYTES = 2 * GIB;
const DEFAULT_MIN_FREE_BYTES = 2 * GIB;

const IGNORE_SEGMENTS = new Set(['.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.cache','node_modules','.depthlens']);
const IGNORE_SUFFIXES = ['.pyc','.pyo','.log','.tmp','.temp','.map'];
const IGNORE_NAMES = new Set(['.DS_Store','Thumbs.db','setup-report.json']);

function parseArgs(argv) {
  const args = { platform: process.platform, arch: process.arch === 'x64' ? 'x64' : process.arch, repoRoot: path.resolve(__dirname, '..', '..'), onnx: 'optional', models: 'midas_small', mode: 'native', format: 'text' };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--platform') args.platform = argv[++i];
    else if (arg === '--arch') args.arch = argv[++i];
    else if (arg === '--repo-root') args.repoRoot = path.resolve(argv[++i]);
    else if (arg === '--onnx') args.onnx = argv[++i];
    else if (arg === '--models') args.models = argv[++i];
    else if (arg === '--json') args.format = 'json';
    else if (arg === '--dmg-size-only') args.format = 'dmg-size-only';
    else throw new Error(`Unknown option: ${arg}`);
  }
  return args;
}

function shouldIgnore(name, rel) {
  if (!name) return false;
  if (IGNORE_NAMES.has(name)) return true;
  if (IGNORE_SUFFIXES.some((suffix) => name.endsWith(suffix))) return true;
  return rel.split(path.sep).some((part) => IGNORE_SEGMENTS.has(part));
}

function directorySize(root, options = {}) {
  const missingRequired = [];
  const entries = [];
  const required = options.required !== false;
  if (!fs.existsSync(root)) {
    if (required) missingRequired.push(root);
    return { bytes: 0, files: 0, missingRequired, entries };
  }
  let bytes = 0;
  let files = 0;
  function walk(current, rel = '') {
    const name = path.basename(current);
    if (rel && shouldIgnore(name, rel)) return;
    const st = fs.lstatSync(current);
    if (st.isSymbolicLink()) return;
    if (st.isDirectory()) {
      for (const child of fs.readdirSync(current)) walk(path.join(current, child), path.join(rel, child));
    } else if (st.isFile()) {
      bytes += st.size;
      files += 1;
    }
  }
  walk(root);
  entries.push({ path: root, bytes, files });
  return { bytes, files, missingRequired, entries };
}

function add(a, b) {
  return { bytes: a.bytes + b.bytes, files: a.files + b.files, missingRequired: [...a.missingRequired, ...b.missingRequired], entries: [...a.entries, ...b.entries] };
}

function computeDmgSizeBytes(payloadBytes, { margin = DEFAULT_MARGIN, minimumBytes = DEFAULT_MIN_DMG_BYTES } = {}) {
  const padded = Math.ceil(payloadBytes * (1 + margin));
  const withFsOverhead = padded + 256 * MIB;
  return Math.ceil(Math.max(withFsOverhead, minimumBytes) / MIB) * MIB;
}

function bytesToBuilderSize(bytes) {
  return `${Math.ceil(bytes / MIB)}m`;
}

function fmt(bytes) {
  return `${(bytes / GIB).toFixed(2)} GiB`;
}

function resourceFootprint(repoRoot) {
  const appRoot = path.join(repoRoot, 'electron-app');
  const parts = [
    ['Electron app main/preload', path.join(appRoot, 'main.js')],
    ['Electron app preload', path.join(appRoot, 'preload.js')],
    ['Electron app src', path.join(appRoot, 'src')],
    ['Electron app assets', path.join(appRoot, 'assets')],
    ['Electron production dependency electron-log', path.join(appRoot, 'node_modules', 'electron-log')],
    ['backend', path.join(repoRoot, 'backend')],
    ['frontend', path.join(repoRoot, 'frontend')],
    ['Python venv', path.join(repoRoot, 'venv')],
    ['models/onnx', path.join(repoRoot, 'models', 'onnx')],
    ['models/torch-cache', path.join(repoRoot, 'models', 'torch-cache')],
  ];
  let total = { bytes: 0, files: 0, missingRequired: [], entries: [] };
  const breakdown = [];
  for (const [label, dir] of parts) {
    const required = ['backend', 'frontend', 'Python venv', 'models/torch-cache'].includes(label);
    const result = directorySize(dir, { required });
    breakdown.push({ label, path: dir, bytes: result.bytes, files: result.files });
    total = add(total, result);
  }
  const electron = directorySize(path.join(appRoot, 'node_modules', 'electron'), { required: false });
  breakdown.push({ label: 'Electron runtime cache', path: path.join(appRoot, 'node_modules', 'electron'), bytes: electron.bytes, files: electron.files });
  total = add(total, electron);
  return { ...total, breakdown };
}

function statFreeBytes(dir) {
  let current = dir;
  while (!fs.existsSync(current)) current = path.dirname(current);
  if (fs.statfsSync) return fs.statfsSync(current).bavail * fs.statfsSync(current).bsize;
  return Number.MAX_SAFE_INTEGER;
}

function preflight(options) {
  const repoRoot = path.resolve(options.repoRoot);
  const footprint = resourceFootprint(repoRoot);
  const requestedModels = String(options.models || 'midas_small') === 'all' ? ['midas_small', 'dpt_hybrid', 'dpt_large'] : String(options.models || 'midas_small').split(',').map((m) => m.trim()).filter(Boolean);
  if (['required', 'require-all'].includes(options.onnx)) {
    for (const model of requestedModels) {
      const file = path.join(repoRoot, 'models', 'onnx', `${model}.onnx`);
      if (!fs.existsSync(file) || fs.statSync(file).size <= 0) footprint.missingRequired.push(file);
    }
  }
  const payloadBytes = footprint.bytes;
  const dmgBytes = computeDmgSizeBytes(payloadBytes, {
    margin: Number(process.env.DEPTHLENS_PACKAGE_SIZE_MARGIN || DEFAULT_MARGIN),
    minimumBytes: Number(process.env.DEPTHLENS_DMG_MIN_BYTES || DEFAULT_MIN_DMG_BYTES),
  });
  const locations = [os.tmpdir(), path.join(repoRoot, 'electron-app', 'dist'), path.join(repoRoot, 'electron-app', 'build')];
  const minFreeBytes = Math.max(Number(process.env.DEPTHLENS_PACKAGE_MIN_FREE_BYTES || DEFAULT_MIN_FREE_BYTES), options.platform === 'darwin' ? dmgBytes + payloadBytes : payloadBytes * 2);
  const freeChecks = locations.map((dir) => ({ path: dir, freeBytes: statFreeBytes(dir), requiredBytes: minFreeBytes, ok: statFreeBytes(dir) >= minFreeBytes }));
  const ok = footprint.missingRequired.length === 0 && freeChecks.every((c) => c.ok) && (options.platform !== 'darwin' || dmgBytes > payloadBytes);
  return { schemaVersion: 1, platform: options.platform, arch: options.arch, repoRoot, payloadBytes, payloadHuman: fmt(payloadBytes), dmgBytes, dmgSize: bytesToBuilderSize(dmgBytes), dmgHuman: fmt(dmgBytes), margin: Number(process.env.DEPTHLENS_PACKAGE_SIZE_MARGIN || DEFAULT_MARGIN), breakdown: footprint.breakdown, missingRequired: footprint.missingRequired, freeChecks, ok };
}

function printText(result) {
  console.log('[DepthLens] Packaging size preflight');
  for (const item of result.breakdown) console.log(`  - ${item.label}: ${fmt(item.bytes)} (${item.files} files)`);
  console.log(`  Estimated packaged payload: ${result.payloadHuman}`);
  if (result.platform === 'darwin') console.log(`  Computed DMG size: ${result.dmgHuman} (${result.dmgSize}) with ${(result.margin * 100).toFixed(0)}% margin plus filesystem overhead`);
  for (const check of result.freeChecks) console.log(`  Host free space ${check.path}: ${fmt(check.freeBytes)} available, ${fmt(check.requiredBytes)} required`);
  if (result.missingRequired.length) console.error(`Missing required packaging resources:\n${result.missingRequired.join('\n')}`);
  const failedFree = result.freeChecks.filter((c) => !c.ok);
  if (failedFree.length) console.error('Insufficient host temp/build/output disk space for packaging. Free space checks failed before electron-builder started.');
  if (result.platform === 'darwin' && result.dmgBytes <= result.payloadBytes) console.error('The DMG image is too small for the packaged app; increase computed dmg.size or reduce bundled resources.');
}

if (require.main === module) {
  try {
    const options = parseArgs(process.argv.slice(2));
    const result = preflight(options);
    if (options.format === 'json') console.log(JSON.stringify(result, null, 2));
    else if (options.format === 'dmg-size-only') console.log(result.dmgSize);
    else printText(result);
    process.exit(result.ok ? 0 : 1);
  } catch (error) {
    console.error(error.message);
    process.exit(2);
  }
}

module.exports = { directorySize, computeDmgSizeBytes, bytesToBuilderSize, resourceFootprint, preflight, shouldIgnore, fmt };
