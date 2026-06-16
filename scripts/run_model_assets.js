#!/usr/bin/env node
const { spawnSync } = require('child_process');
const path = require('path');
const root = path.resolve(__dirname, '..');
const env = { ...process.env, TORCH_HOME: path.join(root, 'models', 'torch-cache'), PYTHONUNBUFFERED: '1' };
const candidates = process.platform === 'win32'
  ? [path.join(root, '.venv', 'Scripts', 'python.exe'), path.join(root, 'venv', 'Scripts', 'python.exe'), 'py', 'python']
  : [path.join(root, '.venv', 'bin', 'python3'), path.join(root, '.venv', 'bin', 'python'), path.join(root, 'venv', 'bin', 'python3'), path.join(root, 'venv', 'bin', 'python'), 'python3', 'python'];
let last;
for (const py of candidates) {
  const args = py === 'py' ? ['-3', path.join(root, 'scripts', 'manage_model_assets.py'), ...process.argv.slice(2)] : [path.join(root, 'scripts', 'manage_model_assets.py'), ...process.argv.slice(2)];
  const proc = spawnSync(py, args, { cwd: root, env, stdio: 'inherit', shell: false });
  if (!proc.error || proc.error.code !== 'ENOENT') process.exit(proc.status ?? 1);
  last = proc.error;
}
console.error(`No Python executable found for model asset manager: ${last && last.message}`);
process.exit(2);
