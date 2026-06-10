#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const { execFileSync } = require('child_process');
const { isDepthLensOwnedProcess } = require('../src/backend-process-policy');

const PORT = Number(process.env.DEPTHLENS_BACKEND_PORT || 8765);
const HOST = '127.0.0.1';
const APP_NAME = 'DepthLens Pro';
const ROOT = path.resolve(__dirname, '..', '..');
const BACKEND_DIR = path.join(ROOT, 'backend');

function userDataCandidates() {
  const home = os.homedir();
  if (process.platform === 'darwin') return [path.join(home, 'Library', 'Application Support', APP_NAME)];
  if (process.platform === 'win32') return [path.join(process.env.APPDATA || path.join(home, 'AppData', 'Roaming'), APP_NAME)];
  return [path.join(process.env.XDG_CONFIG_HOME || path.join(home, '.config'), APP_NAME)];
}

function readFileSafe(filePath) {
  try {
    return fs.readFileSync(filePath, 'utf8').trim();
  } catch (_) {
    return '';
  }
}

function readJsonSafe(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (_) {
    return null;
  }
}

function pidFiles() {
  return userDataCandidates().map((dir) => ({
    pidPath: path.join(dir, 'backend.pid'),
    metaPath: path.join(dir, 'backend.json'),
  }));
}

function execCap(cmd, args, timeout = 3000) {
  try {
    return execFileSync(cmd, args, { encoding: 'utf8', timeout, windowsHide: true }).trim();
  } catch (e) {
    return (e.stdout || '').toString().trim();
  }
}

function listeningPid() {
  if (process.platform === 'win32') {
    const out = execCap('cmd.exe', ['/c', `netstat -ano -p tcp | findstr :${PORT}`]);
    const line = out.split(/\r?\n/).find((row) => row.includes('LISTENING'));
    return line ? Number(line.trim().split(/\s+/).pop()) : null;
  }
  const out = execCap('lsof', ['-nP', `-iTCP:${PORT}`, '-sTCP:LISTEN']);
  const line = out.split(/\r?\n/).find((row) => /\bLISTEN\b/.test(row) && !row.startsWith('COMMAND'));
  return line ? Number(line.trim().split(/\s+/)[1]) : null;
}

function cmdline(pid) {
  if (!pid) return '';
  if (process.platform === 'win32') {
    return execCap('powershell.exe', [
      '-NoProfile',
      '-Command',
      `(Get-CimInstance Win32_Process -Filter "ProcessId=${pid}").CommandLine`,
    ]);
  }
  return execCap('ps', ['-p', String(pid), '-o', 'command=']);
}

function storedBackendState() {
  for (const files of pidFiles()) {
    const pid = Number(readFileSafe(files.pidPath));
    const metadata = readJsonSafe(files.metaPath);
    if (Number.isFinite(pid) && pid > 0) return { pid, metadata, files };
  }
  return { pid: null, metadata: null, files: null };
}

function safe(pid, commandLine, stored) {
  const metadataMatchesPid = Boolean(
    stored?.metadata?.pid
      && Number(pid) === Number(stored.metadata.pid)
      && stored.metadata.host === HOST
      && Number(stored.metadata.port) === Number(PORT),
  );
  if (!metadataMatchesPid) return false;
  return isDepthLensOwnedProcess({
    commandLine,
    storedMetadata: stored.metadata,
    cwd: ROOT,
    backendDir: BACKEND_DIR,
  });
}

function kill(pid, force = false) {
  if (process.platform === 'win32') {
    execCap('taskkill.exe', ['/PID', String(pid), '/T', ...(force ? ['/F'] : [])]);
  } else {
    try {
      process.kill(pid, force ? 'SIGKILL' : 'SIGTERM');
    } catch (e) {
      console.warn(e.message);
    }
  }
}

function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }

function request(endpoint, timeout) {
  return new Promise((resolve) => {
    let connected = false;
    let body = '';
    const req = http.get({ host: HOST, port: PORT, path: endpoint, timeout }, (res) => {
      connected = true;
      res.setEncoding('utf8');
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => resolve({ endpoint, status: res.statusCode, body, empty: !body, connected }));
    });
    req.on('socket', (socket) => socket.on('connect', () => { connected = true; }));
    req.on('timeout', () => {
      req.destroy();
      resolve({ endpoint, error: connected ? 'TCP connected but no HTTP response before timeout' : 'timeout', connected });
    });
    req.on('error', (e) => resolve({ endpoint, error: e.code === 'ECONNREFUSED' ? 'connection refused' : e.message, connected }));
  });
}

async function killBackend() {
  const stored = storedBackendState();
  const pid = listeningPid() || stored.pid;
  console.log(`Detected PID: ${pid || 'none'}`);
  if (pid) {
    const c = cmdline(pid);
    console.log(`Command line: ${c || 'unknown'}`);
    if (!safe(pid, c, stored)) {
      console.log('Refusing to kill: process is not a safe DepthLens backend match.');
      return 2;
    }
    kill(pid, false);
    await sleep(3000);
    if (listeningPid() === pid) {
      console.log(`Still listening; force killing safe DepthLens backend PID ${pid}`);
      kill(pid, true);
      await sleep(1000);
    }
  }
  const final = listeningPid();
  console.log(`Final port ${PORT} state: ${final ? `LISTENING by PID ${final}` : 'free'}`);
  return final ? 1 : 0;
}

async function smoke() {
  let code = 0;
  for (const [ep, to] of [['/live', 3000], ['/ready', 7000], ['/devices', 5000], ['/health', 8000]]) {
    const r = await request(ep, to);
    console.log(`${ep}: ${JSON.stringify(r)}`);
    if (r.error || !r.status || r.status >= 400 || r.empty) {
      code = 1;
      if (r.connected && r.error) console.error(`${ep} failure: TCP connects but receives 0 bytes / no HTTP response.`);
    }
  }
  const pid = listeningPid();
  console.log(`lsof/netstat PID: ${pid || 'none'}`);
  for (const f of pidFiles()) {
    console.log(`${f.pidPath}: ${readFileSafe(f.pidPath) || '(missing)'}`);
    console.log(`${f.metaPath}: ${readFileSafe(f.metaPath) || '(missing)'}`);
  }
  return code;
}

async function main() {
  const mode = process.argv[2];
  const code = mode === 'kill' ? await killBackend() : mode === 'smoke' ? await smoke() : 64;
  process.exit(code);
}

if (require.main === module) {
  main();
}

module.exports = { safe, storedBackendState, userDataCandidates };
