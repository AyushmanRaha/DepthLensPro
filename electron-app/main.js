const { app, BrowserWindow, ipcMain, shell } = require("electron");
const path = require("path");
const { spawn, execFileSync } = require("child_process");
const http = require("http");
const net = require("net");
const fs = require("fs");
const log = require("electron-log");
const { isAllowedAppUrl, isExternalUrl } = require("./src/security-policy");
const {
  isDepthLensOwnedProcess: isDepthLensOwnedProcessPolicy,
} = require("./src/backend-process-policy");

log.transports.file.level = "info";
log.info("DepthLens Pro starting...");

const singleInstanceLock = app.requestSingleInstanceLock();
if (!singleInstanceLock) {
  log.warn("SECOND_INSTANCE_DETECTED");
  app.quit();
} else {
  app.on("second-instance", () => {
    log.warn("SECOND_INSTANCE_DETECTED");
    if (mainWindow && !mainWindow.isDestroyed()) {
      log.info("SECOND_INSTANCE_FOCUS_EXISTING_WINDOW");
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

// ── Constants ──────────────────────────────────────────────
const BACKEND_HOST = "127.0.0.1";
const REQUESTED_BACKEND_PORT = Number.parseInt(process.env.DEPTHLENS_BACKEND_PORT || "8765", 10);
let BACKEND_PORT = REQUESTED_BACKEND_PORT;
let backendUrl = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
const isDev = !app.isPackaged || process.env.NODE_ENV === "development";

function getAppRoot() {
  return isDev ? path.resolve(__dirname, "..") : process.resourcesPath;
}

function getResourcePath(...parts) {
  return path.join(getAppRoot(), ...parts);
}

// ── State ──────────────────────────────────────────────────
let mainWindow = null;
let splashWindow = null;
let backendProcess = null;
let backendReady = false;
let backendOwnedByElectron = false;
let backendPid = null;
let backendMetadata = null;
let shutdownInProgress = false;
const backendOutputTail = [];

function rememberBackendOutput(stream, message) {
  const line = `[${stream}] ${message}`;
  backendOutputTail.push(line);
  while (backendOutputTail.length > 80) backendOutputTail.shift();
}

function backendOutputExcerpt() {
  return backendOutputTail.slice(-30).join("\n") || "n/a";
}

// ── Resource paths ─────────────────────────────────────────
function logPath() {
  return log.transports.file.getFile().path;
}

// ── HTTP helpers ───────────────────────────────────────────
function requestJson(url, timeoutMs = 2000) {
  return new Promise((resolve, reject) => {
    let connected = false;
    const req = http.get(url, { timeout: timeoutMs }, (res) => {
      connected = true;
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { body += chunk; });
      res.on("end", () => {
        let json = null;
        try { json = body ? JSON.parse(body) : null; } catch (_) {}
        resolve({ statusCode: res.statusCode, json, body, connected, empty: body.length === 0 });
      });
    });
    req.on("socket", (socket) => {
      socket.on("connect", () => { connected = true; });
    });
    req.on("error", (err) => {
      err.connected = connected;
      if (err.code === "ECONNREFUSED") err.kind = "connection_refused";
      else if (connected && /timed out|timeout/i.test(err.message)) err.kind = "tcp_connected_http_timeout";
      reject(err);
    });
    req.on("timeout", () => {
      const err = new Error(`Timed out requesting ${url}`);
      err.connected = connected;
      err.kind = connected ? "tcp_connected_http_timeout" : "timeout";
      req.destroy(err);
    });
  });
}

async function probeLive(url, timeoutMs = 1200, attempt = 0) {
  const liveUrl = `${url}/live`;
  try {
    const live = await requestJson(liveUrl, timeoutMs);
    const valid = Boolean(live.statusCode === 200 && live.json?.service === "DepthLens Pro API" && live.json?.status === "ok");
    log.info("LIVE_POLL_ATTEMPT", { attempt, url: liveUrl, statusCode: live.statusCode, json: live.json, valid, empty: live.empty });
    return { ok: valid, kind: valid ? "valid_depthlens_live" : "non_depthlens_response", ...live };
  } catch (err) {
    const kind = err.kind || (err.code === "ECONNREFUSED" ? "connection_refused" : "error");
    if (kind === "tcp_connected_http_timeout") log.warn("LIVE_TCP_CONNECTED_HTTP_TIMEOUT", { attempt, url: liveUrl, error: err.message });
    log.info("LIVE_POLL_ATTEMPT", { attempt, url: liveUrl, error: err.message, kind });
    return { ok: false, kind, error: err };
  }
}

async function isLiveDepthLensBackend(url) {
  return (await probeLive(url, 1200, 0)).ok;
}

function isPortAvailable(port, host = BACKEND_HOST) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", (err) => {
      if (err.code === "EADDRINUSE" || err.code === "EACCES") resolve(false);
      else reject(err);
    });
    server.once("listening", () => {
      server.close(() => resolve(true));
    });
    server.listen(port, host);
  });
}

async function findAvailableBackendPort(startPort = REQUESTED_BACKEND_PORT, host = BACKEND_HOST) {
  const envPinned = Boolean(process.env.DEPTHLENS_BACKEND_PORT);
  const maxAttempts = envPinned ? 1 : 25;
  for (let offset = 0; offset < maxAttempts; offset += 1) {
    const candidate = startPort + offset;
    if (await isPortAvailable(candidate, host)) return candidate;
  }
  return null;
}

function getBackendPidPath() {
  return path.join(app.getPath("userData"), "backend.pid");
}

function getBackendMetadataPath() {
  return path.join(app.getPath("userData"), "backend.json");
}

function readStoredBackendPid() {
  try {
    const raw = fs.readFileSync(getBackendPidPath(), "utf8").trim();
    const pid = Number.parseInt(raw, 10);
    return Number.isFinite(pid) && pid > 0 ? pid : null;
  } catch (_) {
    return null;
  }
}

function readStoredBackendMetadata() {
  try { return JSON.parse(fs.readFileSync(getBackendMetadataPath(), "utf8")); } catch (_) { return null; }
}

function removeBackendPidFiles() {
  for (const filePath of [getBackendPidPath(), getBackendMetadataPath()]) {
    try { if (fs.existsSync(filePath)) fs.unlinkSync(filePath); } catch (err) { log.warn(`Failed to remove ${filePath}: ${err.message}`); }
  }
}

function writePrivateFile(filePath, contents) {
  const handle = fs.openSync(filePath, "w", 0o600);
  try {
    fs.writeFileSync(handle, contents, "utf8");
  } finally {
    fs.closeSync(handle);
  }
  try { fs.chmodSync(filePath, 0o600); } catch (_) {}
}

function writeBackendPidFiles(metadata) {
  fs.mkdirSync(app.getPath("userData"), { recursive: true, mode: 0o700 });
  try { fs.chmodSync(app.getPath("userData"), 0o700); } catch (_) {}
  const lifecycleMetadata = {
    pid: metadata.pid,
    backendUrl: metadata.backendUrl,
    port: metadata.port,
    host: metadata.host,
    startedAt: metadata.startedAt,
    appVersion: metadata.appVersion,
    isPackaged: metadata.isPackaged,
    platform: metadata.platform,
    arch: metadata.arch,
  };
  writePrivateFile(getBackendPidPath(), `${metadata.pid}\n`);
  writePrivateFile(getBackendMetadataPath(), `${JSON.stringify(lifecycleMetadata, null, 2)}\n`);
  log.info("BACKEND_PID_WRITTEN", { pidFile: getBackendPidPath(), metadataFile: getBackendMetadataPath(), pid: metadata.pid });
}

function delay(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }

const { evaluateTarget, isSupportedTarget } = require("./src/platform-support");

function isSupportedArchitecture() {
  return isSupportedTarget(process.platform, process.arch);
}

function showUnsupportedArchitectureDialog() {
  const { dialog } = require("electron");
  const target = evaluateTarget(process.platform, process.arch);
  const message = target.reason || "Unsupported DepthLens Pro native target.";
  log.error("UNSUPPORTED_ARCH_BLOCKED", { platform: process.platform, arch: process.arch, reason: message });
  dialog.showMessageBoxSync({
    type: "error",
    title: "DepthLens Pro — Unsupported Platform",
    message,
    detail: `${message}\n\nDetected platform: ${process.platform}\nDetected architecture: ${process.arch}`,
    buttons: ["Quit"],
  });
}

function execCapture(command, args, options = {}) {
  try {
    return execFileSync(command, args, { encoding: "utf8", timeout: options.timeout || 2500, windowsHide: true }).trim();
  } catch (err) {
    return (err.stdout || "").toString().trim();
  }
}

function getProcessCommandLine(pid) {
  if (!pid) return "";
  if (process.platform === "win32") {
    const escaped = String(pid).replace(/'/g, "''");
    return execCapture("powershell.exe", ["-NoProfile", "-Command", `(Get-CimInstance Win32_Process -Filter \"ProcessId=${escaped}\").CommandLine`]);
  }
  return execCapture("ps", ["-p", String(pid), "-o", "command="]);
}

function parsePidFromText(text) {
  const match = String(text || "").match(/pid=(\d+)/i) || String(text || "").match(/PID[=: ]+(\d+)/i);
  return match ? Number.parseInt(match[1], 10) : null;
}

function getListeningPid(port = BACKEND_PORT) {
  if (process.platform === "win32") {
    const ps = execCapture("powershell.exe", ["-NoProfile", "-Command", `$c=Get-NetTCPConnection -LocalPort ${port} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($c) { $c.OwningProcess }`]);
    const psPid = Number.parseInt(ps, 10);
    if (Number.isFinite(psPid)) return psPid;
    const out = execCapture("cmd.exe", ["/c", `netstat -ano -p tcp | findstr :${port}`]);
    const line = out.split(/\r?\n/).find((row) => row.includes("LISTENING"));
    const pid = line?.trim().split(/\s+/).pop();
    const parsed = pid ? Number.parseInt(pid, 10) : null;
    return Number.isFinite(parsed) ? parsed : null;
  }
  const lsof = execCapture("lsof", ["-nP", `-iTCP:${port}`, "-sTCP:LISTEN"]);
  const line = lsof.split(/\r?\n/).find((row) => /\bLISTEN\b/.test(row) && !row.startsWith("COMMAND"));
  if (line) {
    const pid = Number.parseInt(line.trim().split(/\s+/)[1], 10);
    if (Number.isFinite(pid)) return pid;
  }
  const ss = execCapture("ss", ["-ltnp", `sport = :${port}`]);
  const ssPid = parsePidFromText(ss);
  return Number.isFinite(ssPid) ? ssPid : null;
}

async function isPidAlive(pid) {
  if (!pid) return false;
  try {
    if (process.platform === "win32") {
      return execCapture("cmd.exe", ["/c", `tasklist /FI \"PID eq ${pid}\"`]).includes(String(pid));
    }
    process.kill(pid, 0);
    return true;
  } catch (_) { return false; }
}

function isDepthLensOwnedProcess({
  pid,
  commandLine = "",
  storedMetadata = null,
  cwd = getAppRoot(),
  backendDir = getResourcePath("backend"),
}) {
  const storedPidMatchesMetadata = Boolean(
    pid
    && storedMetadata?.pid
    && Number(pid) === Number(storedMetadata.pid)
    && storedMetadata?.host === BACKEND_HOST
    && Number(storedMetadata?.port) === Number(BACKEND_PORT)
  );

  return storedPidMatchesMetadata && isDepthLensOwnedProcessPolicy({
    commandLine,
    storedMetadata,
    cwd,
    backendDir,
  });
}

async function terminatePid(pid, { force = false } = {}) {
  if (!pid) return;
  if (process.platform === "win32") {
    const args = ["/PID", String(pid), "/T"];
    if (force) args.push("/F");
    execCapture("taskkill.exe", args, { timeout: 5000 });
  } else {
    try { process.kill(pid, force ? "SIGKILL" : "SIGTERM"); } catch (err) { log.warn(`Failed to send ${force ? "SIGKILL" : "SIGTERM"} to ${pid}: ${err.message}`); }
  }
}

async function inspectBackendPort() {
  const storedPid = readStoredBackendPid();
  const storedMetadata = readStoredBackendMetadata();
  const listeningPid = getListeningPid(BACKEND_PORT);
  const pid = listeningPid || storedPid;
  const commandLine = pid ? getProcessCommandLine(pid) : "";
  const pidMatchesPidFile = Boolean(pid && storedPid && Number(pid) === Number(storedPid));
  const depthLensOwned = isDepthLensOwnedProcess({ pid, commandLine, storedPid, storedMetadata });
  if (pid) log.info("PORT_OCCUPIED_PID_DETECTED", { pid, commandLine, pidMatchesPidFile, depthLensOwned });
  return { storedPid, storedMetadata, listeningPid, pid, commandLine, pidMatchesPidFile, depthLensOwned };
}

async function cleanupStaleBackendIfSafe(reason, details) {
  log.warn("STALE_BACKEND_DETECTED", { reason });
  const info = await inspectBackendPort();
  if (!info.pid || !info.depthLensOwned) {
    const pidText = info.pid ? String(info.pid) : "unknown";
    const command = info.pid
      ? (process.platform === "win32" ? `taskkill /PID ${info.pid} /F` : `kill ${info.pid}`)
      : "No safe kill command available because no PID was detected.";
    const { dialog } = require("electron");
    dialog.showMessageBoxSync({
      type: "error",
      title: "DepthLens Pro — Port 8765 Busy",
      message: "Port 8765 is occupied by a process that does not respond like DepthLens.",
      detail: `Detected PID: ${pidText}\nCommand line: ${info.commandLine || "unknown"}\nExact command to run if this is safe to terminate: ${command}`,
      buttons: ["OK"],
    });
    throw createStartupError(`Port ${BACKEND_PORT} is occupied by a non-DepthLens or unidentified process. Detected PID: ${pidText}.`, details);
  }

  log.warn("STALE_BACKEND_SIGTERM_SENT", { pid: info.pid });
  await terminatePid(info.pid, { force: false });
  await delay(3000);

  let remainingPid = getListeningPid(BACKEND_PORT);
  if (remainingPid && remainingPid === info.pid && isDepthLensOwnedProcess({ pid: remainingPid, commandLine: getProcessCommandLine(remainingPid), storedPid: info.storedPid, storedMetadata: info.storedMetadata })) {
    log.warn("STALE_BACKEND_SIGKILL_SENT", { pid: remainingPid });
    await terminatePid(remainingPid, { force: true });
    await delay(1000);
  }

  remainingPid = getListeningPid(BACKEND_PORT);
  if (remainingPid) {
    log.error("STALE_BACKEND_CLEANUP_FAILED", { pid: remainingPid, commandLine: getProcessCommandLine(remainingPid), port: BACKEND_PORT });
    throw createStartupError(`Could not free stale DepthLens backend on port ${BACKEND_PORT}. Remaining PID: ${remainingPid}.`, details);
  }
  log.info("STALE_BACKEND_TERMINATED", { port: BACKEND_PORT, finalPortState: "free" });
  removeBackendPidFiles();
}

// ── Python executable detection ────────────────────────────
function getPythonCandidates() {
  const root = getAppRoot();
  const candidates = [];

  if (process.platform === "win32") {
    candidates.push(path.join(root, "venv", "Scripts", "python.exe"));
    candidates.push(path.join(root, "venv", "Scripts", "python3.exe"));
    candidates.push(path.join(process.resourcesPath || root, "venv", "Scripts", "python.exe"));
    if (isDev) {
      candidates.push("py");
      candidates.push("python");
    }
  } else {
    candidates.push(path.join(root, "venv", "bin", "python3"));
    candidates.push(path.join(root, "venv", "bin", "python"));
    candidates.push(path.join(process.resourcesPath || root, "venv", "bin", "python3"));
    candidates.push(path.join(process.resourcesPath || root, "venv", "bin", "python"));
    if (isDev) {
      candidates.push("python3");
      candidates.push("python");
      candidates.push("/usr/bin/python3");
      candidates.push("/usr/local/bin/python3");
      candidates.push("/opt/homebrew/bin/python3");
    }
  }

  return [...new Set(candidates)];
}

function pythonVersionOk(candidate) {
  try {
    const output = execCapture(candidate, ["-c", "import sys,json; print(json.dumps(list(sys.version_info[:3])))"], { timeout: 3000 });
    const version = JSON.parse(output || "[]");
    const ok = version.length >= 2 && version[0] === 3 && version[1] >= 10 && version[1] <= 12;
    log.info("PYTHON_VERSION_PROBE", { candidate, version, ok });
    return ok;
  } catch (err) {
    log.warn("PYTHON_VERSION_PROBE_FAILED", { candidate, error: err.message });
    return false;
  }
}

function getPythonPath() {
  const candidates = getPythonCandidates();
  for (const candidate of candidates) {
    try {
      if (path.isAbsolute(candidate)) {
        if (fs.existsSync(candidate) && pythonVersionOk(candidate)) {
          log.info(`Python found at: ${candidate}`);
          return candidate;
        }
      } else {
        if (pythonVersionOk(candidate)) {
          log.info(`Using PATH Python candidate: ${candidate}`);
          return candidate;
        }
      }
    } catch (_) {
      // Ignore bad candidate paths and continue.
    }
  }
  const fallback = candidates[0];
  log.warn(`No supported Python found in candidate paths; will try: ${fallback}`);
  return fallback;
}

function pathExists(targetPath) {
  try { return fs.existsSync(targetPath); } catch (_) { return false; }
}

function createStartupDetails(pythonPath, backendDir, cwd, command) {
  const frontendDir = path.join(cwd, "frontend");
  const backendApp = path.join(backendDir, "app.py");
  const frontendIndex = path.join(frontendDir, "index.html");
  const modelsDir = path.join(cwd, "models");
  const onnxDir = path.join(modelsDir, "onnx");
  return {
    backendUrl,
    pythonPath,
    backendDir,
    backendApp,
    frontendDir,
    frontendIndex,
    modelsDir,
    onnxDir,
    cwd,
    command,
    isPackaged: app.isPackaged,
    resourceKind: app.isPackaged ? "packaged resources" : "repo-root resources",
    pythonExists: path.isAbsolute(pythonPath) ? pathExists(pythonPath) : "PATH lookup",
    backendDirExists: pathExists(backendDir),
    backendAppExists: pathExists(backendApp),
    frontendDirExists: pathExists(frontendDir),
    frontendIndexExists: pathExists(frontendIndex),
    modelsDirExists: pathExists(modelsDir),
    onnxDirExists: pathExists(onnxDir),
    logPath: logPath(),
  };
}

function createStartupError(message, details, exitInfo = {}) {
  return new Error(
    `${message}\n\n` +
    `Resource context: ${details.resourceKind || "unknown"}\n` +
    `Backend URL: ${details.backendUrl}\n` +
    `Python path attempted: ${details.pythonPath} (exists: ${details.pythonExists})\n` +
    `Backend directory attempted: ${details.backendDir} (exists: ${details.backendDirExists})\n` +
    `Backend app attempted: ${details.backendApp} (exists: ${details.backendAppExists})\n` +
    `Frontend directory attempted: ${details.frontendDir} (exists: ${details.frontendDirExists})\n` +
    `Frontend index attempted: ${details.frontendIndex} (exists: ${details.frontendIndexExists})\n` +
    `Models directory attempted: ${details.modelsDir} (exists: ${details.modelsDirExists})\n` +
    `ONNX directory attempted: ${details.onnxDir} (exists: ${details.onnxDirExists})\n` +
    `Working directory attempted: ${details.cwd}\n` +
    `Command attempted: ${details.command}\n` +
    `Backend exit code: ${exitInfo.code ?? "n/a"}\n` +
    `Backend exit signal: ${exitInfo.signal ?? "n/a"}\n` +
    `Electron logs: ${details.logPath}`,
  );
}


function isLikelyInstalledAppPath(targetPath) {
  const normalized = targetPath.replace(/\\/g, "/");
  if (process.platform === "darwin") return normalized.startsWith("/Applications/DepthLens Pro.app/") || normalized.includes("/Applications/DepthLens Pro.app/");
  if (process.platform === "win32") return /\/program files( \(arm\))?\/depthlens pro\//i.test(normalized) || /\/appdata\/local\/programs\/depthlens pro\//i.test(normalized);
  if (process.platform === "linux") return normalized.startsWith("/opt/DepthLens Pro/") || normalized.startsWith("/usr/lib/depthlens-pro/") || normalized.includes("/.local/share/applications/");
  return false;
}

function missingResourceEntries(details) {
  const entries = [];
  if (path.isAbsolute(details.pythonPath) && !details.pythonExists) entries.push(["platform Python executable", details.pythonPath]);
  if (!details.backendDirExists) entries.push(["backend/", details.backendDir]);
  if (!details.backendAppExists) entries.push(["backend/app.py", details.backendApp]);
  if (!details.frontendDirExists) entries.push(["frontend/", details.frontendDir]);
  if (!details.frontendIndexExists) entries.push(["frontend/index.html", details.frontendIndex]);
  if (app.isPackaged || details.modelsDirExists === false) {
    if (!details.modelsDirExists) entries.push(["models/", details.modelsDir]);
    if (!details.onnxDirExists) entries.push(["models/onnx/", details.onnxDir]);
  }
  return entries;
}

function createMissingResourcesError(details) {
  const missing = missingResourceEntries(details);
  const missingText = missing.map(([label, target]) => `- ${label}: ${target}`).join("\n") || "- Unknown required resource";
  const context = app.isPackaged ? "packaged app resources" : "repo-root development resources";
  const remediation = app.isPackaged
    ? [
        "Rebuild with the supported root native build script so packaged resources are verified after electron-builder finishes:",
        "  macOS ARM: scripts/build-native-macos.sh",
        "  Windows ARM: .\\scripts\\build-native-windows.ps1",
        "  Linux ARM: scripts/build-native-linux.sh",
        isLikelyInstalledAppPath(details.cwd)
          ? "This app appears to be running from an installed location. You may be launching a stale installed copy; replace it with the newly built artifact before launching again."
          : "If you installed the app previously, remove or replace the stale installed copy before launching again.",
      ].join("\n")
    : [
        "Run setup from the repository root before launching development mode:",
        "  macOS/Linux: npm run setup",
        "  Windows: npm run setup:win",
        "Then verify with: cd electron-app && npm run verify:resources:native",
      ].join("\n");
  return createStartupError(`Missing required ${context}:\n${missingText}\n\n${remediation}`, details);
}

function dependencyFailureHint() {
  const output = backendOutputExcerpt();
  if (/ModuleNotFoundError|ImportError|No module named/i.test(output)) {
    return `\n\nBackend dependency/import failure detected in backend output. Re-run setup and pip check for the packaged venv. Recent backend output:\n${output}`;
  }
  return `\n\nRecent backend output:\n${output}`;
}

function waitForBackendReady(url, details, { timeoutMs = 45_000, intervalMs = 500 } = {}) {
  const start = Date.now();
  let settled = false;
  let timer = null;

  return new Promise((resolve, reject) => {
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      fn(value);
    };

    const poll = async () => {
      if (settled) return;
      const attempt = Math.floor((Date.now() - start) / intervalMs) + 1;
      const liveProbe = await probeLive(url, 1200, attempt);
      if (liveProbe.ok) {
        backendReady = true;
        log.info(`Backend live at ${url}`);
        finish(resolve, url);
        return;
      }
      if (backendProcess?.exitCode !== null && backendProcess?.exitCode !== undefined) {
        finish(reject, createStartupError(`Backend process exited before /live became available.${dependencyFailureHint()}`, details, { code: backendProcess.exitCode, signal: backendProcess.signalCode }));
        return;
      }
      if (Date.now() - start >= timeoutMs) {
        finish(reject, createStartupError(`Backend did not become live within ${Math.round(timeoutMs / 1000)}s at ${url}.${dependencyFailureHint()}`, details, { code: backendProcess?.exitCode, signal: backendProcess?.signalCode }));
        return;
      }
      timer = setTimeout(poll, intervalMs);
    };

    poll();
  });
}

// ── Start FastAPI Backend ──────────────────────────────────
async function startBackend() {
  BACKEND_PORT = REQUESTED_BACKEND_PORT;
  backendUrl = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
  const pythonPath = getPythonPath();
  const backendDir = getResourcePath("backend");
  const cwd = getAppRoot();
  const args = [
    "-m",
    "uvicorn",
    "backend.app:app",
    "--host",
    BACKEND_HOST,
    "--port",
    String(BACKEND_PORT),
    "--workers",
    "1",
  ];
  const command = `${pythonPath} ${args.join(" ")}`;
  const details = createStartupDetails(pythonPath, backendDir, cwd, command);

  log.info(`Backend URL: ${backendUrl}`);
  log.info(`Backend dir: ${backendDir}`);
  log.info(`Backend cwd: ${cwd}`);
  log.info(`Starting backend command: ${command}`);

  const initialLiveProbe = await probeLive(backendUrl, 1200, 0);
  if (initialLiveProbe.ok) {
    backendReady = true;
    backendOwnedByElectron = false;
    log.info("EXISTING_BACKEND_LIVE_REUSED", { backendUrl });
    return backendUrl;
  }

  const portAvailable = await isPortAvailable(BACKEND_PORT).catch((err) => {
    throw createStartupError(`Could not inspect backend port ${BACKEND_PORT}: ${err.message}`, details);
  });
  if (!portAvailable) {
    log.warn("PORT_OCCUPIED", { port: BACKEND_PORT, liveKind: initialLiveProbe.kind });
    const fallbackPort = await findAvailableBackendPort(BACKEND_PORT + 1);
    if (fallbackPort && !process.env.DEPTHLENS_BACKEND_PORT) {
      BACKEND_PORT = fallbackPort;
      backendUrl = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
      details.backendUrl = backendUrl;
      details.command = details.command.replace(/--port\s+\d+/, `--port ${BACKEND_PORT}`);
      log.warn("PORT_FALLBACK_SELECTED", { requestedPort: REQUESTED_BACKEND_PORT, backendUrl });
    } else {
      await cleanupStaleBackendIfSafe(initialLiveProbe.kind || "live_probe_failed", details);
    }
  }

  if (missingResourceEntries(details).length > 0) {
    throw createMissingResourcesError(details);
  }

  args[args.indexOf("--port") + 1] = String(BACKEND_PORT);

  // Keep shell disabled so PATH-based Python commands on Windows are resolved by
  // CreateProcess/SearchPath instead of cmd.exe; uvicorn options remain a true
  // argument array and cannot be interpreted as shell metacharacters.
  backendProcess = spawn(pythonPath, args, {
    cwd,
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      PYTHONPATH: [cwd, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
      DEPTHLENSPRO_MODEL_DIR: path.join(cwd, "models"),
      DEPTHLENS_ONNX_DIR: path.join(cwd, "models", "onnx"),
      DEPTHLENS_PACKAGED_APP: app.isPackaged ? "1" : "0",
      DEPTHLENS_USER_DATA_DIR: app.getPath("userData"),
      SSL_CERT_FILE: process.env.SSL_CERT_FILE || "",
      REQUESTS_CA_BUNDLE: process.env.REQUESTS_CA_BUNDLE || process.env.SSL_CERT_FILE || "",
    },
    shell: false,
    windowsHide: true,
  });
  backendOwnedByElectron = true;
  backendPid = backendProcess.pid;
  backendMetadata = {
    pid: backendPid,
    backendUrl,
    port: BACKEND_PORT,
    host: BACKEND_HOST,
    cwd,
    backendDir,
    pythonPath,
    command: `${pythonPath} ${args.join(" ")}`,
    startedAt: new Date().toISOString(),
    appVersion: app.getVersion(),
    isPackaged: app.isPackaged,
    platform: process.platform,
    arch: process.arch,
  };
  writeBackendPidFiles(backendMetadata);

  backendProcess.stdout.on("data", (data) => {
    const msg = data.toString().trim();
    if (msg) {
      rememberBackendOutput("stdout", msg);
      log.info(`[Backend stdout] ${msg}`);
    }
  });

  backendProcess.stderr.on("data", (data) => {
    const msg = data.toString().trim();
    if (msg) {
      rememberBackendOutput("stderr", msg);
      log.info(`[Backend stderr] ${msg}`);
    }
  });

  backendProcess.on("error", (err) => {
    log.error(`Backend spawn error: ${err.message}`);
  });

  backendProcess.on("close", (code, signal) => {
    log.info(`Backend exited with code ${code}, signal ${signal}`);
    backendReady = false;
    if (!shutdownInProgress) {
      backendOwnedByElectron = false;
      backendProcess = null;
    }
  });

  try {
    return await waitForBackendReady(backendUrl, details);
  } catch (err) {
    if (backendProcess && !backendProcess.killed) await terminatePid(backendProcess.pid, { force: false });
    if (err.message?.includes("Backend URL:")) throw err;
    throw createStartupError(err.message, details, { code: backendProcess?.exitCode, signal: backendProcess?.signalCode });
  }
}

// ── Splash Window ──────────────────────────────────────────
function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 480,
    height: 300,
    frame: false,
    transparent: true,
    resizable: false,
    alwaysOnTop: true,
    webPreferences: { nodeIntegration: false, contextIsolation: true },
  });

  splashWindow.loadFile(path.join(__dirname, "src", "splash.html"));
  splashWindow.center();
}

// ── Main Window ────────────────────────────────────────────
function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    show: false,
    titleBarStyle: process.platform === "darwin" ? "hidden" : "default",
    trafficLightPosition: process.platform === "darwin" ? { x: 16, y: 18 } : undefined,
    vibrancy: "under-window",
    visualEffectState: "active",
    backgroundColor: "#070d17",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      preload: path.join(__dirname, "preload.js"),
      webSecurity: true,
    },
  });

  const frontendPath = getResourcePath("frontend", "index.html");
  mainWindow.loadFile(frontendPath);

  mainWindow.once("ready-to-show", () => {
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.destroy();
    }
    mainWindow.show();
    mainWindow.focus();
    log.info("Main window shown");
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isExternalUrl(url)) {
      shell.openExternal(url);
    } else {
      log.warn(`Blocked new-window request to: ${url}`);
    }
    return { action: "deny" };
  });

  const shouldOpenDevTools = process.env.DEPTHLENS_OPEN_DEVTOOLS === "1";

  if (isDev && shouldOpenDevTools) {
    mainWindow.webContents.openDevTools({ mode: "detach" });
  }
}

// ── IPC Handlers ──────────────────────────────────────────

function storageFilePath() { return path.join(app.getPath("userData"), "renderer-state.json"); }
function readRendererStore() {
  try { return JSON.parse(fs.readFileSync(storageFilePath(), "utf8")); }
  catch (_) { return {}; }
}
function writeRendererStore(data) {
  fs.mkdirSync(app.getPath("userData"), { recursive: true });
  atomicWriteFile(storageFilePath(), JSON.stringify(data, null, 2) + "\n");
}

ipcMain.handle("get-backend-url", () => backendUrl);
ipcMain.handle("get-app-version", () => app.getVersion());
ipcMain.handle("get-platform", () => process.platform);
ipcMain.handle("get-backend-live-path", () => "/live");
ipcMain.handle("storage-path", () => app.getPath("userData"));
ipcMain.handle("settings-get", (_event, key) => { const store = readRendererStore(); return key ? store.settings?.[key] : (store.settings || {}); });
ipcMain.handle("settings-set", (_event, key, value) => { const store = readRendererStore(); store.settings = store.settings || {}; store.settings[key] = value; writeRendererStore(store); return true; });
ipcMain.handle("benchmarks-history", () => readRendererStore().benchmarkRuns || []);
ipcMain.handle("models-status", async () => ({ backendUrl, endpoint: `${backendUrl}/api/models/status` }));
ipcMain.handle("cache-clear", () => { const store = readRendererStore(); store.cacheClearedAt = new Date().toISOString(); writeRendererStore(store); return true; });

ipcMain.handle("show-save-dialog", async (event, options) => {
  const { dialog } = require("electron");
  return dialog.showSaveDialog(mainWindow, options);
});

ipcMain.handle("show-open-dialog", async (event, options) => {
  const { dialog } = require("electron");
  return dialog.showOpenDialog(mainWindow, options);
});

// ── App Lifecycle ──────────────────────────────────────────
if (singleInstanceLock) app.whenReady().then(async () => {
  log.info(
    `App ready — isDev: ${isDev}, platform: ${process.platform}, arch: ${process.arch}`,
  );

  if (!isSupportedArchitecture()) {
    showUnsupportedArchitectureDialog();
    app.quit();
    return;
  }
  log.info("SUPPORTED_ARCH_CHECK_PASSED", { platform: process.platform, arch: process.arch });

  createSplashWindow();

  try {
    await startBackend();
    log.info(`Backend ready for renderer at ${backendUrl}`);
    createMainWindow();
  } catch (err) {
    log.error(`Failed to start backend: ${err.message}`);
    const { dialog } = require("electron");

    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.destroy();
    }

    const choice = dialog.showMessageBoxSync({
      type: "error",
      title: "DepthLens Pro — Backend Error",
      message: "Could not start the inference engine.",
      detail:
        `${err.message}\n\n` +
        `The app will still open but inference will be unavailable.`,
      buttons: ["Open Anyway", "Quit"],
      defaultId: 0,
      cancelId: 1,
    });

    if (choice === 1) {
      backendOwnedByElectron = false;
      app.quit();
      return;
    }

    createMainWindow();
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

let isQuitting = false;

async function shutdownOwnedBackend() {
  if (!backendOwnedByElectron || !backendPid) return true;
  shutdownInProgress = true;
  const pid = backendPid;
  log.info("App quitting — shutting down backend", { pid, port: BACKEND_PORT });

  await terminatePid(pid, { force: false });
  const deadline = Date.now() + 3000;
  while (Date.now() < deadline && await isPidAlive(pid)) {
    await delay(150);
  }

  if (await isPidAlive(pid)) {
    log.warn("Backend did not exit gracefully, forcing SIGKILL", { pid });
    await terminatePid(pid, { force: true });
    await delay(750);
  }

  const alive = await isPidAlive(pid);
  const listeningPid = getListeningPid(BACKEND_PORT);
  if (!alive && (!listeningPid || listeningPid !== pid)) {
    removeBackendPidFiles();
    backendProcess = null;
    backendPid = null;
    backendMetadata = null;
    backendOwnedByElectron = false;
    backendReady = false;
    shutdownInProgress = false;
    log.info("Backend shut down successfully", { pid, port: BACKEND_PORT, finalPortState: listeningPid ? `occupied by ${listeningPid}` : "free" });
    return true;
  }

  const commandLine = getProcessCommandLine(pid || listeningPid);
  log.error("BACKEND_SHUTDOWN_VERIFY_FAILED", { pid, alive, port: BACKEND_PORT, listeningPid, commandLine });
  shutdownInProgress = false;
  return false;
}

app.on("before-quit", (event) => {
  if (backendProcess && backendOwnedByElectron && !isQuitting) {
    event.preventDefault();
    isQuitting = true;
    shutdownOwnedBackend().finally(() => {
      app.quit();
    });
  }
});

// ── Navigation policy ─────────────────────────────────────
app.on("web-contents-created", (event, contents) => {
  contents.on("will-navigate", (event, url) => {
    const frontendPath = getResourcePath("frontend", "index.html");
    if (!isAllowedAppUrl(url, { backendHost: BACKEND_HOST, backendPort: BACKEND_PORT, frontendPath })) {
      log.warn(`Blocked navigation to: ${url}`);
      event.preventDefault();
    }
  });
});
