const { app, BrowserWindow, ipcMain, shell } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");
const net = require("net");
const fs = require("fs");
const log = require("electron-log");

log.transports.file.level = "info";
log.info("DepthLens Pro starting...");

// ── Constants ──────────────────────────────────────────────
const BACKEND_HOST = "127.0.0.1";
const BACKEND_PORT = Number.parseInt(process.env.DEPTHLENS_BACKEND_PORT || "8765", 10);
let backendUrl = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
const isDev = process.env.NODE_ENV === "development" || !app.isPackaged;

// ── State ──────────────────────────────────────────────────
let mainWindow = null;
let splashWindow = null;
let backendProcess = null;
let backendReady = false;
let backendOwnedByElectron = false;

// ── Resource paths ─────────────────────────────────────────
function getResourcePath(...parts) {
  if (isDev) {
    return path.join(__dirname, "..", ...parts);
  }
  return path.join(process.resourcesPath, ...parts);
}

function getAppRoot() {
  return isDev ? path.join(__dirname, "..") : process.resourcesPath;
}

function logPath() {
  return log.transports.file.getFile().path;
}

// ── HTTP helpers ───────────────────────────────────────────
function requestJson(url, timeoutMs = 2000) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, { timeout: timeoutMs }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { body += chunk; });
      res.on("end", () => {
        let json = null;
        try { json = body ? JSON.parse(body) : null; } catch (_) {}
        resolve({ statusCode: res.statusCode, json, body });
      });
    });
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy(new Error(`Timed out requesting ${url}`));
    });
  });
}

async function isHealthyDepthLensBackend(url) {
  try {
    const health = await requestJson(`${url}/health`, 2500);
    if (health.statusCode !== 200 || !health.json || typeof health.json !== "object") return false;
    if (health.json.version && (health.json.devices || health.json.primary_device)) return true;
    const root = await requestJson(`${url}/`, 1500).catch(() => null);
    return Boolean(root?.json?.service === "DepthLens Pro API");
  } catch (_) {
    return false;
  }
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

// ── Python executable detection ────────────────────────────
function getPythonCandidates() {
  const root = getAppRoot();
  const candidates = [];

  if (process.platform === "win32") {
    candidates.push(path.join(root, "venv", "Scripts", "python.exe"));
    candidates.push(path.join(root, "venv", "Scripts", "python3.exe"));
    candidates.push(path.join(process.resourcesPath || root, "venv", "Scripts", "python.exe"));
    candidates.push("py");
    candidates.push("python");
  } else {
    candidates.push(path.join(root, "venv", "bin", "python3"));
    candidates.push(path.join(root, "venv", "bin", "python"));
    candidates.push(path.join(process.resourcesPath || root, "venv", "bin", "python3"));
    candidates.push(path.join(process.resourcesPath || root, "venv", "bin", "python"));
    candidates.push("python3");
    candidates.push("python");
    candidates.push("/usr/bin/python3");
    candidates.push("/usr/local/bin/python3");
    candidates.push("/opt/homebrew/bin/python3");
  }

  return [...new Set(candidates)];
}

function getPythonPath() {
  const candidates = getPythonCandidates();
  for (const candidate of candidates) {
    try {
      if (path.isAbsolute(candidate)) {
        if (fs.existsSync(candidate)) {
          log.info(`Python found at: ${candidate}`);
          return candidate;
        }
      } else {
        log.info(`Using PATH Python candidate: ${candidate}`);
        return candidate;
      }
    } catch (_) {
      // Ignore bad candidate paths and continue.
    }
  }
  const fallback = candidates[0];
  log.warn(`No Python found in candidate paths; will try: ${fallback}`);
  return fallback;
}

function createStartupError(message, details) {
  return new Error(
    `${message}\n\n` +
    `Backend URL: ${details.backendUrl}\n` +
    `Python path attempted: ${details.pythonPath}\n` +
    `Backend directory attempted: ${details.backendDir}\n` +
    `Working directory attempted: ${details.cwd}\n` +
    `Command attempted: ${details.command}\n` +
    `Electron logs: ${logPath()}`,
  );
}

function waitForBackendReady(url, { timeoutMs = 45_000, intervalMs = 800 } = {}) {
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
      if (await isHealthyDepthLensBackend(url)) {
        backendReady = true;
        log.info(`Backend ready at ${url}`);
        finish(resolve, url);
        return;
      }
      if (Date.now() - start >= timeoutMs) {
        finish(reject, new Error(`Backend did not become healthy within ${Math.round(timeoutMs / 1000)}s at ${url}`));
        return;
      }
      timer = setTimeout(poll, intervalMs);
    };

    poll();
  });
}

// ── Start FastAPI Backend ──────────────────────────────────
async function startBackend() {
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
  const details = { backendUrl, pythonPath, backendDir, cwd, command };

  log.info(`Backend URL: ${backendUrl}`);
  log.info(`Backend dir: ${backendDir}`);
  log.info(`Backend cwd: ${cwd}`);
  log.info(`Starting backend command: ${command}`);

  if (await isHealthyDepthLensBackend(backendUrl)) {
    backendReady = true;
    backendOwnedByElectron = false;
    log.info(`Reusing existing healthy DepthLens backend at ${backendUrl}`);
    return backendUrl;
  }

  const portAvailable = await isPortAvailable(BACKEND_PORT).catch((err) => {
    throw createStartupError(`Could not inspect backend port ${BACKEND_PORT}: ${err.message}`, details);
  });
  if (!portAvailable) {
    throw createStartupError(
      `Port ${BACKEND_PORT} is already in use, but it is not a healthy DepthLens backend. Free the port or set DEPTHLENS_BACKEND_PORT consistently before launching.`,
      details,
    );
  }

  if (path.isAbsolute(pythonPath) && !fs.existsSync(pythonPath)) {
    throw createStartupError(
      "Python was not found. Create the repo-root virtual environment and install backend requirements before launching.",
      details,
    );
  }

  if (!fs.existsSync(backendDir)) {
    throw createStartupError("Backend directory was not found in app resources.", details);
  }

  backendProcess = spawn(pythonPath, args, {
    cwd,
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      PYTHONPATH: [cwd, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
    },
    shell: process.platform === "win32" && !path.isAbsolute(pythonPath),
  });
  backendOwnedByElectron = true;

  backendProcess.stdout.on("data", (data) => {
    const msg = data.toString().trim();
    if (msg) log.info(`[Backend stdout] ${msg}`);
  });

  backendProcess.stderr.on("data", (data) => {
    const msg = data.toString().trim();
    if (msg) log.info(`[Backend stderr] ${msg}`);
  });

  backendProcess.on("error", (err) => {
    log.error(`Backend spawn error: ${err.message}`);
  });

  backendProcess.on("close", (code) => {
    log.info(`Backend exited with code ${code}`);
    backendReady = false;
    backendOwnedByElectron = false;
    backendProcess = null;
  });

  try {
    return await waitForBackendReady(backendUrl);
  } catch (err) {
    if (backendProcess && !backendProcess.killed) backendProcess.kill("SIGTERM");
    throw createStartupError(err.message, details);
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
    shell.openExternal(url);
    return { action: "deny" };
  });

  if (isDev) {
    mainWindow.webContents.openDevTools({ mode: "detach" });
  }
}

// ── IPC Handlers ──────────────────────────────────────────
ipcMain.handle("get-backend-url", () => backendUrl);
ipcMain.handle("get-app-version", () => app.getVersion());
ipcMain.handle("get-platform", () => process.platform);

ipcMain.handle("show-save-dialog", async (event, options) => {
  const { dialog } = require("electron");
  return dialog.showSaveDialog(mainWindow, options);
});

ipcMain.handle("show-open-dialog", async (event, options) => {
  const { dialog } = require("electron");
  return dialog.showOpenDialog(mainWindow, options);
});

// ── App Lifecycle ──────────────────────────────────────────
app.whenReady().then(async () => {
  log.info(
    `App ready — isDev: ${isDev}, platform: ${process.platform}, arch: ${process.arch}`,
  );

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

app.on("before-quit", (event) => {
  if (backendProcess && backendOwnedByElectron && !isQuitting) {
    event.preventDefault();
    log.info("App quitting — shutting down backend");

    isQuitting = true;
    backendProcess.kill("SIGTERM");

    const killTimer = setTimeout(() => {
      log.warn("Backend did not exit gracefully, forcing SIGKILL");
      if (backendProcess) {
        backendProcess.kill("SIGKILL");
      }
      app.quit();
    }, 3000);

    backendProcess.on("exit", () => {
      clearTimeout(killTimer);
      backendProcess = null;
      log.info("Backend shut down successfully");
      app.quit();
    });
  }
});

// ── Navigation policy ─────────────────────────────────────
app.on("web-contents-created", (event, contents) => {
  contents.on("will-navigate", (event, url) => {
    const parsedUrl = new URL(url);
    const allowedHosts = ["127.0.0.1", "localhost"];
    if (!allowedHosts.includes(parsedUrl.hostname)) {
      log.warn(`Blocked navigation to: ${url}`);
      event.preventDefault();
    }
  });
});
