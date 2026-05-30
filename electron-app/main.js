const { app, BrowserWindow, ipcMain, shell } = require("electron");
const path = require("path");
const { spawn, exec } = require("child_process");
const http = require("http");
const fs = require("fs");
const log = require("electron-log");

log.transports.file.level = "info";
log.info("DepthLens Pro starting...");

// ── Constants ──────────────────────────────────────────────
const BACKEND_PORT = 8765; // Avoid collisions with common local development ports.
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;
const isDev = process.env.NODE_ENV === "development" || !app.isPackaged;

// ── State ──────────────────────────────────────────────────
let mainWindow = null;
let splashWindow = null;
let backendProcess = null;
let backendReady = false;

// ── Resource paths ─────────────────────────────────────────
// Development uses repository-relative paths; packaged builds use resources.
function getResourcePath(...parts) {
  if (isDev) {
    return path.join(__dirname, "..", ...parts);
  }
  return path.join(process.resourcesPath, ...parts);
}

// ── Python executable detection ────────────────────────────
// Both development and packaged builds execute the bundled virtualenv.
function getPythonPath() {
  if (isDev) {
    return path.join(__dirname, "..", "venv", "bin", "python3");
  }
  return path.join(process.resourcesPath, "venv", "bin", "python3");
}

// ── Start FastAPI Backend ──────────────────────────────────
function startBackend() {
  return new Promise((resolve, reject) => {
    const pythonPath = getPythonPath();
    const backendDir = getResourcePath("backend");
    const backendScript = path.join(backendDir, "app.py");

    log.info(`Starting backend: ${pythonPath} at ${backendDir}`);

    // Launch Uvicorn through the same Python runtime that owns dependencies.
    backendProcess = spawn(
      pythonPath,
      [
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        "127.0.0.1",
        "--port",
        String(BACKEND_PORT),
        "--workers",
        "1",
      ],
      {
        cwd: backendDir,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    backendProcess.stdout.on("data", (data) => {
      const msg = data.toString();
      log.info(`[Backend] ${msg.trim()}`);
      if (msg.includes("Application startup complete")) {
        backendReady = true;
        resolve(BACKEND_URL);
      }
    });

    backendProcess.stderr.on("data", (data) => {
      log.error(`[Backend Err] ${data.toString().trim()}`);
    });

    backendProcess.on("close", (code) => {
      log.info(`Backend exited with code ${code}`);
      backendReady = false;
    });

    // Keep startup resilient when Uvicorn logs readiness on stderr.
    setTimeout(() => {
      if (!backendReady) {
        log.warn("Backend startup timeout, resolving anyway.");
        backendReady = true;
        resolve(BACKEND_URL);
      }
    }, 15000);
  });
}
function pollBackendHealth(maxAttempts, intervalMs, resolve, reject) {
  let attempts = 0;

  const check = () => {
    attempts++;
    const req = http.get(`${BACKEND_URL}/health`, { timeout: 2000 }, (res) => {
      if (res.statusCode === 200) {
        backendReady = true;
        log.info(`Backend ready after ${attempts} attempts`);
        resolve();
      } else {
        retry();
      }
      res.resume();
    });

    req.on("error", retry);
    req.on("timeout", () => {
      req.destroy();
      retry();
    });
  };

  function retry() {
    if (attempts >= maxAttempts) {
      reject(new Error(`Backend did not start after ${maxAttempts} attempts`));
      return;
    }
    setTimeout(check, intervalMs);
  }

  // Delay the first probe until Uvicorn has had a chance to bind the port.
  setTimeout(check, 1200);
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
    show: false, // Reveal the window only after backend startup completes.
    titleBarStyle: "hiddenInset",
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

  // Keep untrusted navigation outside the Electron renderer.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  if (isDev) {
    mainWindow.webContents.openDevTools({ mode: "detach" });
  }
}

// ── IPC Handlers ──────────────────────────────────────────
ipcMain.handle("get-backend-url", () => BACKEND_URL);

ipcMain.handle("get-app-version", () => app.getVersion());

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
    log.info("Backend started successfully");
    createMainWindow();
  } catch (err) {
    log.error(`Failed to start backend: ${err.message}`);
    const { dialog } = require("electron");
    dialog.showErrorBox(
      "DepthLens Pro — Startup Error",
      `Could not start the inference engine.\n\n${err.message}\n\nPlease check that Python dependencies are installed.\n\nSee logs at: ${log.transports.file.getFile().path}`,
    );
    app.quit();
  }

  app.on("activate", () => {
    // macOS reopens a window from the dock when no windows remain.
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow();
  });
});

app.on("window-all-closed", () => {
  // Match the native macOS convention of staying resident until Cmd+Q.
  if (process.platform !== "darwin") app.quit();
});

let isQuitting = false;

app.on("before-quit", (event) => {
  if (backendProcess && !isQuitting) {
    // Defer Electron shutdown until the backend exits or the kill timer fires.
    event.preventDefault();
    log.info("App quitting — shutting down backend");

    isQuitting = true;
    backendProcess.kill("SIGTERM");

    // Bound shutdown so a stuck Python process cannot hang the app.
    const killTimer = setTimeout(() => {
      log.warn("Backend did not exit gracefully, forcing SIGKILL");
      if (backendProcess) {
        backendProcess.kill("SIGKILL");
      }
      app.quit();
    }, 3000);

    // Exit after Python confirms termination.
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
