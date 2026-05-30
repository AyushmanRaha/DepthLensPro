const { app, BrowserWindow, ipcMain, shell } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");
const fs = require("fs");
const log = require("electron-log");

log.transports.file.level = "info";
log.info("DepthLens Pro starting...");

// ── Constants ──────────────────────────────────────────────
const BACKEND_PORT = 8765;
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;
const isDev = process.env.NODE_ENV === "development" || !app.isPackaged;

// ── State ──────────────────────────────────────────────────
let mainWindow = null;
let splashWindow = null;
let backendProcess = null;
let backendReady = false;

// ── Resource paths ─────────────────────────────────────────
function getResourcePath(...parts) {
  if (isDev) {
    return path.join(__dirname, "..", ...parts);
  }
  return path.join(process.resourcesPath, ...parts);
}

// ── Python executable detection ────────────────────────────
// FIX (Issue 1 & 2): Robust Python path resolution that works both in
// development AND inside a packaged .app on any machine.
// The packaged .app has the venv at:
//   DepthLens Pro.app/Contents/Resources/venv/bin/python3
// In dev the venv is at the repo root: <repo>/venv/bin/python3
// We try multiple candidate paths and pick the first one that exists.
function getPythonPath() {
  const candidates = [];

  if (isDev) {
    // Development: venv is one level above electron-app/
    candidates.push(path.join(__dirname, "..", "venv", "bin", "python3"));
    candidates.push(path.join(__dirname, "..", "venv", "bin", "python"));
  } else {
    // Packaged .app: Resources/ is process.resourcesPath
    candidates.push(path.join(process.resourcesPath, "venv", "bin", "python3"));
    candidates.push(path.join(process.resourcesPath, "venv", "bin", "python"));
    // Fallback: some builder configs embed it differently
    candidates.push(path.join(process.resourcesPath, "..", "venv", "bin", "python3"));
    candidates.push(path.join(app.getAppPath(), "..", "venv", "bin", "python3"));
  }

  // Also try system Python as a last resort so the app at least opens
  // and shows a meaningful error rather than crashing with ENOENT.
  candidates.push("/usr/bin/python3");
  candidates.push("/usr/local/bin/python3");
  candidates.push("/opt/homebrew/bin/python3"); // Apple Silicon Homebrew

  for (const candidate of candidates) {
    try {
      if (fs.existsSync(candidate)) {
        log.info(`Python found at: ${candidate}`);
        return candidate;
      }
    } catch (_) {
      // existsSync can throw on bad paths; just continue
    }
  }

  // Nothing found – return the most-expected path so the error message
  // in the dialog is helpful rather than generic.
  const fallback = isDev
    ? path.join(__dirname, "..", "venv", "bin", "python3")
    : path.join(process.resourcesPath, "venv", "bin", "python3");

  log.warn(`No Python found in candidate paths; will try: ${fallback}`);
  return fallback;
}

// ── Start FastAPI Backend ──────────────────────────────────
function startBackend() {
  return new Promise((resolve, reject) => {
    const pythonPath = getPythonPath();
    const backendDir = getResourcePath("backend");

    log.info(`Starting backend: ${pythonPath}`);
    log.info(`Backend dir: ${backendDir}`);

    // Verify python exists before spawning so we can give a clear error
    if (!fs.existsSync(pythonPath)) {
      const msg =
        `Python not found at: ${pythonPath}\n\n` +
        `Please ensure the virtual environment is set up:\n` +
        `  cd <project-root>\n` +
        `  python3 -m venv venv\n` +
        `  source venv/bin/activate\n` +
        `  pip install -r backend/requirements.txt`;
      log.error(msg);
      return reject(new Error(msg));
    }

    if (!fs.existsSync(backendDir)) {
      const msg = `Backend directory not found: ${backendDir}`;
      log.error(msg);
      return reject(new Error(msg));
    }

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
        // Inherit the shell environment so system libs and CUDA paths work
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
        },
      },
    );

    backendProcess.stdout.on("data", (data) => {
      const msg = data.toString();
      log.info(`[Backend] ${msg.trim()}`);
      if (
        msg.includes("Application startup complete") ||
        msg.includes("Uvicorn running on")
      ) {
        backendReady = true;
        resolve(BACKEND_URL);
      }
    });

    backendProcess.stderr.on("data", (data) => {
      const msg = data.toString().trim();
      log.info(`[Backend stderr] ${msg}`);
      // Uvicorn writes its startup banner to stderr, not stdout
      if (
        msg.includes("Application startup complete") ||
        msg.includes("Uvicorn running on")
      ) {
        if (!backendReady) {
          backendReady = true;
          resolve(BACKEND_URL);
        }
      }
    });

    backendProcess.on("error", (err) => {
      log.error(`Backend spawn error: ${err.message}`);
      reject(err);
    });

    backendProcess.on("close", (code) => {
      log.info(`Backend exited with code ${code}`);
      backendReady = false;
    });

    // Fallback: poll the health endpoint regardless of log output
    // (handles cases where uvicorn logs to neither stdout nor stderr)
    setTimeout(() => {
      if (!backendReady) {
        log.warn("Startup log not detected; polling /health endpoint...");
        pollBackendHealth(30, 1000, resolve, reject);
      }
    }, 3000);

    // Hard timeout – resolve anyway so the window opens and shows status
    setTimeout(() => {
      if (!backendReady) {
        log.warn("Backend startup timeout – resolving to allow app to open.");
        backendReady = true;
        resolve(BACKEND_URL);
      }
    }, 20000);
  });
}

function pollBackendHealth(maxAttempts, intervalMs, resolve, reject) {
  let attempts = 0;

  const check = () => {
    attempts++;
    const req = http.get(`${BACKEND_URL}/health`, { timeout: 2000 }, (res) => {
      if (res.statusCode === 200) {
        backendReady = true;
        log.info(`Backend ready after ${attempts} health-poll attempts`);
        resolve(BACKEND_URL);
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
      // Don't reject – the window will show "offline" status instead
      log.warn(`Backend did not respond after ${maxAttempts} polls`);
      resolve(BACKEND_URL);
      return;
    }
    setTimeout(check, intervalMs);
  }

  setTimeout(check, 800);
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
    // FIX (Issue 4): Use "hidden" instead of "hiddenInset" on macOS.
    // "hiddenInset" keeps the traffic-light buttons but moves them INSIDE
    // the window frame, where they overlap the header content.
    // "hidden" removes the title bar entirely and we handle dragging via CSS.
    titleBarStyle: process.platform === "darwin" ? "hidden" : "default",
    // On macOS, expose the traffic-light inset size so the renderer can pad
    trafficLightPosition: process.platform === "darwin"
      ? { x: 16, y: 18 }
      : undefined,
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
ipcMain.handle("get-backend-url", () => BACKEND_URL);
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
    log.info("Backend started (or timed out gracefully)");
    createMainWindow();
  } catch (err) {
    log.error(`Failed to start backend: ${err.message}`);
    const { dialog } = require("electron");

    // Destroy splash before showing dialog
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.destroy();
    }

    const choice = dialog.showMessageBoxSync({
      type: "error",
      title: "DepthLens Pro — Backend Error",
      message: "Could not start the inference engine.",
      detail:
        `${err.message}\n\n` +
        `The app will still open but inference will be unavailable.\n` +
        `See logs at: ${log.transports.file.getFile().path}`,
      buttons: ["Open Anyway", "Quit"],
      defaultId: 0,
      cancelId: 1,
    });

    if (choice === 1) {
      app.quit();
      return;
    }

    // Open the window anyway so the user can see the offline status
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
  if (backendProcess && !isQuitting) {
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