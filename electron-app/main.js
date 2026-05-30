const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path  = require('path');
const { spawn, exec } = require('child_process');
const http  = require('http');
const fs    = require('fs');
const log   = require('electron-log');

// Configure logging
log.transports.file.level = 'info';
log.info('DepthLens Pro starting...');

// ── Constants ──────────────────────────────────────────────
const BACKEND_PORT = 8765;   // use a non-standard port to avoid conflicts
const BACKEND_URL  = `http://127.0.0.1:${BACKEND_PORT}`;
const isDev        = process.env.NODE_ENV === 'development' || !app.isPackaged;

// ── State ──────────────────────────────────────────────────
let mainWindow    = null;
let splashWindow  = null;
let backendProcess = null;
let backendReady   = false;

// ── Resource path helper ───────────────────────────────────
// In development: paths are relative to project root
// In production:  files are in app.getPath('exe')/../Resources/
function getResourcePath(...parts) {
  if (isDev) {
    return path.join(__dirname, '..', ...parts);
  }
  return path.join(process.resourcesPath, ...parts);
}

// ── Python executable detection ────────────────────────────
// On M1 Mac with a venv, the python is inside the venv
// We ship a venv inside the app bundle for production
function getPythonPath() {
  return '/Users/user/Downloads/DepthLensPro/venv/bin/python3';
}

// ── Start FastAPI Backend ──────────────────────────────────
function startBackend() {
  return new Promise((resolve, reject) => {
    // const pythonPath  = getPythonPath();
    // const backendDir  = getResourcePath('backend');
    const pythonPath  = '/Users/user/Downloads/DepthLensPro/venv/bin/python3';
    const backendDir  = '/Users/user/Downloads/DepthLensPro/backend';
    const backendScript = path.join(backendDir, 'app.py');

    log.info(`Starting backend: ${pythonPath} at ${backendDir}`);

    // We run uvicorn programmatically via python -m uvicorn
    backendProcess = spawn(pythonPath, [
      '-m', 'uvicorn',
      'app:app',
      '--host', '127.0.0.1',
      '--port', String(BACKEND_PORT),
      '--log-level', 'warning',
      '--no-access-log',
    ], {
      cwd: backendDir,
      env: {
        ...process.env,
        // Tell Python where to find packages if using bundled venv
        PYTHONPATH: backendDir,
      },
    });

    backendProcess.stdout.on('data', (data) => {
      log.info(`[backend] ${data.toString().trim()}`);
    });

    backendProcess.stderr.on('data', (data) => {
      const msg = data.toString().trim();
      log.warn(`[backend] ${msg}`);
      // uvicorn logs startup on stderr — detect ready state
      if (msg.includes('Application startup complete')) {
        log.info('Backend startup signal detected');
      }
    });

    backendProcess.on('error', (err) => {
      log.error(`Backend process error: ${err.message}`);
      reject(err);
    });

    backendProcess.on('exit', (code) => {
      log.warn(`Backend exited with code: ${code}`);
      if (!backendReady) reject(new Error(`Backend exited early (code ${code})`));
    });

    // Poll the health endpoint until it responds
    pollBackendHealth(30, 800, resolve, reject);
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

    req.on('error', retry);
    req.on('timeout', () => { req.destroy(); retry(); });
  };

  function retry() {
    if (attempts >= maxAttempts) {
      reject(new Error(`Backend did not start after ${maxAttempts} attempts`));
      return;
    }
    setTimeout(check, intervalMs);
  }

  // Start first check after a short delay to give uvicorn time to bind
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

  splashWindow.loadFile(path.join(__dirname, 'src', 'splash.html'));
  splashWindow.center();
}

// ── Main Window ────────────────────────────────────────────
function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    show: false,           // shown after backend is ready
    titleBarStyle: 'hiddenInset',   // native Mac traffic-light buttons
    vibrancy: 'under-window',       // frosted glass effect on Mac
    visualEffectState: 'active',
    backgroundColor: '#070d17',
    webPreferences: {
      nodeIntegration: false,        // NEVER true — security
      contextIsolation: true,        // ALWAYS true — security
      sandbox: true,                 // extra sandboxing
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: true,
    },
  });

  // Load your existing frontend — no changes needed there
  const frontendPath = getResourcePath('frontend', 'index.html');
  mainWindow.loadFile(frontendPath);

  mainWindow.once('ready-to-show', () => {
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.destroy();
    }
    mainWindow.show();
    mainWindow.focus();
    log.info('Main window shown');
  });

  mainWindow.on('closed', () => { mainWindow = null; });

  // Open external links in system browser, not Electron
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // Dev tools in development only
  if (isDev) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }
}

// ── IPC Handlers ──────────────────────────────────────────
// These let the frontend call native capabilities safely via preload.js

ipcMain.handle('get-backend-url', () => BACKEND_URL);

ipcMain.handle('get-app-version', () => app.getVersion());

ipcMain.handle('show-save-dialog', async (event, options) => {
  const { dialog } = require('electron');
  return dialog.showSaveDialog(mainWindow, options);
});

ipcMain.handle('show-open-dialog', async (event, options) => {
  const { dialog } = require('electron');
  return dialog.showOpenDialog(mainWindow, options);
});

// ── App Lifecycle ──────────────────────────────────────────
app.whenReady().then(async () => {
  log.info(`App ready — isDev: ${isDev}, platform: ${process.platform}, arch: ${process.arch}`);

  // Show splash while backend loads
  createSplashWindow();

  try {
    await startBackend();
    log.info('Backend started successfully');
    createMainWindow();
  } catch (err) {
    log.error(`Failed to start backend: ${err.message}`);
    const { dialog } = require('electron');
    dialog.showErrorBox(
      'DepthLens Pro — Startup Error',
      `Could not start the inference engine.\n\n${err.message}\n\nPlease check that Python dependencies are installed.\n\nSee logs at: ${log.transports.file.getFile().path}`
    );
    app.quit();
  }

  app.on('activate', () => {
    // macOS: re-create window when dock icon is clicked and no windows open
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow();
  });
});

app.on('window-all-closed', () => {
  // On macOS, apps stay "running" until Cmd+Q
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  log.info('App quitting — shutting down backend');
  if (backendProcess) {
    backendProcess.kill('SIGTERM');
    // Force kill if graceful shutdown takes too long
    setTimeout(() => backendProcess?.kill('SIGKILL'), 3000);
  }
});

// ── Security: block navigation to external URLs ────────────
app.on('web-contents-created', (event, contents) => {
  contents.on('will-navigate', (event, url) => {
    const parsedUrl = new URL(url);
    const allowedHosts = ['127.0.0.1', 'localhost'];
    if (!allowedHosts.includes(parsedUrl.hostname)) {
      log.warn(`Blocked navigation to: ${url}`);
      event.preventDefault();
    }
  });
});