const { app, BrowserWindow, ipcMain, shell } = require("electron");
const log = require("electron-log");
const { isAllowedAppUrl, isExternalUrl } = require("./src/security-policy");
const { getAppRoot: resolveAppRoot, getResourcePath: resolveResourcePath, logPath } = require("./src/main/paths");
const pidStore = require("./src/main/backend-pid-store");
const { BACKEND_HOST } = require("./src/main/ports");
const { getPythonPath: resolvePythonPath } = require("./src/main/python-resolver");
const { createBackendLifecycle } = require("./src/main/backend-lifecycle");
const { readPersistedSettings, writePersistedSettings } = require("./src/main/settings-store");
const { createSplashWindow: buildSplashWindow, createMainWindow: buildMainWindow } = require("./src/main/windows");

log.transports.file.level = "info";
log.info("DepthLens Pro starting...");

const isDev = !app.isPackaged || process.env.NODE_ENV === "development";
const REQUESTED_BACKEND_PORT = Number.parseInt(process.env.DEPTHLENS_BACKEND_PORT || "8765", 10);
let BACKEND_PORT = REQUESTED_BACKEND_PORT;
let backendUrl = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
let mainWindow = null;
let splashWindow = null;
const state = { backendProcess: null, backendReady: false, backendOwnedByElectron: false, backendPid: null, backendMetadata: null, shutdownInProgress: false };
let isQuitting = false;

function getAppRoot() { return resolveAppRoot({ app, dirname: __dirname }); }
function getResourcePath(...parts) { return resolveResourcePath({ app, dirname: __dirname }, ...parts); }
function getPythonPath() { return resolvePythonPath({ root: getAppRoot(), isDev, log }); }
function getState() { return state; }
function setState(patch) { Object.assign(state, patch); }

const backendLifecycle = createBackendLifecycle({
  app,
  log,
  getAppRoot,
  getResourcePath,
  getPythonPath,
  pidStore,
  BACKEND_HOST,
  getBackendPort: () => BACKEND_PORT,
  setBackendPort: (port) => { BACKEND_PORT = port; },
  getBackendUrl: () => backendUrl,
  setBackendUrl: (url) => { backendUrl = url; },
  getState,
  setState,
  logPath: () => logPath(log),
});

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

function isSupportedArchitecture() {
  return ["darwin:arm64", "win32:arm64", "win32:x64", "linux:arm64", "linux:x64"].includes(`${process.platform}:${process.arch}`);
}

function showUnsupportedArchitectureDialog() {
  const { dialog } = require("electron");
  const message = "DepthLens Pro supports macOS Apple Silicon arm64, Windows arm64/x64, and Linux arm64/x64. macOS x64 and universal builds are not supported.";
  log.error("UNSUPPORTED_ARCH_BLOCKED", { platform: process.platform, arch: process.arch });
  dialog.showMessageBoxSync({ type: "error", title: "DepthLens Pro — Unsupported Architecture", message, detail: `${message}\n\nDetected platform: ${process.platform}\nDetected architecture: ${process.arch}`, buttons: ["Quit"] });
}

function createSplashWindow() { splashWindow = buildSplashWindow({ BrowserWindow, dirname: __dirname }); }
function createMainWindow() {
  mainWindow = buildMainWindow({ BrowserWindow, shell, log, dirname: __dirname, frontendPath: getResourcePath("frontend", "index.html"), isDev, isExternalUrl, getSplashWindow: () => splashWindow });
  mainWindow.on("closed", () => { mainWindow = null; });
}

ipcMain.handle("get-backend-url", () => backendUrl);
ipcMain.handle("get-app-version", () => app.getVersion());
ipcMain.handle("get-platform", () => process.platform);
ipcMain.handle("get-backend-live-path", () => "/live");
ipcMain.handle("settings:load", () => readPersistedSettings(app, log));
ipcMain.handle("settings:save", (_event, payload) => writePersistedSettings(app, payload));
ipcMain.handle("show-save-dialog", async (_event, options) => { const { dialog } = require("electron"); return dialog.showSaveDialog(mainWindow, options); });
ipcMain.handle("show-open-dialog", async (_event, options) => { const { dialog } = require("electron"); return dialog.showOpenDialog(mainWindow, options); });

if (singleInstanceLock) app.whenReady().then(async () => {
  log.info(`App ready — isDev: ${isDev}, platform: ${process.platform}, arch: ${process.arch}`);
  if (!isSupportedArchitecture()) { showUnsupportedArchitectureDialog(); app.quit(); return; }
  log.info("SUPPORTED_ARCH_CHECK_PASSED", { platform: process.platform, arch: process.arch });
  createSplashWindow();
  try {
    await backendLifecycle.startBackend();
    log.info(`Backend ready for renderer at ${backendUrl}`);
    createMainWindow();
  } catch (err) {
    log.error(`Failed to start backend: ${err.message}`);
    const { dialog } = require("electron");
    if (splashWindow && !splashWindow.isDestroyed()) splashWindow.destroy();
    const choice = dialog.showMessageBoxSync({ type: "error", title: "DepthLens Pro — Backend Error", message: "Could not start the inference engine.", detail: `${err.message}\n\nThe app will still open but inference will be unavailable.`, buttons: ["Open Anyway", "Quit"], defaultId: 0, cancelId: 1 });
    if (choice === 1) { setState({ backendOwnedByElectron: false }); app.quit(); return; }
    createMainWindow();
  }
  app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createMainWindow(); });
});

app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });
app.on("before-quit", (event) => {
  if (state.backendProcess && state.backendOwnedByElectron && !isQuitting) {
    event.preventDefault();
    isQuitting = true;
    backendLifecycle.shutdownOwnedBackend().finally(() => {
      if (!app.isQuittingDepthLens) {
        app.isQuittingDepthLens = true;
        app.quit();
      }
    });
  }
});
app.on("web-contents-created", (_event, contents) => {
  contents.on("will-navigate", (event, url) => {
    const frontendPath = getResourcePath("frontend", "index.html");
    if (!isAllowedAppUrl(url, { backendHost: BACKEND_HOST, backendPort: BACKEND_PORT, frontendPath })) {
      log.warn(`Blocked navigation to: ${url}`);
      event.preventDefault();
    }
  });
});
