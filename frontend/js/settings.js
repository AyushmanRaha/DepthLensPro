"use strict";

/**
 * DepthLens Pro — script.js v5.0
 * Changes from v4.1:
 *  - applyTheme() dispatches "depthlens-theme-changed" for welcome-anim.js
 *  - enterWorkspace() updated: no more brand-svg reference, just fade+migrate
 *  - migrateThemeToggle() updated: works with new .revealed class system
 * All inference, gallery, compare, lightbox, metrics logic is IDENTICAL to v4.1.
 */
"use strict";

// ══════════════════════════════════════════════════════════════
// THEME SYSTEM
// ══════════════════════════════════════════════════════════════
const THEME_KEY = "depthlens_theme_v1";

function getSavedTheme() {
  try { return localStorage.getItem(THEME_KEY) || "dark"; } catch { return "dark"; }
}
function saveTheme(t) {
  try { localStorage.setItem(THEME_KEY, t); } catch {}
}
function applyTheme(theme, animated = true) {
  const html = document.documentElement;
  if (!animated) html.style.transition = "none";
  html.setAttribute("data-theme", theme);
  if (!animated) requestAnimationFrame(() => { html.style.transition = ""; });
  // Notify welcome animation (bg lines + logo recolor)
  document.dispatchEvent(new CustomEvent("depthlens-theme-changed", { detail: theme }));
}

// Apply theme immediately before paint to avoid flash
applyTheme(getSavedTheme(), false);

// ══════════════════════════════════════════════════════════════
// CONFIG & STATE
// ══════════════════════════════════════════════════════════════
const DEFAULT_API_BASE_URL = "http://127.0.0.1:8765";
let API = null;
let apiResolved = false;
let apiResolutionPromise = null;
let backendOnline = false;
let inferenceReady = false;
let readinessDetails = null;
let runningInElectron = false;

const SETTINGS_KEY = "depthlens_settings_v1";
const LAST_TAB_KEY = "depthlens_last_tab_v1";
const DEFAULT_SETTINGS = {
  palette: "depthlens", accentIntensity: 1, glassIntensity: "normal", compactUi: false, highContrast: false, largerText: false, backgroundGrid: true,
  motion: "full", panelPhysics: true, animatedBorders: true, reduceMotionOverride: false,
  backendUrl: "", autoCheckEngine: true, diagnosticsRefresh: "normal", refreshDiagnosticsOnOpen: true, showAdvancedDiagnostics: false, warnOnDegradedEngine: true, allowFallbackEngine: true, autoRetryEngineChecks: true,
  useInferenceCache: true, showCacheBadges: true, maxInteractiveDim: "default",
  warnLargeFiles: true, autoClearQueueAfterBatch: false, autoClearResultsOnClose: false, stripFilenamesFromExports: false, pauseWebcamWhenHidden: true, stopCameraOnTabSwitch: false,
  toastLevel: "all", toastDuration: "normal", dedupeWarnings: true, endpointTimingLogs: false, verboseConsoleLogs: false,
  rememberLastTab: true, skipWelcome: false, closePanelsOnOutsideClick: true, navDragSensitivity: "normal"
};
let settings = loadSettings();
hydratePersistedSettings();

function loadSettings() {
  try {
    const raw = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    return { ...DEFAULT_SETTINGS, ...(raw && typeof raw === "object" ? raw : {}) };
  } catch { return { ...DEFAULT_SETTINGS }; }
}
function persistedSettingsPayload() {
  return {
    selectedModel: selModel?.() || "MiDaS_small",
    selectedDevice: selDevice?.() || "auto",
    selectedColormap: selCmap?.() || "inferno",
    targetFps: getWebcamTargetFps?.() || 2,
    webcamFrameMaxDimension: getWebcamMaxDim?.() || 384,
    smoothingPreference: getWebcamSmoothingAlpha?.() || 0.25,
    recentBenchmarkSettings: { model: el.benchmarkModel?.value || "MiDaS_small", device: el.benchmarkDevice?.value || "auto", iterations: 3 },
    onnxPreference: settings.onnxPreference || "auto",
    onnxStatus: state?.engineStatus?.onnx?.status || "unknown",
    ui: { palette: settings.palette, motion: settings.motion, compactUi: settings.compactUi },
    privacy: { pauseWebcamWhenHidden: settings.pauseWebcamWhenHidden, stopCameraOnTabSwitch: settings.stopCameraOnTabSwitch }
  };
}
async function hydratePersistedSettings() {
  if (!window.electronAPI?.loadSettings) return;
  try {
    const persisted = await window.electronAPI.loadSettings();
    if (!persisted || typeof persisted !== "object") return;
    if (persisted.selectedModel) document.querySelector(`input[name="model"][value="${CSS.escape(persisted.selectedModel)}"]`)?.click?.();
    if (persisted.selectedDevice) document.querySelector(`input[name="device"][value="${CSS.escape(persisted.selectedDevice)}"]`)?.click?.();
    if (persisted.selectedColormap) document.querySelector(`input[name="colormap"][value="${CSS.escape(persisted.selectedColormap)}"]`)?.click?.();
    if (el.webcamTargetFps && persisted.targetFps) el.webcamTargetFps.value = String(persisted.targetFps);
    if (el.webcamMaxDim && persisted.webcamFrameMaxDimension) el.webcamMaxDim.value = String(persisted.webcamFrameMaxDimension);
    if (el.webcamSmoothing && persisted.smoothingPreference !== undefined) el.webcamSmoothing.value = String(persisted.smoothingPreference);
    if (persisted.ui && typeof persisted.ui === "object") settings = { ...settings, ...persisted.ui };
    if (persisted.privacy && typeof persisted.privacy === "object") settings = { ...settings, ...persisted.privacy };
    applySettings({ persist: true, notify: false });
    updateWebcamTelemetry?.();
  } catch (err) { console.warn(`[DepthLens] persisted settings unavailable: ${err.message || err}`); }
}
function saveSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch {}
  if (window.electronAPI?.saveSettings) window.electronAPI.saveSettings(persistedSettingsPayload()).catch(err => console.warn(`[DepthLens] settings save failed: ${err.message || err}`));
}
function applyPalette() {
  document.documentElement.setAttribute("data-palette", settings.palette || "depthlens");
  document.documentElement.style.setProperty("--accent-intensity", String(settings.accentIntensity || 1));
  document.documentElement.style.setProperty("--accent-glow", `color-mix(in srgb, var(--accent) ${Math.round((settings.accentIntensity || 1) * 16)}%, transparent)`);
}
function applyUiDensity() {
  const html = document.documentElement;
  html.setAttribute("data-glass", settings.glassIntensity || "normal");
  html.setAttribute("data-compact-ui", String(Boolean(settings.compactUi)));
  html.setAttribute("data-high-contrast", String(Boolean(settings.highContrast)));
  html.setAttribute("data-larger-text", String(Boolean(settings.largerText)));
  html.setAttribute("data-background-grid", String(settings.backgroundGrid !== false));
}
function applyMotionSettings() {
  const html = document.documentElement;
  html.setAttribute("data-motion", settings.motion || "full");
  html.setAttribute("data-reduce-motion", String(Boolean(settings.reduceMotionOverride) || settings.motion !== "full"));
  html.setAttribute("data-animated-borders", String(settings.animatedBorders !== false));
}
function applySettings({ persist = false, notify = true } = {}) {
  applyPalette(); applyUiDensity(); applyMotionSettings();
  if (settings.backendUrl && !runningInElectron) { try { localStorage.setItem("depthlens_api_url", settings.backendUrl); } catch {} }
  if (persist) saveSettings();
  if (notify) document.dispatchEvent(new CustomEvent("depthlens-theme-changed", { detail: currentTheme }));
  updateSettingsControlState();
  try { updateChartTheme(); } catch {}
}
function getInteractiveMaxDim() {
  const n = Number(settings.maxInteractiveDim);
  return Number.isFinite(n) && n > 0 ? n : null;
}
function diagnosticsIntervalMs() { return ({ slow: 120000, fast: 30000, normal: 60000 })[settings.diagnosticsRefresh] || 60000; }
function toastDurationMs(dur) {
  if (typeof dur === "number") return dur;
  return ({ short: 2200, long: 6500, normal: 3500 })[settings.toastDuration] || 3500;
}
function toastAllowed(type) {
  if (settings.toastLevel === "silent") return false;
  if (settings.toastLevel === "errors" || settings.toastLevel === "errorsOnly") return type === "error";
  if (settings.toastLevel === "warnings") return type === "warning" || type === "error";
  return true;
}

function normalizeApiBaseUrl(url) {
  return String(url || DEFAULT_API_BASE_URL).trim().replace(/\/+$/, "");
}

async function resolveApiBaseUrl() {
  if (apiResolved && API) return API;
  if (apiResolutionPromise) return apiResolutionPromise;

  apiResolutionPromise = (async () => {
    let resolved = null;
    runningInElectron = Boolean(window.electronAPI?.getBackendUrl);
    if (runningInElectron) {
      resolved = await window.electronAPI.getBackendUrl();
      const platform = await window.electronAPI.getPlatform?.() || window.electronAPI.platform;
      if (platform === "darwin") document.body.classList.add("macos");
    } else {
      resolved = window.DEPTHLENS_API_URL;
      try { resolved = resolved || localStorage.getItem("depthlens_api_url"); } catch {}
    }

    API = normalizeApiBaseUrl(resolved || DEFAULT_API_BASE_URL);
    apiResolved = true;
    console.log(`[DepthLens] Resolved backend URL: ${API}`);
    if (!runningInElectron) console.warn("[DepthLens] Electron bridge unavailable; running in browser/file mode. Start the backend manually or set DEPTHLENS_API_URL/localStorage.");
    return API;
  })();

  try {
    return await apiResolutionPromise;
  } catch (err) {
    apiResolutionPromise = null;
    apiResolved = false;
    throw err;
  }
}

function buildApiUrl(base, path) {
  const route = String(path || "");
  return `${base}${route.startsWith("/") ? route : `/${route}`}`;
}

function apiUrl(path) {
  return buildApiUrl(API || DEFAULT_API_BASE_URL, path);
}

function timeoutSignal(ms) {
  if (window.AbortSignal?.timeout) return AbortSignal.timeout(ms);
  const controller = new AbortController();
  setTimeout(() => controller.abort(), ms);
  return controller.signal;
}

function anySignal(signals) {
  const liveSignals = signals.filter(Boolean);
  if (window.AbortSignal?.any) return AbortSignal.any(liveSignals);
  const controller = new AbortController();
  const abort = () => controller.abort();
  for (const signal of liveSignals) {
    if (signal.aborted) { abort(); break; }
    signal.addEventListener("abort", abort, { once: true });
  }
  return controller.signal;
}

function requestSignal(signal, timeoutMs) {
  const timeout = timeoutSignal(timeoutMs);
  return signal ? anySignal([signal, timeout]) : timeout;
}

async function apiFetch(path, options = {}) {
  const base = await resolveApiBaseUrl();
  const url = buildApiUrl(base, path);
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const err = await res.clone().json();
        const rawDetail = err.detail || err.error || detail;
        detail = typeof rawDetail === "object"
          ? (rawDetail.message || rawDetail.error_code || JSON.stringify(rawDetail))
          : rawDetail;
      } catch {}
      throw new Error(`${detail} (${url})`);
    }
    return res;
  } catch (err) {
    const baseMsg = err.message || String(err);
    const msg = err.name === "AbortError"
      ? `Request cancelled or timed out (${url})`
      : (baseMsg.includes(url) ? baseMsg : `${baseMsg} (${url})`);
    console.error(`[DepthLens] API request failed: ${msg}`);
    const wrapped = new Error(msg);
    wrapped.name = err.name || "ApiError";
    wrapped.cause = err;
    throw wrapped;
  }
}

const state = {
  files: [],
  results: [],
  session: {
    total: 0, cached: 0, errors: 0,
    latencies: [], totalInferenceMs: 0,
  },
  cacheMetrics: null,
  lb: { current: null },
  abort: null,
  compareAbort: null,
  healthAbort: null,
  benchmarkAbort: null,
  performanceView: "benchmark",
  observability: { snapshot: null, loading: false, lastUpdatedAt: null, abort: null, chart: null },
  pollTimers: {},
  pollMode: null,
  pollControllers: {},
  pollInFlight: {},
  pollEpoch: 0,
  lastToast: { message: "", at: 0 },
  compareFile: null,
  devices: {},
  primaryDevice: "cpu",
  deviceFilter: "all",
  deviceListSignature: "",
  deviceFilterSignature: "",
  selectedDevice: null,
  timing: { workspace: {}, compare: {} },
  compareView: { metricKey: "latency_ms", open: true, results: [] },
  initializingBackend: true,
  engineStatus: {
  cls: "connecting",
  label: "Starting depth engine",
  sub: DEFAULT_API_BASE_URL,
  deviceText: "",
  live: null,
  readiness: null,
  health: null,
  lastUpdatedAt: null,
  panelOpen: false,
  },
  gtMode: false,
  gtFile: null,
  webcam: {
    stream: null,
    running: false,
    paused: false,
    hiddenPaused: false,
    starting: false,
    inFlight: false,
    abort: null,
    loopTimer: null,
    lastLoopStartedAt: 0,
    lastDepthBase64: null,
    lastDepthDataUrl: "",
    lastFrameBlobUrl: null,
    processed: 0,
    skipped: 0,
    errors: 0,
    consecutiveErrors: 0,
    latencies: [],
    e2eLatencies: [],
    startedAt: null,
    lastResult: null,
    lastCacheMetricsAt: 0,
    smoothingAlpha: 0.25,
    previousDepthImageData: null,
  },
  reconstruct: {
    file: null,
    filePreview: "",
    abort: null,
    running: false,
    result: null,
    progressTimer: null,
    capture: {
      stream: null,
      running: false,
      starting: false,
      detecting: false,
      detectTimer: null,
      detectAbort: null,
      lastDetections: [],
      lastCapturedFile: null,
      previousFocus: null,
    },
    viewer: {
      gl: null,
      ctx2d: null,
      mode: "none",
      points: [],
      normalized: [],
      rotationX: -0.4,
      rotationY: 0.65,
      zoom: 1.8,
      pointSize: 2,
      dragging: false,
      lastX: 0,
      lastY: 0,
      animationId: null,
      autoRotate: false,
    },
  },
  experiment: { name: "DepthLens validation run", results: [], startedAt: null },
};
