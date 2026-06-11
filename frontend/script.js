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
  label: "Starting engine…",
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

// ══════════════════════════════════════════════════════════════
// DOM BINDINGS
// ══════════════════════════════════════════════════════════════
const $ = (s, ctx = document) => ctx.querySelector(s);
const $$ = (s, ctx = document) => [...ctx.querySelectorAll(s)];

const el = {
  welcomeScreen:   $("#welcomeScreen"),
  getStartedBtn:   $("#getStartedBtn"),
  appShell:        $("#appShell"),

  // Theme
  themeToggleLanding: $("#themeToggleLanding"),
  themeToggleHeader:  $("#themeToggleHeader"),
  themeToggleBtn:     $("#themeToggleBtn"),

  // Header
  headerNavShell:          $("#headerNavShell"),
  engineStatusHost:        $("#engineStatusHost"),
  engineStatusButton:      $("#engineStatusButton"),
  engineStatusPanel:       $("#engineStatusPanel"),
  engineStatusClose:       $("#engineStatusClose"),
  engineStatusRefresh:     $("#engineStatusRefresh"),
  engineStatusStateValue:  $("#engineStatusStateValue"),
  engineStatusBackendUrl:  $("#engineStatusBackendUrl"),
  engineStatusReadiness:   $("#engineStatusReadinessValue"),
  engineStatusDiagnostics: $("#engineStatusDiagnosticsValue"),
  engineStatusRuntime:     $("#engineStatusRuntimeValue"),
  engineStatusCache:       $("#engineStatusCacheValue"),
  engineStatusLoadedModels:$("#engineStatusLoadedModels"),
  engineStatusSystem:      $("#engineStatusSystemValue"),
  engineStatusModules:     $("#engineStatusModules"),
  engineStatusUpdatedAt:   $("#engineStatusUpdatedAt"),

  statusDot:    $("#statusDot"),
  statusLabel:  $("#statusLabel"),
  statusSub:    $("#statusSub"),
  deviceBadge:  $("#deviceBadge"),
  navBtns:      $$(".nav-btn"),
  panels:       $$(".panel"), 

  // Device
  deviceInfoBanner: $("#deviceInfoBanner"),
  deviceTypeToggle: $("#deviceTypeToggle"),
  deviceSelector:   $("#deviceSelector"),

  // Upload queue
  dropZone:  $("#dropZone"),
  fileInput: $("#fileInput"),
  fileQueue: $("#fileQueue"),
  clearBtn:  $("#clearBtn"),
  cancelBtn: $("#cancelBtn"),
  runBtn:    $("#runBtn"),
  gtToggle: $("#gtToggle"),
  gtUpload: $("#gtUpload"),
  gtFileInput: $("#gtFileInput"),
  gtFileName: $("#gtFileName"),
  clearGtBtn: $("#clearGtBtn"),

  // Progress
  progressBlock:      $("#progressBlock"),
  progressFill:       $("#progressFill"),
  progressPct:        $("#progressPct"),
  progressBar:        $("#progressBar"),
  progressStatusText: $("#progressStatusText"),
  progressEta:        $("#progressEta"),
  progressCurrentFile:$("#progressCurrentFile"),
  progressItemCount:  $("#progressItemCount"),

  // Results
  resultsCard:     $("#resultsCard"),
  gallery:         $("#gallery"),
  downloadAllBtn:  $("#downloadAllBtn"),
  clearResultsBtn: $("#clearResultsBtn"),

  // Metrics
  metricTotal:      $("#metricTotal"),
  metricAvgLatency: $("#metricAvgLatency"),
  metricCached:     $("#metricCached"),
  metricErrors:     $("#metricErrors"),
  metricMinLat:     $("#metricMinLat"),
  metricMaxLat:     $("#metricMaxLat"),
  metricThroughput: $("#metricThroughput"),
  metricTotalTime:  $("#metricTotalTime"),

  // Compare
  compareDropZone:       $("#compareDropZone"),
  compareFileInput:      $("#compareFileInput"),
  compareFileName:       $("#compareFileName"),
  compareRunBtn:         $("#compareRunBtn"),
  compareCancelBtn:      $("#compareCancelBtn"),
  compareCmap:           $("#compareCmap"),
  compareDevice:         $("#compareDevice"),
  compareResults:        $("#compareResults"),
  compareProgressBlock:  $("#compareProgressBlock"),
  compareProgressFill:   $("#compareProgressFill"),
  compareProgressPct:    $("#compareProgressPct"),
  compareProgressText:   $("#compareProgressText"),
  compareProgressEta:    $("#compareProgressEta"),
  compareChartCard:      $("#compareChartCard"),
  compareChartBody:      $("#compareChartBody"),
  compareChartToggle:    $("#compareChartToggle"),
  compareMetricSelect:   $("#compareMetricSelect"),
  compareMetricGrid:     $("#compareMetricGrid"),

  // Benchmark
  benchmarkModel:      $("#benchmarkModel"),
  benchmarkDevice:     $("#benchmarkDevice"),
  benchmarkRunBtn:     $("#benchmarkRunBtn"),
  benchTorchLatency:   $("#benchTorchLatency"),
  benchOnnxLatency:    $("#benchOnnxLatency"),
  benchSpeedup:        $("#benchSpeedup"),
  benchThroughput:     $("#benchThroughput"),
  benchMemory:         $("#benchMemory"),
  benchProvider:       $("#benchProvider"),
  benchStatus:         $("#benchStatus"),

  // Experiments
  experimentName:         $("#experimentName"),
  experimentRunBtn:       $("#experimentRunBtn"),
  experimentExportJsonBtn:$("#experimentExportJsonBtn"),
  experimentExportCsvBtn: $("#experimentExportCsvBtn"),
  experimentStatus:       $("#experimentStatus"),
  experimentCount:        $("#experimentCount"),
  experimentAvgLatency:   $("#experimentAvgLatency"),
  experimentBestAbsRel:   $("#experimentBestAbsRel"),
  experimentStatusMetric: $("#experimentStatusMetric"),
  experimentTableBody:    $("#experimentTableBody"),
  experimentPreviews:     $("#experimentPreviews"),

  // Lightbox
  lightboxBackdrop: $("#lightboxBackdrop"),
  lightboxClose:    $("#lightboxClose"),
  lbOrigImg:        $("#lbOrigImg"),
  lbDepthImg:       $("#lbDepthImg"),
  lightboxMetrics:  $("#lightboxMetrics"),
  lbSlider:         $("#lbSlider"),
  lbRangeValue:     $("#lbRangeValue"),
  lbTags:           $("#lbTags"),
  lbDlDepth:        $("#lbDlDepth"),
  lbDlGray:         $("#lbDlGray"),

  // Webcam
  webcamStatusPill:       $("#webcamStatusPill"),
  webcamStartBtn:         $("#webcamStartBtn"),
  webcamStopBtn:          $("#webcamStopBtn"),
  webcamPauseBtn:         $("#webcamPauseBtn"),
  webcamCaptureBtn:       $("#webcamCaptureBtn"),
  webcamTargetFps:        $("#webcamTargetFps"),
  webcamMaxDim:           $("#webcamMaxDim"),
  webcamSmoothing:        $("#webcamSmoothing"),
  webcamVideo:            $("#webcamVideo"),
  webcamCaptureCanvas:    $("#webcamCaptureCanvas"),
  webcamDepthImg:         $("#webcamDepthImg"),
  webcamDepthPlaceholder: $("#webcamDepthPlaceholder"),
  webcamCameraState:      $("#webcamCameraState"),
  webcamInferenceState:   $("#webcamInferenceState"),
  webcamTargetFpsMetric:  $("#webcamTargetFpsMetric"),
  webcamEffectiveFps:     $("#webcamEffectiveFps"),
  webcamBackendLatency:   $("#webcamBackendLatency"),
  webcamEndToEndLatency:  $("#webcamEndToEndLatency"),
  webcamProcessedFrames:  $("#webcamProcessedFrames"),
  webcamSkippedFrames:    $("#webcamSkippedFrames"),
  webcamErrorCount:       $("#webcamErrorCount"),
  webcamActiveModel:      $("#webcamActiveModel"),
  webcamActiveDevice:     $("#webcamActiveDevice"),
  webcamActiveColormap:   $("#webcamActiveColormap"),
  webcamLog:              $("#webcamLog"),


  // 3D Reconstruction
  reconstructStatusPill:       $("#reconstructStatusPill"),
  reconstructDropZone:         $("#reconstructDropZone"),
  reconstructFileInput:        $("#reconstructFileInput"),
  reconstructInputPreview:     $("#reconstructInputPreview"),
  reconstructFileName:         $("#reconstructFileName"),
  reconstructBrowseBtn:        $("#reconstructBrowseBtn"),
  reconstructUseLatestBtn:     $("#reconstructUseLatestBtn"),
  reconstructClearBtn:         $("#reconstructClearBtn"),
  reconstructFormat:           $("#reconstructFormat"),
  reconstructMaxDim:           $("#reconstructMaxDim"),
  reconstructMaxPoints:        $("#reconstructMaxPoints"),
  reconstructPreviewPoints:    $("#reconstructPreviewPoints"),
  reconstructSampling:         $("#reconstructSampling"),
  reconstructCoordinateSystem: $("#reconstructCoordinateSystem"),
  reconstructIncludeRgb:       $("#reconstructIncludeRgb"),
  reconstructFocalScale:       $("#reconstructFocalScale"),
  reconstructDepthScale:       $("#reconstructDepthScale"),
  reconstructNearPct:          $("#reconstructNearPct"),
  reconstructFarPct:           $("#reconstructFarPct"),
  reconstructRunBtn:           $("#reconstructRunBtn"),
  reconstructCancelBtn:        $("#reconstructCancelBtn"),
  reconstructProgressBlock:    $("#reconstructProgressBlock"),
  reconstructProgressFill:     $("#reconstructProgressFill"),
  reconstructProgressPct:      $("#reconstructProgressPct"),
  reconstructProgressText:     $("#reconstructProgressText"),
  reconstructProgressEta:      $("#reconstructProgressEta"),
  pointCloudCanvas:            $("#pointCloudCanvas"),
  pointCloudPlaceholder:       $("#pointCloudPlaceholder"),
  pointCloudModeMessage:       $("#pointCloudModeMessage"),
  pointCloudResetViewBtn:      $("#pointCloudResetViewBtn"),
  pointCloudPointSize:         $("#pointCloudPointSize"),
  pointCloudAutoRotate:        $("#pointCloudAutoRotate"),
  reconstructPointCount:       $("#reconstructPointCount"),
  reconstructPreviewCount:     $("#reconstructPreviewCount"),
  reconstructArtifactSize:     $("#reconstructArtifactSize"),
  reconstructLatency:          $("#reconstructLatency"),
  reconstructModel:            $("#reconstructModel"),
  reconstructDevice:           $("#reconstructDevice"),
  reconstructEngine:           $("#reconstructEngine"),
  reconstructResolution:       $("#reconstructResolution"),
  reconstructBounds:           $("#reconstructBounds"),
  reconstructWarnings:         $("#reconstructWarnings"),
  reconstructDepthCached:      $("#reconstructDepthCached"),
  reconstructFormatLabel:      $("#reconstructFormatLabel"),
  reconstructDepthPreview:     $("#reconstructDepthPreview"),
  reconstructDepthEmpty:       $("#reconstructDepthEmpty"),
  reconstructDownloadDepthBtn: $("#reconstructDownloadDepthBtn"),
  reconstructDownloadBtn:      $("#reconstructDownloadBtn"),
  reconstructCopyMetaBtn:      $("#reconstructCopyMetaBtn"),
  reconstructDownloadMetaBtn:  $("#reconstructDownloadMetaBtn"),

  toastContainer: $("#toastContainer"),
  bgCanvas:       $("#bgCanvas"),
};

// ══════════════════════════════════════════════════════════════
// THEME TOGGLE LOGIC
// ══════════════════════════════════════════════════════════════
let currentTheme = getSavedTheme();

function toggleTheme() {
  currentTheme = currentTheme === "dark" ? "light" : "dark";
  applyTheme(currentTheme, true);
  saveTheme(currentTheme);
  updateChartTheme();
}

el.themeToggleBtn?.addEventListener("click", toggleTheme);

// ══════════════════════════════════════════════════════════════
// LANDING → WORKSPACE TRANSITION
// ══════════════════════════════════════════════════════════════
function enterWorkspace() {
  if (!el.welcomeScreen || !el.appShell) return;
  el.getStartedBtn?.setAttribute("disabled", "true");

  // Migrate theme toggle from landing corner → header
  migrateThemeToggle(() => {
    el.welcomeScreen.classList.add("is-exiting");
    setTimeout(() => {
      el.welcomeScreen.hidden = true;
      el.appShell.classList.add("ready");
    }, 650);
  });
}

function migrateThemeToggle(onComplete) {
  const landingWrap = el.themeToggleLanding;
  const headerWrap  = el.themeToggleHeader;
  const btn         = el.themeToggleBtn;
  if (!landingWrap || !headerWrap || !btn) { onComplete(); return; }

  const srcRect = btn.getBoundingClientRect();

  // Move button into header slot
  headerWrap.appendChild(btn);
  const dstRect = btn.getBoundingClientRect();

  const dx = srcRect.left - dstRect.left;
  const dy = srcRect.top  - dstRect.top;

  btn.style.transform  = `translate(${dx}px, ${dy}px) scale(1.1)`;
  btn.style.transition = "none";
  btn.style.opacity    = "1";

  // Hide the landing placeholder visually
  landingWrap.classList.add("is-migrating");

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      btn.style.transition = "transform 0.62s cubic-bezier(0.22,1,0.36,1), opacity 0.4s ease";
      btn.style.transform  = "translate(0,0) scale(1)";
      btn.classList.add("visible");
    });
  });

  setTimeout(() => {
    btn.style.transform  = "";
    btn.style.transition = "";
    onComplete();
  }, 480);
}

el.getStartedBtn?.addEventListener("click", enterWorkspace);

// ══════════════════════════════════════════════════════════════
// PERSISTENCE
// ══════════════════════════════════════════════════════════════
const PREFS_KEY = "depthlens_prefs_v3";

function savePrefs() {
  const prefs = {
    model:   $('input[name="model"]:checked')?.value   || "MiDaS_small",
    colormap:$('input[name="colormap"]:checked')?.value|| "inferno",
    device:  $('input[name="device"]:checked')?.value  || "auto",
  };
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch {}
}

function loadPrefs() {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (!raw) return;
    const p = JSON.parse(raw);
    const modelEl = $(`input[name="model"][value="${p.model}"]`);
    if (modelEl) modelEl.checked = true;
    const cmapEl = $(`input[name="colormap"][value="${p.colormap}"]`);
    if (cmapEl) cmapEl.checked = true;
    window._savedDevice = p.device;
  } catch {}
}

document.addEventListener("change", (e) => {
  if (["model","colormap","device"].includes(e.target.name)) {
    savePrefs();
    updateWebcamTelemetry();
  }
});

// ══════════════════════════════════════════════════════════════
// BACKGROUND CANVAS (workspace)
// ══════════════════════════════════════════════════════════════
(function bgCanvas() {
  const cv = el.bgCanvas, ctx = cv.getContext("2d");
  let W, H, pts = [];
  const N = 50;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function mkP() {
    return {
      x: Math.random()*W, y: Math.random()*H,
      vx:(Math.random()-0.5)*0.28, vy:(Math.random()-0.5)*0.28,
      r: Math.random()*1.3+0.3, a: Math.random(),
    };
  }
  function resize() {
    const dpr = Math.min(window.devicePixelRatio||1,2);
    W = window.innerWidth; H = window.innerHeight;
    cv.width = Math.floor(W*dpr); cv.height = Math.floor(H*dpr);
    cv.style.width = `${W}px`; cv.style.height = `${H}px`;
    ctx.setTransform(dpr,0,0,dpr,0,0);
  }
  function reset() { resize(); pts = Array.from({length:N},mkP); }

  function draw() {
    ctx.clearRect(0,0,W,H);
    ctx.strokeStyle = "rgba(0,200,255,.055)"; ctx.lineWidth=1;
    for (let x=0;x<W;x+=64) { ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke(); }
    for (let y=0;y<H;y+=64) { ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke(); }
    for (let i=0;i<pts.length;i++) {
      const p=pts[i];
      p.x=(p.x+p.vx+W)%W; p.y=(p.y+p.vy+H)%H;
      for (let j=i+1;j<pts.length;j++) {
        const q=pts[j], d=Math.hypot(p.x-q.x,p.y-q.y);
        if (d<130) {
          ctx.strokeStyle=`rgba(0,200,255,${0.11*(1-d/130)})`;
          ctx.lineWidth=0.55; ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke();
        }
      }
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(0,200,255,${p.a*0.55})`; ctx.fill();
    }
    if (!reduce) requestAnimationFrame(draw);
  }
  window.addEventListener("resize",resize);
  reset(); draw();
})();

// ══════════════════════════════════════════════════════════════
// CHARTS
// ══════════════════════════════════════════════════════════════
let latencyChart, compareChart, benchmarkChart;

const COMPARE_METRICS = [
  { key:"latency_ms",    label:"Latency (ms)",        source:"root",    better:"lower",  fmt:(v)=>`${Math.round(v)} ms` },
  { key:"ssim",          label:"SSIM",                source:"metrics", better:"higher", fmt:(v)=>Number(v).toFixed(3) },
  { key:"silog",         label:"SILog",               source:"metrics", better:"lower",  fmt:(v)=>Number(v).toFixed(2) },
  { key:"psnr",          label:"PSNR (dB)",           source:"metrics", better:"higher", fmt:(v)=>`${Number(v).toFixed(2)} dB` },
  { key:"gradient_mean", label:"Gradient Mean",       source:"metrics", better:"higher", fmt:(v)=>Number(v).toFixed(3) },
  { key:"edge_density",  label:"Edge Density",        source:"metrics", better:"higher", fmt:(v)=>`${(Number(v)*100).toFixed(1)}%` },
  { key:"entropy",       label:"Entropy (bits)",      source:"metrics", better:"higher", fmt:(v)=>Number(v).toFixed(2) },
  { key:"dynamic_range", label:"Dynamic Range (bits)",source:"metrics", better:"higher", fmt:(v)=>`${Number(v).toFixed(2)} bits` },
];

function chartColors() {
  const isDark = document.documentElement.getAttribute("data-theme") !== "light";
  return {
    grid:    isDark ? "rgba(0,200,255,.07)"  : "rgba(0,100,180,.08)",
    tick:    isDark ? "#3a5a72"               : "#5a7a99",
    line:    isDark ? "#00c8ff"               : "#0070cc",
    fill:    isDark ? "rgba(0,200,255,.07)"   : "rgba(0,112,204,.06)",
    bar:     isDark ? "rgba(0,200,255,.55)"   : "rgba(0,112,204,.55)",
    barBrd:  isDark ? "#00c8ff"               : "#0070cc",
    tooltip: isDark ? "#101e2e"               : "#ffffff",
    ttBrd:   isDark ? "#00c8ff"               : "#0070cc",
    ttTitle: isDark ? "#7faac8"               : "#2d4a66",
    ttBody:  isDark ? "#dff0ff"               : "#0d1f33",
  };
}

function initLatencyChart() {
  const c = chartColors();
  latencyChart = new Chart($("#latencyChart").getContext("2d"), {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Inference ms",
        data: [],
        borderColor: c.line,
        backgroundColor: c.fill,
        borderWidth: 1.5, pointRadius: 2, tension: 0.4, fill: true,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 280 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: c.tooltip, borderColor: c.ttBrd, borderWidth: 1,
          titleColor: c.ttTitle, bodyColor: c.ttBody,
          callbacks: { label: (ctx) => `${ctx.raw} ms` },
        },
      },
      scales: {
        x: { display: false },
        y: {
          display: true,
          grid: { color: c.grid },
          ticks: { color: c.tick, font: { family:"JetBrains Mono", size:9 }, maxTicksLimit:4 },
        },
      },
    },
  });
}

function applyChartPalette(chart, c) {
  if (!chart) return;
  const tooltip = chart.options?.plugins?.tooltip;
  if (tooltip) {
    tooltip.backgroundColor = c.tooltip;
    tooltip.borderColor = c.ttBrd;
    tooltip.titleColor = c.ttTitle;
    tooltip.bodyColor = c.ttBody;
  }
  const legend = chart.options?.plugins?.legend;
  if (legend?.labels) legend.labels.color = c.ttTitle;
  if (chart.options?.scales?.x) {
    if (chart.options.scales.x.grid) chart.options.scales.x.grid.color = c.grid;
    if (chart.options.scales.x.ticks) chart.options.scales.x.ticks.color = c.ttTitle;
  }
  if (chart.options?.scales?.y) {
    if (chart.options.scales.y.grid) chart.options.scales.y.grid.color = c.grid;
    if (chart.options.scales.y.ticks) chart.options.scales.y.ticks.color = c.tick;
  }
}

function updateChartTheme() {
  const c = chartColors();
  if (latencyChart) {
    latencyChart.data.datasets[0].borderColor = c.line;
    latencyChart.data.datasets[0].backgroundColor = c.fill;
    applyChartPalette(latencyChart, c);
    latencyChart.update("none");
  }

  if (compareChart && state.compareView.results.length) {
    renderCompareChart(state.compareView.results, state.compareView.metricKey, { preserveInstance: true });
  }
  if (benchmarkChart) {
    applyChartPalette(benchmarkChart, c);
    const ds = benchmarkChart.data?.datasets?.[0];
    if (ds) {
      ds.backgroundColor = ds.data.map(v => v === null ? "rgba(127,140,153,.45)" : c.bar);
      ds.borderColor = ds.data.map(v => v === null ? "#5e6f81" : c.barBrd);
    }
    benchmarkChart.update("none");
  }
}

function pushLatency(ms) {
  const d = state.session.latencies.slice(-20);
  latencyChart.data.labels = d.map((_,i) => i+1);
  latencyChart.data.datasets[0].data = d;
  latencyChart.update("none");
}

function compareMetricValue(result, spec) {
  return spec.source === "root" ? result?.[spec.key] : result?.metrics?.[spec.key];
}

function renderCompareSummary(results) {
  el.compareMetricGrid.innerHTML = "";
  const summaryMetrics = COMPARE_METRICS.filter(m =>
    ["latency_ms","ssim","silog","psnr","edge_density"].includes(m.key)
  );
  summaryMetrics.forEach(spec => {
    const valid = results
      .map(r => ({ model: r.model, value: Number(compareMetricValue(r,spec)) }))
      .filter(v => Number.isFinite(v.value));
    const cell = document.createElement("div");
    cell.className = "compare-metric-cell";
    if (!valid.length) {
      cell.innerHTML = `<span class="compare-metric-label">${esc(spec.label)}</span><span class="compare-metric-values">Unavailable</span>`;
      el.compareMetricGrid.appendChild(cell); return;
    }
    const sorted = [...valid].sort((a,b) => spec.better === "higher" ? b.value-a.value : a.value-b.value);
    const [best, runnerUp] = sorted;
    const cleanName = s => esc(s).replace("MiDaS_","").replace("DPT_","DPT ");
    cell.innerHTML = `
      <span class="compare-metric-label">${esc(spec.label)}</span>
      <div class="compare-metric-values">
        <span>Best: ${cleanName(best.model)} <strong>${esc(spec.fmt(best.value))}</strong></span>
        ${runnerUp ? `<span>2nd: ${cleanName(runnerUp.model)} <strong>${esc(spec.fmt(runnerUp.value))}</strong></span>` : ""}
      </div>`;
    el.compareMetricGrid.appendChild(cell);
  });
}

function renderCompareChart(results, metricKey = state.compareView.metricKey, { preserveInstance = false } = {}) {
  const metric = COMPARE_METRICS.find(m => m.key === metricKey) || COMPARE_METRICS[0];
  const values = results.map(r => {
    const v = Number(compareMetricValue(r, metric));
    return Number.isFinite(v) ? v : null;
  });
  const labels = results.map(r => escText(r.model).replace("MiDaS_","").replace("DPT_","DPT "));
  el.compareChartCard.hidden = false;
  const c = chartColors();
  const ctx = $("#compareChart")?.getContext("2d");
  if (!ctx) return;
  if (compareChart && preserveInstance) {
    compareChart.data.labels = labels;
    compareChart.data.datasets[0].label = metric.label;
    compareChart.data.datasets[0].data = values;
    compareChart.data.datasets[0].backgroundColor = values.map(v => v === null ? "rgba(127,140,153,.45)" : c.bar);
    compareChart.data.datasets[0].borderColor = values.map(v => v === null ? "#5e6f81" : c.barBrd);
    compareChart.options.scales.y.ticks.callback = v => metric.fmt(v);
    compareChart.options.plugins.tooltip.callbacks.label = ctx => ctx.raw === null ? "Not available" : metric.fmt(ctx.raw);
    applyChartPalette(compareChart, c);
    compareChart.update("none");
    return;
  }
  if (compareChart) {
    const oldChart = compareChart;
    compareChart = null;
    oldChart.destroy();
  }
  compareChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: metric.label,
        data: values,
        backgroundColor: values.map(v => v === null ? "rgba(127,140,153,.45)" : c.bar),
        borderColor:     values.map(v => v === null ? "#5e6f81" : c.barBrd),
        borderWidth: 1.5, borderRadius: 4,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 380 },
      plugins: {
        legend: { labels: { color: c.ttTitle, font: { family:"JetBrains Mono", size:10 } } },
        tooltip: {
          backgroundColor: c.tooltip, borderColor: c.ttBrd, borderWidth:1,
          titleColor: c.ttTitle, bodyColor: c.ttBody,
          callbacks: { label: ctx => ctx.raw === null ? "Not available" : metric.fmt(ctx.raw) },
        },
      },
      scales: {
        x: {
          ticks: { color: c.ttTitle, font: { family:"Rajdhani", size:12, weight:"600" } },
          grid: { color: c.grid },
        },
        y: {
          ticks: { color: c.tick, font: { family:"JetBrains Mono", size:9 }, callback: v => metric.fmt(v) },
          grid: { color: c.grid },
        },
      },
    },
  });
}

function initCompareControls() {
  el.compareMetricSelect.innerHTML = COMPARE_METRICS.map(m =>
    `<option value="${m.key}">${m.label}</option>`
  ).join("");
  el.compareMetricSelect.value = state.compareView.metricKey;

  el.compareMetricSelect.addEventListener("change", () => {
    state.compareView.metricKey = el.compareMetricSelect.value;
    if (state.compareView.results.length)
      renderCompareChart(state.compareView.results, state.compareView.metricKey);
  });

  el.compareChartToggle.addEventListener("click", () => {
    state.compareView.open = !state.compareView.open;
    el.compareChartBody.hidden = !state.compareView.open;
    el.compareChartToggle.textContent = state.compareView.open ? "Hide Graph" : "Show Graph";
    el.compareChartToggle.setAttribute("aria-expanded", String(state.compareView.open));
  });
}

function logEndpointTiming(label, started, ok, extra = "") {
  const ms = Math.round(performance.now() - started);
  console.debug(`[DepthLens] ${label} ${ok ? "ok" : "failed"} in ${ms}ms${extra ? ` · ${extra}` : ""}`);
}

// ══════════════════════════════════════════════════════════════
// SERVER HEALTH & DEVICE LIST
// ══════════════════════════════════════════════════════════════
function setStatus(cls, label, sub, deviceText) {
  const safeCls = ["online", "offline", "connecting"].includes(cls) ? cls : "offline";

  state.engineStatus.cls = safeCls;
  state.engineStatus.label = label || "Depth Engine";
  state.engineStatus.sub = sub || "";
  state.engineStatus.deviceText = deviceText || "";
  state.engineStatus.lastUpdatedAt = new Date();

  if (el.statusDot) el.statusDot.className = `status-dot ${safeCls}`;

  if (el.engineStatusButton) {
    el.engineStatusButton.classList.remove("online", "offline", "connecting");
    el.engineStatusButton.classList.add(safeCls);
    el.engineStatusButton.setAttribute(
      "aria-label",
      `${state.engineStatus.label}. ${state.engineStatus.sub || ""}`.trim()
    );
    el.engineStatusButton.title = `${state.engineStatus.label}${state.engineStatus.sub ? ` — ${state.engineStatus.sub}` : ""}`;
  }

  if (el.engineStatusPanel) {
    el.engineStatusPanel.classList.remove("online", "offline", "connecting");
    el.engineStatusPanel.classList.add(safeCls);
  }

  if (el.statusLabel) el.statusLabel.textContent = state.engineStatus.label;
  if (el.statusSub) el.statusSub.textContent = state.engineStatus.sub;

  if (el.deviceBadge) {
    if (deviceText) {
      el.deviceBadge.textContent = deviceText;
      el.deviceBadge.hidden = false;
    } else {
      el.deviceBadge.hidden = true;
    }
  }

  renderEngineStatusPanel();
}

function formatEngineTime(date) {
  if (!date) return "Not updated yet";
  try {
    return `Updated ${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
  } catch {
    return "Updated just now";
  }
}

function formatEngineBytes(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return "n/a";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function engineStatusStateText() {
  const { cls, live } = state.engineStatus;
  if (cls === "online") {
    if (live?.busy) return "Backend live · busy";
    if (inferenceReady) return "Backend live · inference ready";
    return "Backend live · checking inference";
  }
  if (cls === "connecting") return "Connecting to local backend";
  return "Backend unavailable";
}

function engineReadinessText() {
  const readiness = state.engineStatus.readiness || readinessDetails || state.engineStatus.health?.readiness;

  if (inferenceReady) return "Ready for inference";
  if (readiness?.required) return readinessSummary(readiness);
  if (state.engineStatus.cls === "offline") return "Readiness unavailable";
  return "Checking runtime dependencies…";
}

function engineDiagnosticsText() {
  const health = state.engineStatus.health;
  if (!health) {
    return backendOnline ? "Waiting for /health" : "Diagnostics unavailable";
  }

  const status = health.diagnostics_status || health.status || "unknown";
  const accel = health.acceleration_ok === false ? "accelerator degraded" : "accelerator OK";
  return `${status} · ${accel}`;
}

function engineRuntimeText() {
  const health = state.engineStatus.health;
  const readiness = state.engineStatus.readiness || readinessDetails || health?.readiness;
  const torchVersion =
    health?.torch_version ||
    readiness?.required?.torch?.version ||
    readiness?.torch_runtime?.torch_version;

  if (torchVersion) return `PyTorch ${torchVersion}`;
  if (readiness?.torch_runtime?.python_version) {
    return `Python ${readiness.torch_runtime.python_version}`;
  }
  return "Runtime not reported yet";
}

function engineCacheText() {
  const cache = state.cacheMetrics;
  if (!cache) return "No metrics yet";

  return `${cache.backend} · ${cache.keyspaceSize} entries · ${cache.totalHits} hits`;
}

function engineLoadedModelsText() {
  const loaded = state.engineStatus.health?.loaded_models;
  if (!Array.isArray(loaded) || !loaded.length) return "None reported";
  return loaded.join(", ");
}

function engineSystemText() {
  const health = state.engineStatus.health;
  const system = health?.system;
  const telemetry = health?.telemetry;

  if (!system && !telemetry) return "Waiting for diagnostics…";

  const parts = [];
  if (system?.os) parts.push(system.os);
  if (system?.machine) parts.push(system.machine);
  if (system?.cpu) parts.push(`CPU: ${system.cpu}`);

  const memory = telemetry?.memory;
  if (memory?.pressure_percent !== undefined && memory?.pressure_percent !== null) {
    parts.push(`Memory pressure: ${memory.pressure_percent}%`);
  }

  const disk = telemetry?.disk;
  if (disk?.usage_percent !== undefined && disk?.usage_percent !== null) {
    parts.push(`Disk usage: ${disk.usage_percent}%`);
  }

  return parts.length ? parts.join(" · ") : "Diagnostics returned no system summary";
}

function engineModulePillsHtml() {
  const readiness = state.engineStatus.readiness || readinessDetails || state.engineStatus.health?.readiness;
  const required = readiness?.required || {};

  const entries = Object.entries(required);
  if (!entries.length) {
    return `<span class="engine-module-pill muted">Waiting for readiness check</span>`;
  }

  return entries.map(([name, info]) => {
    const ok = Boolean(info?.available);
    const label = `${name}: ${ok ? "ok" : (info?.status || "missing")}`;
    return `<span class="engine-module-pill ${ok ? "" : "bad"}">${esc(label)}</span>`;
  }).join("");
}

function renderEngineStatusPanel() {
  if (!el.engineStatusPanel) return;

  const api = API || DEFAULT_API_BASE_URL;

  if (el.engineStatusStateValue) {
    el.engineStatusStateValue.textContent = engineStatusStateText();
  }

  if (el.engineStatusBackendUrl) {
    el.engineStatusBackendUrl.textContent = api;
  }

  if (el.engineStatusReadiness) {
    el.engineStatusReadiness.textContent = engineReadinessText();
  }

  if (el.engineStatusDiagnostics) {
    el.engineStatusDiagnostics.textContent = engineDiagnosticsText();
  }

  if (el.engineStatusRuntime) {
    el.engineStatusRuntime.textContent = engineRuntimeText();
  }

  if (el.engineStatusCache) {
    el.engineStatusCache.textContent = engineCacheText();
  }

  if (el.engineStatusLoadedModels) {
    el.engineStatusLoadedModels.textContent = engineLoadedModelsText();
  }

  if (el.engineStatusSystem) {
    el.engineStatusSystem.textContent = engineSystemText();
  }

  if (el.engineStatusModules) {
    el.engineStatusModules.innerHTML = engineModulePillsHtml();
  }

  if (el.engineStatusUpdatedAt) {
    el.engineStatusUpdatedAt.textContent = formatEngineTime(state.engineStatus.lastUpdatedAt);
  }
}

function setEngineStatusPanelOpen(open) {
  state.engineStatus.panelOpen = Boolean(open);

  if (!el.engineStatusPanel || !el.engineStatusButton) return;

  if (open) {
    renderEngineStatusPanel();
    el.engineStatusPanel.hidden = false;
    el.engineStatusButton.setAttribute("aria-expanded", "true");

    requestAnimationFrame(() => {
      el.engineStatusPanel.classList.add("open");
    });
  } else {
    el.engineStatusButton.setAttribute("aria-expanded", "false");
    el.engineStatusPanel.classList.remove("open");

    window.setTimeout(() => {
      if (!state.engineStatus.panelOpen) {
        el.engineStatusPanel.hidden = true;
      }
    }, 260);
  }
}

function applyPointerPhysics(target, event, options = {}) {
  if (!target) return;

  const rect = target.getBoundingClientRect();
  const px = (event.clientX - rect.left) / rect.width;
  const py = (event.clientY - rect.top) / rect.height;

  const strength = options.strength || 8;
  const tiltY = (px - 0.5) * strength;
  const tiltX = (0.5 - py) * strength;

  if (options.panel) {
    target.style.setProperty("--panel-tilt-x", `${tiltX.toFixed(2)}deg`);
    target.style.setProperty("--panel-tilt-y", `${tiltY.toFixed(2)}deg`);
    target.style.setProperty("--panel-mx", `${(px * 100).toFixed(1)}%`);
    target.style.setProperty("--panel-my", `${(py * 100).toFixed(1)}%`);
  } else {
    target.style.setProperty("--orb-tilt-x", `${tiltX.toFixed(2)}deg`);
    target.style.setProperty("--orb-tilt-y", `${tiltY.toFixed(2)}deg`);
  }
}

function resetPointerPhysics(target, options = {}) {
  if (!target) return;

  if (options.panel) {
    target.style.setProperty("--panel-tilt-x", "0deg");
    target.style.setProperty("--panel-tilt-y", "0deg");
    target.style.setProperty("--panel-mx", "50%");
    target.style.setProperty("--panel-my", "20%");
  } else {
    target.style.setProperty("--orb-tilt-x", "0deg");
    target.style.setProperty("--orb-tilt-y", "0deg");
  }
}

function initEngineStatusPanel() {
  el.engineStatusButton?.addEventListener("click", (event) => {
    event.stopPropagation();
    setEngineStatusPanelOpen(!state.engineStatus.panelOpen);
  });

  el.engineStatusClose?.addEventListener("click", () => {
    setEngineStatusPanelOpen(false);
  });

  el.engineStatusRefresh?.addEventListener("click", async () => {
    el.engineStatusRefresh.disabled = true;
    el.engineStatusRefresh.textContent = "Checking…";

    try {
      const live = await checkLive({ quiet: true });
      if (live) {
        await checkReadiness({ quiet: true });
        await checkDiagnostics({ quiet: true });
        await loadCacheMetrics();
      }
      renderEngineStatusPanel();
    } finally {
      el.engineStatusRefresh.disabled = false;
      el.engineStatusRefresh.textContent = "Refresh";
    }
  });

  document.addEventListener("click", (event) => {
    if (!state.engineStatus.panelOpen) return;
    if (el.engineStatusHost?.contains(event.target)) return;
    if (el.engineStatusPanel?.contains(event.target)) return;
    setEngineStatusPanelOpen(false);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.engineStatus.panelOpen) {
      setEngineStatusPanelOpen(false);
    }
  });

  window.addEventListener("resize", () => {
    if (state.engineStatus.panelOpen) renderEngineStatusPanel();
  });

  el.engineStatusButton?.addEventListener("mousemove", (event) => {
    applyPointerPhysics(el.engineStatusButton, event, { strength: 10 });
  });

  el.engineStatusButton?.addEventListener("mouseleave", () => {
    resetPointerPhysics(el.engineStatusButton);
  });

  el.engineStatusPanel?.addEventListener("mousemove", (event) => {
    applyPointerPhysics(el.engineStatusPanel, event, { strength: 2.2, panel: true });
  });

  el.engineStatusPanel?.addEventListener("mouseleave", () => {
    resetPointerPhysics(el.engineStatusPanel, { panel: true });
  });
}

function classifyKinds(info = {}) {
  const set = new Set(
    (Array.isArray(info.compute_classes) ? info.compute_classes : [info.type||"cpu"])
    .map(v => String(v).toLowerCase())
  );
  if (set.has("cuda")||set.has("mps")||set.has("xpu")) set.add("gpu");
  if (set.has("ane")) set.add("npu");
  return [...set];
}

function matchesDeviceFilter(kinds, filter) {
  if (filter==="all") return true;
  if (filter==="gpu") return kinds.includes("gpu")||kinds.includes("cuda")||kinds.includes("mps")||kinds.includes("xpu");
  if (filter==="npu") return kinds.includes("npu")||kinds.includes("ane");
  return kinds.includes(filter);
}

function accelerationSummary(data = {}) {
  const checks = data.acceleration_checks || {};
  const read = (k,label) => {
    const c = checks[k];
    if (!c?.available) return `${label}: n/a`;
    return `${label}: ${c.operational ? "ok" : "fail"}`;
  };
  return [read("cuda","CUDA"), read("mps","MPS"), read("xpu","XPU")].join(" · ");
}

function normalizeCacheMetrics(raw = {}) {
  return {
    totalHits: Number(raw.total_hits ?? raw.hits ?? 0),
    cacheMisses: Number(raw.cache_misses ?? raw.misses ?? 0),
    keyspaceSize: Number(raw.keyspace_size ?? raw.cache_entries ?? 0),
    backend: String(raw.backend || (raw.redis_available ? "redis" : "memory")),
    redisAvailable: Boolean(raw.redis_available),
    ttlSeconds: Number(raw.ttl_seconds ?? 3600),
  };
}

function applyCacheMetrics(raw) {
  if (!raw) return;
  state.cacheMetrics = normalizeCacheMetrics(raw);
  updateMetrics();
}

function engineReady() {
  return backendOnline && inferenceReady;
}

function readinessSummary(payload) {
  const failed = Object.entries(payload?.required || {})
    .filter(([, item]) => !item?.available)
    .map(([name, item]) => `${name}: ${item?.error || item?.status || "unavailable"}`);
  return failed.length ? failed.join(" · ") : "Inference runtime is ready";
}

async function checkReadiness({ quiet = true } = {}) {
  if (!backendOnline) { inferenceReady = false; return false; }
  const started = performance.now();
  try {
    const res = await apiFetch("/ready", { signal: timeoutSignal(5000) });
    const data = await res.json();
    state.engineStatus.readiness = data;
    readinessDetails = data;
    inferenceReady = Boolean(data.inference_ready);
    if (inferenceReady) {
      setStatus("online", "Depth Engine: Online", `Backend ready · ${API || DEFAULT_API_BASE_URL}`, deviceBadge(state.devices, state.primaryDevice));
    } else {
      const summary = readinessSummary(data);
      setStatus("offline", "Depth Engine: Degraded", summary);
      const bannerText = el.deviceInfoBanner?.querySelector("span:last-child");
      if (bannerText) bannerText.textContent = `Inference runtime degraded: ${summary}`;
      if (!quiet) toastOnce(`Inference runtime is not ready: ${summary}`, "error", 8000);
    }
    logEndpointTiming("/ready", started, inferenceReady);
    syncQueueControls();
    syncReconstructControls();
    return inferenceReady;
  } catch (err) {
    logEndpointTiming("/ready", started, false, err.message);
    readinessDetails = null;
    inferenceReady = false;
    state.engineStatus.readiness = null;
    setStatus("offline", "Depth Engine: Degraded", `Readiness check failed · ${err.message}`);
    if (!quiet) toastOnce(`Readiness check failed: ${err.message}`, "error", 6000);
    syncQueueControls();
    return false;
  }
}

async function loadCacheMetrics({ signal } = {}) {
  if (!backendOnline || document.hidden) return false;
  const started = performance.now();
  try {
    const res = await apiFetch("/cache/metrics", {
      signal: requestSignal(signal, 8000)
    });
    applyCacheMetrics(await res.json());
    logEndpointTiming("/cache/metrics", started, true);
    return true;
  } catch (err) {
    logEndpointTiming("/cache/metrics", started, false, err.message);
    return false;
  }
}

function deviceBadge(devs, primary) {
  const info = devs?.[primary];
  if (info?.type==="cuda") return `GPU: ${info.name} (${info.memory_gb} GB)`;
  if (info?.type==="mps") return `Apple ${info.chip || "Silicon"} · Metal`;
  if (info?.type==="xpu") return `NPU/GPU: ${info.name||primary}`;
  return "CPU";
}

async function checkLive({ quiet = false, signal } = {}) {
  if (!quiet) setStatus("connecting","Connecting…", API || DEFAULT_API_BASE_URL);
  const started = performance.now();
  try {
    const res = await apiFetch("/live", { signal: requestSignal(signal, 2500) });
    const data = await res.json();
    state.engineStatus.live = data;
    backendOnline = data.status === "ok";
    if (!backendOnline) throw new Error("Unexpected /live response");
    if (data.busy) {
      setStatus("online", "Depth Engine: Busy", "Benchmark/model load running · liveness OK", deviceBadge(state.devices, state.primaryDevice));
    } else {
      setStatus("online","Depth Engine: Detected",`Backend live · checking inference readiness`, deviceBadge(state.devices, state.primaryDevice));
    }
    logEndpointTiming("/live", started, true);
    syncQueueControls();
    syncReconstructControls();
    return true;
  } catch (err) {
    logEndpointTiming("/live", started, false, err.message);
    backendOnline = false;
    inferenceReady = false;
    state.engineStatus.live = null;
    state.engineStatus.readiness = null;
    state.engineStatus.health = null;
    setStatus("offline","Depth Engine: Offline",`No /live response from ${API || DEFAULT_API_BASE_URL}`);
    const bannerText = el.deviceInfoBanner?.querySelector("span:last-child");
    if (bannerText) bannerText.textContent = `Depth engine unavailable at ${API || DEFAULT_API_BASE_URL}`;
    if (!quiet) toastOnce(`Backend liveness check failed: ${err.message}`, "error", 6000);
    syncQueueControls();
    return false;
  }
}

async function checkDiagnostics({ quiet = true, signal } = {}) {
  if (!backendOnline) return false;
  const started = performance.now();
  const controller = signal ? null : new AbortController();
  if (!signal) state.healthAbort = controller;
  const requestSignal = signal || controller.signal;
  try {
    const res = await apiFetch("/health", { signal: anySignal([requestSignal, timeoutSignal(6000)]) });
    const data = await res.json();
    state.engineStatus.health = data;
    applyCacheMetrics(data.cache_metrics);
    const devs = data.devices || state.devices || {};
    const primary = data.primary_device || state.primaryDevice || "cpu";
    state.devices = devs; state.primaryDevice = primary;
    const badge = deviceBadge(devs, primary);
    const accelText = accelerationSummary(data);
    setStatus("online","Depth Engine: Online",`PyTorch ${data.torch_version} · ${primary} · diagnostics ${data.status || "ok"} · ${accelText}`,badge);
    await loadDevices(devs, primary);
    updateDeviceInfoBanner(`${badge} · ${accelText}`);
    logEndpointTiming("/health", started, true, data.status || "ok");
    if (data.status === "degraded" && !quiet) toastOnce("Diagnostics degraded; inference remains available.","warning");
    return true;
  } catch (err) {
    logEndpointTiming("/health", started, false, err.message);
    if (backendOnline) {
      setStatus("online","Depth Engine: Online",`Diagnostics degraded · ${err.message}`, deviceBadge(state.devices, state.primaryDevice));
      if (!quiet) toastOnce(`Diagnostics check failed: ${err.message}`, "warning", 5000);
    }
    state.engineStatus.health = {
    status: "degraded",
    diagnostics_status: "degraded",
    error: err.message,
    };
    renderEngineStatusPanel();
    return false;
  }
}

async function checkHealth() {
  const live = await checkLive();
  if (live) {
    await checkReadiness({ quiet: true });
    await checkDiagnostics({ quiet: true });
  }
  return live && inferenceReady;
}

function autoDeviceLabel(devs, primary) {
  const preferred = devs?.[primary];
  if (!preferred) return "Best available compute";
  if (preferred.type==="cuda") return `Prefers GPU · ${preferred.name||primary}`;
  if (preferred.type==="mps")  return `Prefers Apple acceleration · ${preferred.name||primary}`;
  if ((preferred.compute_classes||[]).includes("npu")) return `Prefers NPU · ${preferred.name||primary}`;
  return `Prefers CPU · ${preferred.name||primary}`;
}

function deviceListSignature(devs = {}, filters = null, primary = state.primaryDevice) {
  return JSON.stringify({
    primary,
    devices: Object.entries(devs).map(([key, info]) => [
      key,
      info?.name || "",
      info?.type || "",
      info?.chip || "",
      info?.memory_gb || "",
      info?.hardware_name || "",
      info?.compute_classes || [],
    ]).sort((a,b) => a[0].localeCompare(b[0])),
    filters,
  });
}

function cssEscapeValue(value) {
  if (window.CSS?.escape) return CSS.escape(String(value));
  return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}

function currentSelectedDevice(fallback = "auto") {
  const domValue = $('input[name="device"]:checked', el.deviceSelector)?.value;
  const candidate = domValue || state.selectedDevice || window._savedDevice || fallback || "auto";
  return candidate === "auto" || state.devices?.[candidate] ? candidate : "auto";
}

async function loadDevices(devs, primaryFromHealth = null) {
  if (!devs || !Object.keys(devs).length) {
    try {
      const r = await apiFetch("/devices",{signal:timeoutSignal(3000)});
      const payload = await r.json();
      devs = payload.devices || {};
      primaryFromHealth = payload.primary_device || primaryFromHealth;
    } catch (err) { console.error(`[DepthLens] Device load failed: ${err.message}`); return; }
  }
  const keys = Object.keys(devs);
  if (!keys.length) return;
  const previousSelection = currentSelectedDevice("auto");
  state.devices = devs;
  const primary = primaryFromHealth && devs[primaryFromHealth] ? primaryFromHealth
    : devs.mps ? "mps" : keys.includes("cuda:0") ? "cuda:0" : keys.includes("xpu:0") ? "xpu:0" : keys[0];
  state.primaryDevice = primary;
  const savedRaw = previousSelection || window._savedDevice || "auto";
  const saved = savedRaw === "auto" || devs[savedRaw] ? savedRaw : "auto";
  state.selectedDevice = saved;

  const withKinds = Object.entries(devs).map(([key,info]) => ({key,info,kinds:classifyKinds(info)}));
  const kindsPresent = {
    cpu: withKinds.some(d=>d.kinds.includes("cpu")),
    gpu: withKinds.some(d=>d.kinds.includes("gpu")||d.kinds.includes("cuda")||d.kinds.includes("mps")),
    npu: withKinds.some(d=>d.kinds.includes("npu")||d.kinds.includes("ane")),
  };
  const filters = [
    {id:"all",label:"All"},
    ...(kindsPresent.cpu?[{id:"cpu",label:"CPU"}]:[]),
    ...(kindsPresent.gpu?[{id:"gpu",label:"GPU"}]:[]),
    ...(kindsPresent.npu?[{id:"npu",label:"NPU"}]:[]),
  ];
  if (!filters.find(f=>f.id===state.deviceFilter)) state.deviceFilter = "all";

  const filterSig = JSON.stringify({ filters: filters.map(f=>f.id), devices: deviceListSignature(devs, null, primary) });
  if (state.deviceFilterSignature !== filterSig) {
    state.deviceFilterSignature = filterSig;
    el.deviceTypeToggle.hidden = filters.length <= 1;
    el.deviceTypeToggle.innerHTML = "";
    filters.forEach(f => {
      const b = document.createElement("button");
      b.type="button"; b.className=`device-filter-btn ${state.deviceFilter===f.id?"active":""}`;
      b.textContent = f.label;
      b.addEventListener("click",() => {
        state.deviceFilter = f.id;
        state.selectedDevice = currentSelectedDevice(saved);
        renderDeviceSelector(withKinds,primary,state.selectedDevice,devs, { force: true });
        [...el.deviceTypeToggle.children].forEach(ch=>ch.classList.remove("active"));
        b.classList.add("active");
      });
      el.deviceTypeToggle.appendChild(b);
    });
  } else {
    [...el.deviceTypeToggle.children].forEach(ch=>ch.classList.toggle("active", ch.textContent.toLowerCase() === state.deviceFilter));
  }

  renderDeviceSelector(withKinds, primary, saved, devs);

  [el.compareDevice, el.benchmarkDevice].filter(Boolean).forEach(select => {
    const selectCurrent = select.value || saved;
    const selectSig = deviceListSignature(devs, null, primary);
    if (select.dataset.deviceSignature === selectSig) {
      if (selectCurrent === "auto" || devs[selectCurrent]) select.value = selectCurrent;
      return;
    }
    select.dataset.deviceSignature = selectSig;
    select.innerHTML = "";
    const autoOpt = document.createElement("option");
    autoOpt.value="auto"; autoOpt.textContent=`Auto (${autoDeviceLabel(devs,primary)})`;
    select.appendChild(autoOpt);
    Object.entries(devs).forEach(([key,info]) => {
      const opt = document.createElement("option");
      opt.value=key; opt.textContent=info.name||key;
      select.appendChild(opt);
    });
    select.value = (selectCurrent === "auto" || devs[selectCurrent]) ? selectCurrent : saved;
  });
}

function renderDeviceSelector(deviceEntries, primary, saved, devs, { force = false } = {}) {
  const visibleKeys = deviceEntries.filter(({kinds}) => matchesDeviceFilter(kinds, state.deviceFilter)).map(({key}) => key);
  const sig = deviceListSignature(devs, { filter: state.deviceFilter, visibleKeys }, primary);
  const selected = saved === "auto" || devs?.[saved] ? saved : "auto";
  if (!force && state.deviceListSignature === sig) {
    const input = $(`input[name="device"][value="${cssEscapeValue(selected)}"]`, el.deviceSelector);
    if (input && !input.checked) {
      input.checked = true;
      $$(".device-opt", el.deviceSelector).forEach(l => l.classList.toggle("selected", l.dataset.device === selected));
    }
    return;
  }
  state.deviceListSignature = sig;
  el.deviceSelector.innerHTML = "";
  const autoChecked = selected==="auto" || !selected;
  const auto = document.createElement("label");
  auto.className = "device-opt"+(autoChecked?" selected":"");
  auto.dataset.device = "auto";
  auto.innerHTML = `
    <input type="radio" name="device" value="auto" ${autoChecked?"checked":""} />
    <div class="device-opt-inner">
      <span class="device-opt-icon">◎</span>
      <div>
        <span class="device-opt-name">Auto Select</span>
        <span class="device-opt-sub">${esc(autoDeviceLabel(devs,primary))}</span>
      </div>
    </div>`;
  el.deviceSelector.appendChild(auto);

  deviceEntries.forEach(({key,info,kinds}) => {
    if (!matchesDeviceFilter(kinds, state.deviceFilter)) return;
    const isCuda=info.type==="cuda", isMps=info.type==="mps", isXpu=info.type==="xpu";
    const icon = isCuda?"▦":isMps?"⬢":isXpu?"⬣":"◻";
    const classLabel=[kinds.includes("gpu")?"GPU":null,kinds.includes("npu")?"NPU":null,kinds.includes("cpu")?"CPU":null].filter(Boolean).join("/");
    const sub = isCuda ? `CUDA · ${info.memory_gb} GB VRAM · ${classLabel}`
      : isMps  ? `Apple ${info.chip||"Silicon"} · Metal + Neural Engine · ${classLabel}`
      : isXpu  ? `XPU backend · ${classLabel}`
      : `${info.hardware_name||"System processor"} · ${classLabel||"CPU"}`;
    const checked = selected===key;
    const lbl = document.createElement("label");
    lbl.className = "device-opt"+(checked?" selected":"");
    lbl.dataset.device = key;
    lbl.innerHTML = `
      <input type="radio" name="device" value="${esc(key)}" ${checked?"checked":""} />
      <div class="device-opt-inner">
        <span class="device-opt-icon">${esc(icon)}</span>
        <div>
          <span class="device-opt-name">${esc(info.name||key)}</span>
          <span class="device-opt-sub">${esc(sub)}</span>
        </div>
      </div>`;
    el.deviceSelector.appendChild(lbl);
  });

  $$(".device-opt input", el.deviceSelector).forEach(inp => {
    inp.addEventListener("change", () => {
      state.selectedDevice = inp.value;
      $$(".device-opt", el.deviceSelector).forEach(l => l.classList.remove("selected"));
      inp.closest(".device-opt").classList.add("selected");
      savePrefs();
    });
  });
}

function updateDeviceInfoBanner(text) {
  const bannerText = el.deviceInfoBanner?.querySelector("span:last-child");
  if (bannerText) bannerText.textContent = `Detected: ${text}. Choose compute target below.`;
}


// ══════════════════════════════════════════════════════════════
// 3D RECONSTRUCTION
// ══════════════════════════════════════════════════════════════
function reconstructSupported() {
  const canvas = document.createElement("canvas");
  return Boolean(window.File && window.FormData && window.fetch && canvas.getContext);
}

function pointCloudWebglAvailable() {
  const canvas = document.createElement("canvas");
  try { return Boolean(canvas.getContext("webgl") || canvas.getContext("experimental-webgl")); } catch { return false; }
}

function latestResultWithOriginal() {
  return [...(state.results || [])].reverse().find(r => r?.originalSrc);
}

function setReconstructStatus(label, stateName = "idle") {
  if (!el.reconstructStatusPill) return;
  el.reconstructStatusPill.textContent = label;
  el.reconstructStatusPill.dataset.state = stateName;
}

function syncReconstructControls() {
  const hasFile = Boolean(state.reconstruct.file);
  const hasResult = Boolean(state.reconstruct.result);
  const running = Boolean(state.reconstruct.running);
  const supported = reconstructSupported();
  if (el.reconstructRunBtn) el.reconstructRunBtn.disabled = !supported || !hasFile || running || !engineReady();
  if (el.reconstructCancelBtn) el.reconstructCancelBtn.hidden = !running;
  if (el.reconstructDownloadBtn) el.reconstructDownloadBtn.disabled = !hasResult || running;
  if (el.reconstructCopyMetaBtn) el.reconstructCopyMetaBtn.disabled = !hasResult;
  if (el.reconstructDownloadMetaBtn) el.reconstructDownloadMetaBtn.disabled = !hasResult;
  if (el.reconstructDownloadDepthBtn) el.reconstructDownloadDepthBtn.disabled = !hasResult || !state.reconstruct.result?.depth_map;
  if (el.reconstructUseLatestBtn) el.reconstructUseLatestBtn.disabled = !latestResultWithOriginal() || running;
  if (el.reconstructClearBtn) el.reconstructClearBtn.disabled = running && !hasFile && !hasResult;
  if (!supported) setReconstructStatus("Unsupported browser", "error");
}

function setReconstructFile(file, previewDataUrl = "") {
  if (!file) return;
  if (!file.type?.startsWith("image/")) { toast("Select an image file for reconstruction", "warning"); return; }
  if (file.size > 20 * 1024 * 1024) { toast("Reconstruction image exceeds 20 MB", "warning"); return; }
  state.reconstruct.file = file;
  state.reconstruct.result = null;
  state.reconstruct.filePreview = previewDataUrl || "";
  if (el.reconstructFileName) el.reconstructFileName.textContent = file.name || "Selected image";
  if (el.reconstructInputPreview) {
    el.reconstructInputPreview.hidden = !state.reconstruct.filePreview;
    if (state.reconstruct.filePreview) el.reconstructInputPreview.src = state.reconstruct.filePreview;
  }
  setReconstructStatus("Ready", "ready");
  resetReconstructionTelemetry();
  if (previewDataUrl) { syncReconstructControls(); return; }
  const reader = new FileReader();
  reader.onload = () => {
    state.reconstruct.filePreview = String(reader.result || "");
    if (el.reconstructInputPreview) {
      el.reconstructInputPreview.src = state.reconstruct.filePreview;
      el.reconstructInputPreview.hidden = !state.reconstruct.filePreview;
    }
  };
  reader.onerror = () => toast("Could not read reconstruction preview", "warning");
  reader.readAsDataURL(file);
  syncReconstructControls();
}

function clearReconstruct() {
  cancelReconstruction({ quiet: true });
  state.reconstruct.file = null;
  state.reconstruct.filePreview = "";
  state.reconstruct.result = null;
  if (el.reconstructFileInput) el.reconstructFileInput.value = "";
  if (el.reconstructFileName) el.reconstructFileName.textContent = "No image selected";
  if (el.reconstructInputPreview) { el.reconstructInputPreview.hidden = true; el.reconstructInputPreview.removeAttribute("src"); }
  if (el.reconstructDepthPreview) { el.reconstructDepthPreview.hidden = true; el.reconstructDepthPreview.removeAttribute("src"); }
  if (el.reconstructDepthEmpty) el.reconstructDepthEmpty.hidden = false;
  if (el.reconstructProgressBlock) el.reconstructProgressBlock.hidden = true;
  setReconstructStatus("Idle", "idle");
  clearPointCloudViewer();
  resetReconstructionTelemetry();
  syncReconstructControls();
}

function dataUrlToFile(dataUrl, filename) {
  const match = String(dataUrl || "").match(/^data:(image\/[\w.+-]+);base64,(.+)$/);
  if (!match) throw new Error("Latest result does not include a base64 image source");
  const mime = match[1];
  const binary = atob(match[2].replace(/\s/g, ""));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new File([bytes], filename || "latest-depthlens-source.jpg", { type: mime });
}

function useLatestReconstructionSource() {
  try {
    const latest = latestResultWithOriginal();
    if (!latest) { toast("Run a workspace inference first, or upload an image directly.", "warning"); return; }
    const file = dataUrlToFile(latest.originalSrc, latest.filename || "latest-depthlens-source.jpg");
    setReconstructFile(file, latest.originalSrc);
    toast("Latest workspace source loaded for 3D reconstruction", "success");
  } catch (err) {
    toast(`Could not use latest result: ${err.message}`, "error", 6000);
  }
}

function readNumberInput(node, fallback) {
  const value = Number(node?.value);
  return Number.isFinite(value) ? value : fallback;
}

function readReconstructOptions() {
  return {
    exportFormat: el.reconstructFormat?.value || "ply",
    maxDim: readNumberInput(el.reconstructMaxDim, 512),
    maxPoints: readNumberInput(el.reconstructMaxPoints, 120000),
    previewPoints: readNumberInput(el.reconstructPreviewPoints, 5000),
    focalScale: readNumberInput(el.reconstructFocalScale, 1.2),
    depthScale: readNumberInput(el.reconstructDepthScale, 1.0),
    nearPct: readNumberInput(el.reconstructNearPct, 2),
    farPct: readNumberInput(el.reconstructFarPct, 98),
    sampling: el.reconstructSampling?.value || "grid",
    includeRgb: Boolean(el.reconstructIncludeRgb?.checked),
    coordinateSystem: el.reconstructCoordinateSystem?.value || "y_up",
  };
}

function startReconstructProgress() {
  let pct = 2;
  const started = performance.now();
  if (el.reconstructProgressBlock) el.reconstructProgressBlock.hidden = false;
  updateReconstructProgress(pct, "Uploading image and options…", "Estimating…");
  clearInterval(state.reconstruct.progressTimer);
  state.reconstruct.progressTimer = setInterval(() => {
    const elapsed = performance.now() - started;
    const ceiling = elapsed < 15000 ? 70 : 95;
    pct = Math.min(ceiling, pct + Math.max(0.4, (ceiling - pct) * 0.045));
    updateReconstructProgress(pct, pct < 75 ? "Generating relative depth…" : "Building point-cloud artifact…", `${Math.round(elapsed / 1000)}s elapsed`);
  }, 450);
}

function stopReconstructProgress(finalPct = 100, text = "Complete") {
  clearInterval(state.reconstruct.progressTimer);
  state.reconstruct.progressTimer = null;
  updateReconstructProgress(finalPct, text, "");
}

function updateReconstructProgress(pct, text, eta) {
  const clamped = Math.max(0, Math.min(100, Math.round(pct)));
  if (el.reconstructProgressFill) el.reconstructProgressFill.style.width = `${clamped}%`;
  if (el.reconstructProgressPct) el.reconstructProgressPct.textContent = `${clamped}%`;
  if (el.reconstructProgressText) el.reconstructProgressText.textContent = text;
  if (el.reconstructProgressEta) el.reconstructProgressEta.textContent = eta || "";
}

async function ensureReconstructionBackendReady() {
  if (engineReady()) return true;
  const live = await checkLive({ quiet: true });
  if (!live) return false;
  return await checkReadiness({ quiet: true });
}

async function runReconstruction() {
  if (!state.reconstruct.file) { toast("Choose an image before generating a point cloud", "warning"); return; }
  if (!reconstructSupported()) { toast("This browser does not support the reconstruction upload APIs", "error"); return; }
  if (state.reconstruct.running) return;
  const ready = await ensureReconstructionBackendReady();
  if (!ready) { toast("Depth engine is not ready for reconstruction yet", "warning", 6000); syncReconstructControls(); return; }

  const opts = readReconstructOptions();
  const controller = new AbortController();
  state.reconstruct.abort = controller;
  state.reconstruct.running = true;
  state.reconstruct.result = null;
  setReconstructStatus("Running", "running");
  syncReconstructControls();
  startReconstructProgress();

  const form = new FormData();
  form.append("file", state.reconstruct.file, state.reconstruct.file.name || "reconstruction-source.png");
  form.append("model", selModel());
  form.append("device", selDevice());
  form.append("colormap", selCmap());
  form.append("max_dim", String(opts.maxDim));
  form.append("export_format", opts.exportFormat);
  form.append("max_points", String(opts.maxPoints));
  form.append("preview_points", String(opts.previewPoints));
  form.append("focal_scale", String(opts.focalScale));
  form.append("depth_scale", String(opts.depthScale));
  form.append("depth_near_percentile", String(opts.nearPct));
  form.append("depth_far_percentile", String(opts.farPct));
  form.append("sampling", opts.sampling);
  form.append("include_rgb", String(opts.includeRgb));
  form.append("coordinate_system", opts.coordinateSystem);

  try {
    const response = await apiFetch("/api/reconstruct", { method: "POST", body: form, signal: requestSignal(controller.signal, 240_000) });
    const result = await response.json();
    if (result.status && result.status !== "ok") throw new Error(`Unexpected reconstruction status: ${result.status}`);
    state.reconstruct.result = result;
    renderReconstructionResult(result);
    renderPointCloudViewer(result.preview?.points || []);
    stopReconstructProgress(100, "Point cloud ready");
    setReconstructStatus("Complete", "ready");
    toast(`Point cloud ready (${String(result.artifact_format || opts.exportFormat).toUpperCase()})`, "success");
  } catch (err) {
    if (err.name === "AbortError") {
      stopReconstructProgress(0, "Cancelled");
      setReconstructStatus("Cancelled", "idle");
      toast("3D reconstruction cancelled", "info");
    } else {
      stopReconstructProgress(0, "Failed");
      setReconstructStatus("Error", "error");
      toast(`3D reconstruction failed: ${err.message}`, "error", 7000);
    }
  } finally {
    state.reconstruct.running = false;
    state.reconstruct.abort = null;
    syncReconstructControls();
  }
}

function cancelReconstruction({ quiet = false } = {}) {
  if (state.reconstruct.abort) state.reconstruct.abort.abort();
  clearInterval(state.reconstruct.progressTimer);
  state.reconstruct.progressTimer = null;
  state.reconstruct.running = false;
  state.reconstruct.abort = null;
  updateReconstructProgress(0, "Cancelled", "");
  if (!quiet) setReconstructStatus("Cancelled", "idle");
  syncReconstructControls();
}

function resetReconstructionTelemetry() {
  const fields = [
    el.reconstructPointCount, el.reconstructPreviewCount, el.reconstructArtifactSize, el.reconstructLatency,
    el.reconstructModel, el.reconstructDevice, el.reconstructEngine, el.reconstructResolution,
    el.reconstructBounds, el.reconstructDepthCached, el.reconstructFormatLabel,
  ];
  fields.forEach(node => { if (node) node.textContent = "—"; });
  if (el.reconstructWarnings) el.reconstructWarnings.innerHTML = "";
}

function fmtInt(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString() : "—";
}

function fmtMs(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${Math.round(n)} ms` : "—";
}

function fmtBytes(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let v = n, i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i ? 1 : 0)} ${units[i]}`;
}

function fmtBounds(bounds) {
  const min = bounds?.min, max = bounds?.max;
  const valid = arr => Array.isArray(arr) && arr.length >= 3 && arr.every(v => Number.isFinite(Number(v)));
  if (!valid(min) || !valid(max)) return "—";
  const fmt = arr => arr.slice(0, 3).map(v => Number(v).toFixed(2)).join(", ");
  return `min [${fmt(min)}] · max [${fmt(max)}]`;
}

function renderReconstructionResult(result) {
  if (!result) { resetReconstructionTelemetry(); return; }
  const reconstruction = result.reconstruction || {};
  const preview = result.preview || {};
  if (el.reconstructPointCount) el.reconstructPointCount.textContent = fmtInt(reconstruction.point_count ?? preview.point_count);
  if (el.reconstructPreviewCount) el.reconstructPreviewCount.textContent = `${fmtInt(preview.point_count ?? preview.points?.length)}${preview.truncated ? " shown" : ""}`;
  if (el.reconstructArtifactSize) el.reconstructArtifactSize.textContent = fmtBytes(result.artifact_size_bytes);
  if (el.reconstructLatency) el.reconstructLatency.textContent = fmtMs(result.total_latency_ms ?? result.latency_ms);
  if (el.reconstructModel) el.reconstructModel.textContent = result.model_display_name || result.model_id || result.model || selModel();
  if (el.reconstructDevice) el.reconstructDevice.textContent = result.device_used || selDevice();
  if (el.reconstructEngine) el.reconstructEngine.textContent = result.engine_used || "—";
  if (el.reconstructResolution) {
    const w = result.resolution?.width ?? reconstruction.source_width;
    const h = result.resolution?.height ?? reconstruction.source_height;
    el.reconstructResolution.textContent = Number.isFinite(Number(w)) && Number.isFinite(Number(h)) ? `${w}×${h}` : "—";
  }
  if (el.reconstructBounds) el.reconstructBounds.textContent = fmtBounds(reconstruction.bounds);
  if (el.reconstructDepthCached) el.reconstructDepthCached.textContent = result.depth_cached ? "Yes" : "No";
  if (el.reconstructFormatLabel) el.reconstructFormatLabel.textContent = String(result.artifact_format || readReconstructOptions().exportFormat).toUpperCase();
  if (el.reconstructWarnings) {
    const warnings = Array.isArray(result.warnings) ? result.warnings.filter(Boolean) : [];
    el.reconstructWarnings.innerHTML = warnings.length ? warnings.map(w => `<li>${esc(w)}</li>`).join("") : "";
  }
  const depthSrc = safeDataImagePng(result.depth_map);
  if (el.reconstructDepthPreview) {
    el.reconstructDepthPreview.src = depthSrc;
    el.reconstructDepthPreview.hidden = !depthSrc;
  }
  if (el.reconstructDepthEmpty) el.reconstructDepthEmpty.hidden = Boolean(depthSrc);
  syncReconstructControls();
}

function b64ToBlob(base64, mime = "application/octet-stream") {
  const binary = atob(String(base64 || "").replace(/\s/g, ""));
  const chunks = [];
  for (let i = 0; i < binary.length; i += 65536) {
    const slice = binary.slice(i, i + 65536);
    const bytes = new Uint8Array(slice.length);
    for (let j = 0; j < slice.length; j++) bytes[j] = slice.charCodeAt(j);
    chunks.push(bytes);
  }
  return new Blob(chunks, { type: mime });
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function selectedReconstructionMetadata(result = state.reconstruct.result) {
  if (!result) return null;
  const { artifact_base64, depth_map, preview, ...rest } = result;
  return {
    ...rest,
    preview: preview ? { point_count: preview.point_count, truncated: preview.truncated, points_included: Array.isArray(preview.points) ? preview.points.length : 0 } : null,
    options: readReconstructOptions(),
  };
}

function downloadReconstructionArtifact() {
  const result = state.reconstruct.result;
  if (!result?.artifact_base64) { toast("No point-cloud artifact is available yet", "warning"); return; }
  try {
    const blob = b64ToBlob(result.artifact_base64, result.artifact_mime || "text/plain");
    downloadBlob(blob, result.artifact_filename || `depthlens_point_cloud.${result.artifact_format || "ply"}`);
  } catch (err) {
    toast(`Could not download point cloud: ${err.message}`, "error");
  }
}

function downloadReconstructionMetadata() {
  const meta = selectedReconstructionMetadata();
  if (!meta) return;
  const blob = new Blob([JSON.stringify(meta, null, 2)], { type: "application/json" });
  const stem = String(state.reconstruct.result?.artifact_filename || "depthlens_reconstruction").replace(/\.[^.]+$/, "");
  downloadBlob(blob, `${stem}.json`);
}

async function copyReconstructionMetadata() {
  const meta = selectedReconstructionMetadata();
  if (!meta) return;
  const text = JSON.stringify(meta, null, 2);
  try {
    if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
    else {
      const ta = Object.assign(document.createElement("textarea"), { value: text });
      ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
    }
    toast("Reconstruction metadata copied", "success");
  } catch (err) {
    toast(`Could not copy metadata: ${err.message}`, "error");
  }
}

function downloadReconstructionDepth() {
  const b64 = state.reconstruct.result?.depth_map;
  if (!b64) { toast("No depth preview is available yet", "warning"); return; }
  dlB64("reconstruction_depth.png", b64);
}

function clearPointCloudViewer() {
  stopPointCloudViewer();
  const viewer = state.reconstruct.viewer;
  viewer.points = [];
  viewer.normalized = [];
  viewer.mode = "none";
  if (el.pointCloudPlaceholder) { el.pointCloudPlaceholder.hidden = false; el.pointCloudPlaceholder.textContent = "Generate a reconstruction to preview points."; }
  if (el.pointCloudCanvas) {
    const ctx = el.pointCloudCanvas.getContext("2d");
    ctx?.clearRect(0, 0, el.pointCloudCanvas.width, el.pointCloudCanvas.height);
  }
}

function normalizePointCloud(points) {
  const clean = (Array.isArray(points) ? points : []).map(p => {
    const x = Number(p?.[0]), y = Number(p?.[1]), z = Number(p?.[2]);
    if (![x, y, z].every(Number.isFinite)) return null;
    return { x, y, z, r: Number(p?.[3]), g: Number(p?.[4]), b: Number(p?.[5]) };
  }).filter(Boolean);
  if (!clean.length) return [];
  const min = { x: Infinity, y: Infinity, z: Infinity }, max = { x: -Infinity, y: -Infinity, z: -Infinity };
  clean.forEach(p => { min.x = Math.min(min.x, p.x); min.y = Math.min(min.y, p.y); min.z = Math.min(min.z, p.z); max.x = Math.max(max.x, p.x); max.y = Math.max(max.y, p.y); max.z = Math.max(max.z, p.z); });
  const center = { x: (min.x + max.x) / 2, y: (min.y + max.y) / 2, z: (min.z + max.z) / 2 };
  const extent = Math.max(max.x - min.x, max.y - min.y, max.z - min.z, 1e-6);
  return clean.map(p => ({ x: (p.x - center.x) / extent, y: (p.y - center.y) / extent, z: (p.z - center.z) / extent, r: p.r, g: p.g, b: p.b }));
}

function renderPointCloudViewer(points = []) {
  const viewer = state.reconstruct.viewer;
  viewer.points = Array.isArray(points) ? points : [];
  viewer.normalized = normalizePointCloud(viewer.points);
  viewer.ctx2d = el.pointCloudCanvas?.getContext("2d") || null;
  viewer.gl = null;
  viewer.mode = viewer.ctx2d ? "2d" : "none";
  if (el.pointCloudPlaceholder) {
    el.pointCloudPlaceholder.hidden = Boolean(viewer.normalized.length);
    el.pointCloudPlaceholder.textContent = viewer.normalized.length ? "" : "No preview points returned by the backend.";
  }
  if (el.pointCloudModeMessage) {
    el.pointCloudModeMessage.textContent = pointCloudWebglAvailable()
      ? "Native 2D canvas preview · drag to rotate · wheel to zoom · double-click to reset."
      : "WebGL unavailable — showing 2D preview.";
  }
  resizePointCloudCanvas();
  drawPointCloudFrame();
  startPointCloudViewer();
}

function startPointCloudViewer() {
  const viewer = state.reconstruct.viewer;
  stopPointCloudViewer();
  const tick = () => {
    if (viewer.autoRotate) viewer.rotationY += 0.006;
    drawPointCloudFrame();
    if (viewer.autoRotate) viewer.animationId = requestAnimationFrame(tick);
  };
  if (viewer.autoRotate && viewer.normalized.length) viewer.animationId = requestAnimationFrame(tick);
}

function stopPointCloudViewer() {
  const viewer = state.reconstruct.viewer;
  if (viewer.animationId) cancelAnimationFrame(viewer.animationId);
  viewer.animationId = null;
}

function resizePointCloudCanvas() {
  const canvas = el.pointCloudCanvas;
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width || canvas.clientWidth || 640));
  const height = Math.max(260, Math.floor(rect.height || canvas.clientHeight || 420));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const nextW = Math.floor(width * dpr), nextH = Math.floor(height * dpr);
  if (canvas.width !== nextW || canvas.height !== nextH) {
    canvas.width = nextW; canvas.height = nextH;
  }
  canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
}

function drawPointCloudFrame() {
  const canvas = el.pointCloudCanvas;
  const ctx = state.reconstruct.viewer.ctx2d;
  const points = state.reconstruct.viewer.normalized;
  if (!canvas || !ctx) return;
  resizePointCloudCanvas();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.width / dpr, height = canvas.height / dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const dark = document.documentElement.getAttribute("data-theme") !== "light";
  ctx.fillStyle = dark ? "rgba(5,10,18,0.92)" : "rgba(245,248,252,0.96)";
  ctx.fillRect(0, 0, width, height);
  if (!points.length) return;

  const { rotationX, rotationY, zoom, pointSize } = state.reconstruct.viewer;
  const cx = Math.cos(rotationX), sx = Math.sin(rotationX), cy = Math.cos(rotationY), sy = Math.sin(rotationY);
  const screenScale = Math.min(width, height) * 0.42 * zoom;
  const fallback = dark ? [0, 200, 255] : [0, 112, 204];
  const projected = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const x1 = p.x * cy + p.z * sy;
    const z1 = -p.x * sy + p.z * cy;
    const y2 = p.y * cx - z1 * sx;
    const z2 = p.y * sx + z1 * cx;
    const perspective = 1 / Math.max(0.25, 1 + z2 * 0.35);
    projected.push({
      sx: width / 2 + x1 * screenScale * perspective,
      sy: height / 2 - y2 * screenScale * perspective,
      z: z2,
      r: Number.isFinite(p.r) ? p.r : fallback[0],
      g: Number.isFinite(p.g) ? p.g : fallback[1],
      b: Number.isFinite(p.b) ? p.b : fallback[2],
    });
  }
  projected.sort((a, b) => a.z - b.z);
  const size = Math.max(1, Number(pointSize) || 2);
  for (const p of projected) {
    if (p.sx < -10 || p.sy < -10 || p.sx > width + 10 || p.sy > height + 10) continue;
    ctx.fillStyle = `rgb(${Math.max(0, Math.min(255, p.r))},${Math.max(0, Math.min(255, p.g))},${Math.max(0, Math.min(255, p.b))})`;
    ctx.fillRect(p.sx, p.sy, size, size);
  }
}

function resetPointCloudView() {
  Object.assign(state.reconstruct.viewer, { rotationX: -0.4, rotationY: 0.65, zoom: 1.8 });
  drawPointCloudFrame();
}

function initReconstructionPanel() {
  if (!el.reconstructDropZone) return;
  syncReconstructControls();
  resetReconstructionTelemetry();
  el.reconstructBrowseBtn?.addEventListener("click", () => el.reconstructFileInput?.click());
  el.reconstructDropZone.addEventListener("click", event => { if (!event.target.closest("button")) el.reconstructFileInput?.click(); });
  el.reconstructDropZone.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); el.reconstructFileInput?.click(); } });
  el.reconstructDropZone.addEventListener("dragover", event => { event.preventDefault(); el.reconstructDropZone.classList.add("drag-over"); });
  el.reconstructDropZone.addEventListener("dragleave", event => { if (!el.reconstructDropZone.contains(event.relatedTarget)) el.reconstructDropZone.classList.remove("drag-over"); });
  el.reconstructDropZone.addEventListener("drop", event => { event.preventDefault(); el.reconstructDropZone.classList.remove("drag-over"); setReconstructFile(event.dataTransfer?.files?.[0]); });
  el.reconstructFileInput?.addEventListener("change", () => { setReconstructFile(el.reconstructFileInput.files?.[0]); el.reconstructFileInput.value = ""; });
  el.reconstructUseLatestBtn?.addEventListener("click", useLatestReconstructionSource);
  el.reconstructClearBtn?.addEventListener("click", clearReconstruct);
  el.reconstructRunBtn?.addEventListener("click", runReconstruction);
  el.reconstructCancelBtn?.addEventListener("click", () => cancelReconstruction());
  el.reconstructDownloadBtn?.addEventListener("click", downloadReconstructionArtifact);
  el.reconstructDownloadMetaBtn?.addEventListener("click", downloadReconstructionMetadata);
  el.reconstructCopyMetaBtn?.addEventListener("click", copyReconstructionMetadata);
  el.reconstructDownloadDepthBtn?.addEventListener("click", downloadReconstructionDepth);
  el.pointCloudResetViewBtn?.addEventListener("click", resetPointCloudView);
  el.pointCloudPointSize?.addEventListener("input", () => { state.reconstruct.viewer.pointSize = readNumberInput(el.pointCloudPointSize, 2); drawPointCloudFrame(); });
  el.pointCloudAutoRotate?.addEventListener("change", () => { state.reconstruct.viewer.autoRotate = Boolean(el.pointCloudAutoRotate.checked); state.reconstruct.viewer.autoRotate ? startPointCloudViewer() : stopPointCloudViewer(); drawPointCloudFrame(); });
  el.pointCloudCanvas?.addEventListener("pointerdown", event => { state.reconstruct.viewer.dragging = true; state.reconstruct.viewer.lastX = event.clientX; state.reconstruct.viewer.lastY = event.clientY; el.pointCloudCanvas.setPointerCapture?.(event.pointerId); });
  el.pointCloudCanvas?.addEventListener("pointermove", event => {
    const viewer = state.reconstruct.viewer;
    if (!viewer.dragging) return;
    viewer.rotationY += (event.clientX - viewer.lastX) * 0.008;
    viewer.rotationX += (event.clientY - viewer.lastY) * 0.008;
    viewer.rotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, viewer.rotationX));
    viewer.lastX = event.clientX; viewer.lastY = event.clientY;
    drawPointCloudFrame();
  });
  ["pointerup", "pointerleave", "pointercancel"].forEach(type => el.pointCloudCanvas?.addEventListener(type, () => { state.reconstruct.viewer.dragging = false; }));
  el.pointCloudCanvas?.addEventListener("wheel", event => { event.preventDefault(); const viewer = state.reconstruct.viewer; viewer.zoom = Math.max(0.35, Math.min(6, viewer.zoom * (event.deltaY > 0 ? 0.9 : 1.1))); drawPointCloudFrame(); }, { passive: false });
  el.pointCloudCanvas?.addEventListener("dblclick", resetPointCloudView);
  window.addEventListener("resize", () => { if (!document.getElementById("panel-reconstruct")?.hidden) { resizePointCloudCanvas(); drawPointCloudFrame(); } });
  document.addEventListener("depthlens-theme-changed", () => drawPointCloudFrame());
}

// ══════════════════════════════════════════════════════════════
// POLISHED UI MOTION + GUIDE ACCORDION
// ══════════════════════════════════════════════════════════════
function prefersReducedMotion() {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches || false;
}

function bindPointerGlow(selector, { tilt = 0 } = {}) {
  $$(selector).forEach(node => {
    if (node.dataset.pointerGlowBound === "true") return;
    node.dataset.pointerGlowBound = "true";
    node.style.setProperty("--mx", "50%");
    node.style.setProperty("--my", "50%");
    node.style.setProperty("--tilt-x", "0deg");
    node.style.setProperty("--tilt-y", "0deg");

    node.addEventListener("pointermove", event => {
      const rect = node.getBoundingClientRect();
      const px = (event.clientX - rect.left) / Math.max(rect.width, 1);
      const py = (event.clientY - rect.top) / Math.max(rect.height, 1);
      node.style.setProperty("--mx", `${(px * 100).toFixed(1)}%`);
      node.style.setProperty("--my", `${(py * 100).toFixed(1)}%`);
      if (!prefersReducedMotion() && tilt) {
        node.style.setProperty("--tilt-y", `${((px - 0.5) * tilt).toFixed(2)}deg`);
        node.style.setProperty("--tilt-x", `${((0.5 - py) * tilt).toFixed(2)}deg`);
      }
    });

    node.addEventListener("pointerleave", () => {
      node.style.setProperty("--mx", "50%");
      node.style.setProperty("--my", "50%");
      node.style.setProperty("--tilt-x", "0deg");
      node.style.setProperty("--tilt-y", "0deg");
    });
  });
}

function initScrollableNav() {
  const shell = el.headerNavShell;
  if (!shell || shell.dataset.scrollNavBound === "true") return;
  shell.dataset.scrollNavBound = "true";

  let pointerActive = false;
  let dragMoved = false;
  let ignoreNextClick = false;
  let startX = 0;
  let startY = 0;
  let startScrollLeft = 0;
  const dragThreshold = 8;

  shell.addEventListener("wheel", event => {
    if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
    if (shell.scrollWidth <= shell.clientWidth) return;
    event.preventDefault();
    shell.scrollLeft += event.deltaY;
  }, { passive: false });

  shell.addEventListener("pointerdown", event => {
    if (event.button !== undefined && event.button !== 0) return;
    pointerActive = true;
    dragMoved = false;
    ignoreNextClick = false;
    startX = event.clientX;
    startY = event.clientY;
    startScrollLeft = shell.scrollLeft;
    shell.setPointerCapture?.(event.pointerId);
  });

  shell.addEventListener("pointermove", event => {
    if (!pointerActive) return;
    const dx = event.clientX - startX;
    const dy = event.clientY - startY;
    const hasHorizontalIntent = Math.abs(dx) > dragThreshold && Math.abs(dx) > Math.abs(dy) * 1.15;
    if (!dragMoved && hasHorizontalIntent) {
      dragMoved = true;
      shell.classList.add("dragging");
    }
    if (dragMoved) {
      event.preventDefault();
      shell.scrollLeft = startScrollLeft - dx;
    }
  });

  ["pointerup", "pointercancel", "pointerleave"].forEach(type => {
    shell.addEventListener(type, event => {
      if (!pointerActive) return;
      pointerActive = false;
      ignoreNextClick = dragMoved;
      shell.releasePointerCapture?.(event.pointerId);
      shell.classList.remove("dragging");
      window.setTimeout(() => { ignoreNextClick = false; }, 0);
    });
  });

  shell.addEventListener("click", event => {
    if (!ignoreNextClick) return;
    event.preventDefault();
    event.stopPropagation();
    ignoreNextClick = false;
  }, true);
}

function setGuideSectionOpen(section, open) {
  const toggle = $(".guide-section-toggle", section);
  const body = $(".guide-section-body", section);
  if (!toggle || !body) return;

  section.classList.toggle("open", open);
  toggle.setAttribute("aria-expanded", String(open));

  if (prefersReducedMotion()) {
    body.style.maxHeight = open ? "none" : "0px";
    return;
  }

  if (open) {
    body.style.maxHeight = `${body.scrollHeight}px`;
  } else {
    body.style.maxHeight = `${body.scrollHeight}px`;
    requestAnimationFrame(() => { body.style.maxHeight = "0px"; });
  }
}

function refreshOpenGuideHeights() {
  $$(".guide-section.open .guide-section-body").forEach(body => {
    body.style.maxHeight = prefersReducedMotion() ? "none" : `${body.scrollHeight}px`;
  });
}

function initGuideAccordion() {
  const accordion = $("#guideAccordion");
  if (!accordion || accordion.dataset.guideBound === "true") return;
  accordion.dataset.guideBound = "true";

  const toggles = $$(".guide-section-toggle", accordion);
  toggles.forEach((toggle, index) => {
    const section = toggle.closest(".guide-section");
    if (!section) return;
    setGuideSectionOpen(section, section.classList.contains("open") || toggle.getAttribute("aria-expanded") === "true");

    toggle.addEventListener("click", () => {
      setGuideSectionOpen(section, toggle.getAttribute("aria-expanded") !== "true");
    });

    toggle.addEventListener("keydown", event => {
      if (!["ArrowDown", "ArrowUp"].includes(event.key)) return;
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      const next = toggles[(index + direction + toggles.length) % toggles.length];
      next?.focus();
    });
  });

  window.addEventListener("resize", refreshOpenGuideHeights);
}

// ══════════════════════════════════════════════════════════════
// PANEL NAVIGATION
// ══════════════════════════════════════════════════════════════
function switchPanel(name) {
  const targetId = `panel-${name}`;
  const targetPanel = el.panels.find?.(p => p.id === targetId) || document.getElementById(targetId);
  if (!targetPanel) return;

  el.navBtns.forEach(b => b.classList.toggle("active", b.dataset.panel === name));

  el.panels.forEach(p => {
    const isActive = p.id === targetId;
    p.hidden = !isActive;
    p.classList.toggle("active", isActive);
  });

  const activeBtn = el.navBtns.find?.(b => b.dataset.panel === name) || $(`.nav-btn[data-panel="${cssEscapeValue(name)}"]`);
  activeBtn?.scrollIntoView({
    block: "nearest",
    inline: "center",
    behavior: prefersReducedMotion() ? "auto" : "smooth",
  });

  if (name === "guide") refreshOpenGuideHeights();

  if (name === "reconstruct") {
    syncReconstructControls();
    resizePointCloudCanvas();
    drawPointCloudFrame();
    if (state.reconstruct.viewer.autoRotate) startPointCloudViewer();
  } else {
    stopPointCloudViewer();
  }
}
el.navBtns.forEach(b => b.addEventListener("click", event => {
  const panelName = event.currentTarget?.dataset?.panel;
  if (panelName) switchPanel(panelName);
}));

// ══════════════════════════════════════════════════════════════
// GROUND TRUTH MODE
// ══════════════════════════════════════════════════════════════
const GT_ALLOWED = /\.(png|tif|tiff|npy)$/i;
function syncGtUi() {
  if (!el.gtToggle) return;
  state.gtMode = Boolean(el.gtToggle.checked);
  if (el.gtUpload) el.gtUpload.hidden = !state.gtMode;
  if (el.fileInput) el.fileInput.multiple = !state.gtMode;
  if (!state.gtMode) {
    state.gtFile = null;
    if (el.gtFileInput) el.gtFileInput.value = "";
    if (el.gtFileName) el.gtFileName.textContent = "No GT file selected";
  }
  syncQueueControls();
}
function setGtFile(file) {
  if (!file) return;
  if (!GT_ALLOWED.test(file.name)) { toast("GT depth must be PNG, TIFF, or NPY", "warning"); return; }
  if (file.size > 20*1024*1024) { toast("GT depth exceeds 20 MB", "warning"); return; }
  state.gtFile = file;
  if (el.gtFileName) el.gtFileName.textContent = file.name;
  syncQueueControls();
}
el.gtToggle?.addEventListener("change", () => {
  syncGtUi();
  toast(state.gtMode ? "GT mode enabled — select one image and one GT depth file" : "GT mode disabled — standard batch upload restored", "info");
});
el.gtFileInput?.addEventListener("change", () => { setGtFile(el.gtFileInput.files[0]); });
el.clearGtBtn?.addEventListener("click", () => { state.gtFile=null; if (el.gtFileInput) el.gtFileInput.value=""; if (el.gtFileName) el.gtFileName.textContent="No GT file selected"; syncQueueControls(); });

// ══════════════════════════════════════════════════════════════
// FILE HANDLING
// ══════════════════════════════════════════════════════════════
const ALLOWED = /^image\//;
function uid() { return Math.random().toString(36).slice(2,9); }
function fmtSize(b) {
  if (b<1024) return `${b} B`;
  if (b<1048576) return `${(b/1024).toFixed(1)} KB`;
  return `${(b/1048576).toFixed(1)} MB`;
}

function addFiles(list) {
  let added = 0;
  for (const file of list) {
    if (state.gtMode && state.files.length + added >= 1) { toast("GT mode accepts exactly one source image", "warning"); break; }
    if (!ALLOWED.test(file.type)) { toast(`Skipped "${file.name}" — not an image`,"warning"); continue; }
    if (file.size > 20*1024*1024) { toast(`Skipped "${file.name}" — exceeds 20 MB`,"warning"); continue; }
    if (state.files.some(f=>f.file.name===file.name&&f.file.size===file.size)) continue;
    const entry = {id:uid(),file,thumb:null,status:"pending",result:null};
    state.files.push(entry);
    renderFileItem(entry);
    const rd = new FileReader();
    rd.onload = e => {
      entry.thumb = e.target.result;
      const img = $(`#fthumb-${entry.id}`);
      if (img) img.src = e.target.result;
    };
    rd.readAsDataURL(file);
    added++;
  }
  if (added) { syncQueueControls(); toast(`${added} file${added>1?"s":""} added`); }
}

function renderFileItem(entry) {
  const li = document.createElement("li");
  li.className="file-item"; li.id=`fitem-${entry.id}`;
  li.innerHTML = `
    <img class="file-thumb" id="fthumb-${entry.id}" src="" alt="" />
    <div class="file-meta">
      <div class="file-name" title="${esc(entry.file.name)}">${esc(entry.file.name)}</div>
      <div class="file-size">${fmtSize(entry.file.size)}</div>
    </div>
    <span class="file-status pending" id="fst-${entry.id}">Pending</span>
    <button class="file-remove" data-id="${entry.id}" aria-label="Remove ${esc(entry.file.name)}">✕</button>`;
  el.fileQueue.appendChild(li);
  $(".file-remove",li).addEventListener("click",()=>removeFile(entry.id));
}

function removeFile(id) {
  if (state.files.find(f=>f.id===id)?.status==="running") return;
  state.files = state.files.filter(f=>f.id!==id);
  $(`#fitem-${id}`)?.remove();
  syncQueueControls();
}

function syncQueueControls() {
  const has = state.files.length>0;
  if (el.clearBtn) el.clearBtn.disabled = !has || Boolean(state.abort);
  const ready = engineReady();
  const gtBlocked = state.gtMode && (state.files.length !== 1 || !state.gtFile);
  if (el.runBtn) {
    el.runBtn.disabled = !has || Boolean(state.abort) || state.initializingBackend || !ready || gtBlocked;
    el.runBtn.title = state.initializingBackend ? "Starting depth engine…" : (!backendOnline ? `Depth engine offline at ${API || DEFAULT_API_BASE_URL}` : (!inferenceReady ? "Inference runtime is not ready; open Diagnostics or check /ready" : (gtBlocked ? "GT mode requires exactly one source image and one GT depth file" : "")));
  }
}

function setFileSt(id,cls,txt) {
  const s=$(`#fst-${id}`); if (!s) return;
  s.className=`file-status ${cls}`; s.textContent=txt;
}

el.fileInput?.addEventListener("change",()=>{ addFiles(el.fileInput.files); el.fileInput.value=""; });
el.dropZone?.addEventListener("dragover",e=>{ e.preventDefault(); el.dropZone.classList.add("drag-over"); });
el.dropZone?.addEventListener("dragleave",e=>{ if (!el.dropZone.contains(e.relatedTarget)) el.dropZone.classList.remove("drag-over"); });
el.dropZone?.addEventListener("drop",e=>{ e.preventDefault(); el.dropZone.classList.remove("drag-over"); addFiles(e.dataTransfer.files); });
el.dropZone?.addEventListener("click",e=>{ if (e.target.closest("#fileInput")) return; el.fileInput.click(); });
el.dropZone?.addEventListener("keydown",e=>{ if (e.key==="Enter"||e.key===" "){e.preventDefault();el.fileInput.click();} });
el.clearBtn?.addEventListener("click",()=>{ state.files=[]; el.fileQueue.innerHTML=""; syncQueueControls(); });

// ══════════════════════════════════════════════════════════════
// INFERENCE
// ══════════════════════════════════════════════════════════════
function selModel()  { return $('input[name="model"]:checked')?.value    || "MiDaS_small"; }
function selCmap()   { return $('input[name="colormap"]:checked')?.value || "inferno"; }
function selDevice() { return $('input[name="device"]:checked')?.value || "auto"; }

function setProgress(pct,status,eta,currentFile,countStr) {
  el.progressFill.style.width = `${pct}%`;
  el.progressPct.textContent = `${Math.round(pct)}%`;
  if (el.progressBar) el.progressBar.setAttribute("aria-valuenow",pct);
  el.progressStatusText.textContent = status;
  el.progressEta.textContent = eta||"";
  if (currentFile!==undefined) el.progressCurrentFile.textContent = currentFile;
  if (countStr!==undefined) el.progressItemCount.textContent = countStr;
}

function timingKey(model,device) { return `${model}::${device||"cpu"}`; }
function getEstimate(mode,model,device) {
  return state.timing[mode]?.[timingKey(model,device)] || state.timing[mode]?.default || 1300;
}
function updateEstimate(mode,model,device,ms) {
  const key=timingKey(model,device);
  const prev=state.timing[mode][key];
  state.timing[mode][key] = prev ? prev*0.65+ms*0.35 : ms;
  const vals=Object.values(state.timing[mode]).filter(v=>typeof v==="number");
  if (vals.length) state.timing[mode].default = vals.reduce((a,b)=>a+b,0)/vals.length;
}

function fmtDuration(ms) {
  const s=Math.round(ms/1000);
  if (s<60) return `${s}s`;
  if (s<3600) return `${Math.floor(s/60)}m ${s%60}s`;
  return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m ${s%60}s`;
}

el.runBtn?.addEventListener("click",runBatch);
el.cancelBtn?.addEventListener("click",cancelBatch);

function cancelBatch() {
  if (state.abort) { state.abort.abort(); state.abort=null; }
  toast("Batch cancelled","warning");
}

async function runBatch() {
  if (!engineReady()) {
    const ok = await checkHealth();
    if (!ok) { toast(`Depth engine is unavailable at ${API || DEFAULT_API_BASE_URL}`, "error", 6000); return; }
  }
  const pending = state.files.filter(f=>f.status==="pending"||f.status==="error");
  if (!pending.length) return;
  state.abort = new AbortController();
  el.runBtn.disabled=true; el.clearBtn.disabled=true;
  el.cancelBtn.hidden=false; el.progressBlock.hidden=false;
  setProgress(0,"Starting batch…","","",`0 / ${pending.length}`);

  const batchStart=Date.now();
  const model=selModel(), colormap=selCmap(), device=selDevice();

  try {
  for (let i=0;i<pending.length;i++) {
    if (state.abort?.signal?.aborted) break;
    const entry=pending[i];
    setFileSt(entry.id,"running","Running…");
    const estCurrent=getEstimate("workspace",model,device);
    const estRemainingStatic=pending.slice(i+1).reduce(acc=>acc+getEstimate("workspace",model,device),0);
    const itemStart=Date.now();
    const tick=setInterval(()=>{
      const elapsedCurrent=Date.now()-itemStart;
      const unitDone=i+Math.min(elapsedCurrent/estCurrent,0.98);
      const pct=Math.min((unitDone/pending.length)*100,99);
      const rem=Math.max(0,estCurrent-elapsedCurrent+estRemainingStatic);
      setProgress(pct,`Processing ${i+1} of ${pending.length}…`,`ETA: ${fmtDuration(rem)}`,esc(entry.file.name),`${i+1} / ${pending.length}`);
    },120);

    try {
      const result=await inferOne(entry.file,model,colormap,device,state.abort.signal,state.gtMode?"full":"fast",state.gtMode?"color,gray":"color",state.gtMode?state.gtFile:null,state.gtMode);
      clearInterval(tick);
      updateEstimate("workspace",model,device,result.latency_ms);
      entry.result=result; entry.status=result.fallback_used?"completed_with_warning":"done";
      setFileSt(entry.id,result.fallback_used?"warning":"done",`${result.fallback_used?"⚠":"✓"} ${result.latency_ms}ms${result.engine_used?` · ${result.engine_used}`:""}`);
      if (result.fallback_used) toastOnce("Depth map generated using PyTorch fallback because ONNX is unavailable.", "warning", 4500);
      state.session.total++; state.session.totalInferenceMs+=result.latency_ms;
      if (result.cached) state.session.cached++;
      state.session.latencies.push(result.latency_ms);
      updateMetrics(); pushLatency(result.latency_ms);
      loadCacheMetrics();
      state.results.push({...result,originalSrc:entry.thumb,filename:entry.file.name});
      appendGalleryItem(state.results.at(-1));
      syncReconstructControls();
      el.resultsCard.hidden=false;
    } catch(err) {
      clearInterval(tick);
      if (err.name==="AbortError") { setFileSt(entry.id,"pending","Cancelled"); break; }
      entry.status="error"; setFileSt(entry.id,"error","Error");
      state.session.errors++; updateMetrics();
      toast(`"${entry.file.name}": ${err.message}`,"error");
    }
  }

  const elapsed=Date.now()-batchStart;
  const done=pending.filter(e=>e.status==="done"||e.status==="completed_with_warning").length;
  setProgress(100,`Done — ${done} image${done!==1?"s":""} in ${fmtDuration(elapsed)}`,"");
  if (done>0) toast(`Batch complete — ${done} succeeded`,"success");
  } catch (err) {
    pending.filter(e=>e.status==="running").forEach(e=>{ e.status="error"; setFileSt(e.id,"error","Error"); });
    state.session.errors += pending.filter(e=>e.status==="error").length;
    updateMetrics();
    setProgress(100,"Batch failed",err.message || "Inference failed");
    toast(`Batch failed: ${err.message}`, "error", 6000);
  } finally {
    setTimeout(()=>{ el.progressBlock.hidden=true; },3000);
    state.abort=null;
    el.cancelBtn.hidden=true;
    syncQueueControls();
  }
}

async function inferOne(file,model,colormap,device,signal,metrics="fast",outputs="color",gtFile=null,gtRequired=false,maxDim=null) {
  const fd=new FormData();
  fd.append("file",file); fd.append("model",model);
  fd.append("colormap",colormap); fd.append("device",device);
  fd.append("metrics", metrics);
  fd.append("outputs", outputs);
  if (gtFile) fd.append("gt_file", gtFile);
  if (gtRequired) fd.append("gt_required", "true");
  if (maxDim) fd.append("max_dim", String(maxDim));
  if (!engineReady()) throw new Error(`Depth engine is unavailable at ${API || DEFAULT_API_BASE_URL}`);
  const res=await apiFetch("/estimate",{
    method:"POST", body:fd,
    signal: requestSignal(signal, 180_000),
  });
  return res.json();
}

// ══════════════════════════════════════════════════════════════
// REAL-TIME WEBCAM DEPTH
// ══════════════════════════════════════════════════════════════
function webcamSupported() {
  return Boolean(navigator.mediaDevices?.getUserMedia && window.HTMLCanvasElement && window.File);
}
function getWebcamTargetFps() {
  const fps = Number(el.webcamTargetFps?.value || 2);
  return [1,2,3,5].includes(fps) ? fps : 2;
}
function getWebcamMaxDim() {
  const maxDim = Number(el.webcamMaxDim?.value || 384);
  return [256,384,512].includes(maxDim) ? maxDim : 384;
}
function getWebcamSmoothingAlpha() {
  const alpha = Number(el.webcamSmoothing?.value || 0.25);
  return Number.isFinite(alpha) ? Math.max(0, Math.min(0.95, alpha)) : 0.25;
}
function fmtMs(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${Math.round(n)} ms` : "—";
}
function setWebcamStatus(cameraState, inferenceState) {
  if (cameraState && el.webcamCameraState) el.webcamCameraState.textContent = cameraState;
  if (inferenceState && el.webcamInferenceState) el.webcamInferenceState.textContent = inferenceState;
  if (el.webcamStatusPill) {
    const camera = el.webcamCameraState?.textContent || "Stopped";
    const inference = el.webcamInferenceState?.textContent || "Idle";
    el.webcamStatusPill.textContent = `${camera} · ${inference}`;
    el.webcamStatusPill.dataset.state = state.webcam.running ? (state.webcam.paused || state.webcam.hiddenPaused ? "paused" : "running") : "stopped";
  }
}
function appendWebcamLog(message, type = "info") {
  if (!el.webcamLog) return;
  const item = document.createElement("div");
  item.className = `webcam-log-item ${type}`;
  item.textContent = `${new Date().toLocaleTimeString()} · ${message}`;
  el.webcamLog.prepend(item);
  while (el.webcamLog.children.length > 5) el.webcamLog.lastElementChild?.remove();
}
function updateWebcamTelemetry() {
  const wc = state.webcam;
  const elapsed = wc.startedAt ? Math.max((performance.now() - wc.startedAt) / 1000, 0.001) : 0;
  const effective = wc.processed && elapsed ? wc.processed / elapsed : 0;
  if (el.webcamTargetFpsMetric) el.webcamTargetFpsMetric.textContent = String(getWebcamTargetFps());
  if (el.webcamEffectiveFps) el.webcamEffectiveFps.textContent = effective.toFixed(2);
  if (el.webcamBackendLatency) el.webcamBackendLatency.textContent = fmtMs(wc.latencies.at(-1));
  if (el.webcamEndToEndLatency) el.webcamEndToEndLatency.textContent = fmtMs(wc.e2eLatencies.at(-1));
  if (el.webcamProcessedFrames) el.webcamProcessedFrames.textContent = String(wc.processed);
  if (el.webcamSkippedFrames) el.webcamSkippedFrames.textContent = String(wc.skipped);
  if (el.webcamErrorCount) el.webcamErrorCount.textContent = String(wc.errors);
  if (el.webcamActiveModel) el.webcamActiveModel.textContent = selModel();
  if (el.webcamActiveDevice) el.webcamActiveDevice.textContent = selDevice();
  if (el.webcamActiveColormap) el.webcamActiveColormap.textContent = selCmap();
  if (!wc.running) setWebcamStatus("Stopped", wc.lastDepthBase64 || wc.lastDepthDataUrl ? "Idle" : "Idle");
  syncWebcamControls();
}
function syncWebcamControls() {
  const supported = webcamSupported();
  const wc = state.webcam;
  if (el.webcamStartBtn) {
    el.webcamStartBtn.disabled = wc.running || wc.starting || !supported;
    el.webcamStartBtn.title = supported ? (engineReady() ? "Start webcam depth inference" : "Depth engine readiness will be checked before starting") : "This browser does not support camera capture";
  }
  if (el.webcamStopBtn) el.webcamStopBtn.disabled = !wc.running;
  if (el.webcamPauseBtn) {
    el.webcamPauseBtn.disabled = !wc.running;
    el.webcamPauseBtn.textContent = wc.paused ? "Resume Inference" : "Pause Inference";
  }
  if (el.webcamCaptureBtn) el.webcamCaptureBtn.disabled = !(wc.lastDepthBase64 || wc.lastDepthDataUrl);
}
async function startWebcam() {
  if (!webcamSupported()) { toast("Camera capture is not supported in this browser.", "error"); return; }
  if (state.webcam.running || state.webcam.starting) return;
  state.webcam.starting = true;
  syncWebcamControls();
  if (!engineReady()) {
    appendWebcamLog("Checking depth engine readiness…");
    const ok = await checkHealth();
    if (!ok) {
      state.webcam.starting = false;
      toast(`Depth engine is unavailable at ${API || DEFAULT_API_BASE_URL}`, "error", 6000);
      updateWebcamTelemetry();
      return;
    }
  }
  try {
    setWebcamStatus("Requesting camera", "Idle");
    syncWebcamControls();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    state.webcam.starting = false;
    state.webcam.stream = stream;
    if (el.webcamVideo) {
      el.webcamVideo.srcObject = stream;
      await waitForVideoMetadata(el.webcamVideo);
      await el.webcamVideo.play?.();
    }
    Object.assign(state.webcam, {
      running: true, paused: false, hiddenPaused: false, inFlight: false,
      processed: 0, skipped: 0, errors: 0, consecutiveErrors: 0,
      latencies: [], e2eLatencies: [], startedAt: performance.now(),
      previousDepthImageData: null,
    });
    state.webcam.smoothingAlpha = getWebcamSmoothingAlpha();
    setWebcamStatus("Running", "Scheduled");
    appendWebcamLog("Camera started", "success");
    syncWebcamControls();
    updateWebcamTelemetry();
    startWebcamLoop();
  } catch (err) {
    state.webcam.starting = false;
    state.webcam.errors++;
    setWebcamStatus("Stopped", "Camera error");
    appendWebcamLog(`Camera error: ${err.message}`, "error");
    toast(`Camera unavailable: ${err.message}`, "error", 7000);
    stopWebcam({ quiet: true });
  }
}
function waitForVideoMetadata(video) {
  if (!video) return Promise.reject(new Error("Video element is missing"));
  if (video.videoWidth && video.videoHeight) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const cleanup = () => { video.removeEventListener("loadedmetadata", onLoaded); video.removeEventListener("error", onError); };
    const onLoaded = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error("Unable to load webcam metadata")); };
    video.addEventListener("loadedmetadata", onLoaded, { once: true });
    video.addEventListener("error", onError, { once: true });
    setTimeout(() => { cleanup(); reject(new Error("Timed out waiting for webcam metadata")); }, 8000);
  });
}
function stopWebcam({ quiet = false } = {}) {
  clearTimeout(state.webcam.loopTimer);
  state.webcam.loopTimer = null;
  state.webcam.abort?.abort();
  state.webcam.abort = null;
  state.webcam.stream?.getTracks().forEach(track => track.stop());
  state.webcam.stream = null;
  if (el.webcamVideo) el.webcamVideo.srcObject = null;
  state.webcam.running = false;
  state.webcam.starting = false;
  state.webcam.paused = false;
  state.webcam.hiddenPaused = false;
  state.webcam.inFlight = false;
  setWebcamStatus("Stopped", "Idle");
  syncWebcamControls();
  updateWebcamTelemetry();
  if (!quiet) { appendWebcamLog("Camera stopped", "warning"); toast("Webcam stopped", "info"); }
}
function toggleWebcamPause() {
  if (!state.webcam.running) return;
  state.webcam.paused = !state.webcam.paused;
  state.webcam.hiddenPaused = false;
  setWebcamStatus("Running", state.webcam.paused ? "Paused" : "Scheduled");
  appendWebcamLog(state.webcam.paused ? "Inference paused" : "Inference resumed", state.webcam.paused ? "warning" : "success");
  syncWebcamControls();
  scheduleNextWebcamFrame(0);
}
function startWebcamLoop() {
  clearTimeout(state.webcam.loopTimer);
  scheduleNextWebcamFrame(0);
}
function scheduleNextWebcamFrame(delayMs = null) {
  clearTimeout(state.webcam.loopTimer);
  if (!state.webcam.running) return;
  const interval = 1000 / getWebcamTargetFps();
  state.webcam.loopTimer = setTimeout(processWebcamFrame, delayMs ?? interval);
}
async function processWebcamFrame() {
  const wc = state.webcam;
  if (!wc.running) return;
  if (document.hidden) {
    wc.hiddenPaused = true;
    setWebcamStatus("Running", "Paused while hidden");
    updateWebcamTelemetry();
    scheduleNextWebcamFrame(1000);
    return;
  }
  if (wc.hiddenPaused) {
    wc.hiddenPaused = false;
    if (!wc.paused) setWebcamStatus("Running", "Scheduled");
  }
  if (wc.paused) { setWebcamStatus("Running", "Paused"); updateWebcamTelemetry(); scheduleNextWebcamFrame(); return; }
  if (wc.inFlight) { wc.skipped++; updateWebcamTelemetry(); scheduleNextWebcamFrame(); return; }
  wc.inFlight = true;
  wc.abort = new AbortController();
  wc.lastLoopStartedAt = performance.now();
  setWebcamStatus("Running", "Processing");
  try {
    const maxDim = getWebcamMaxDim();
    const file = await captureVideoFrameFile(maxDim);
    const result = await inferOne(file, selModel(), selCmap(), selDevice(), wc.abort.signal, "fast", "color", null, false, maxDim);
    const e2e = performance.now() - wc.lastLoopStartedAt;
    wc.lastResult = result;
    wc.processed++;
    wc.consecutiveErrors = 0;
    if (Number.isFinite(Number(result.latency_ms))) wc.latencies.push(Number(result.latency_ms));
    wc.e2eLatencies.push(e2e);
    wc.latencies = wc.latencies.slice(-60);
    wc.e2eLatencies = wc.e2eLatencies.slice(-60);
    await setWebcamDepthPreview(result);
    setWebcamStatus("Running", "Scheduled");
    if (wc.processed % 10 === 0 || performance.now() - wc.lastCacheMetricsAt > 30_000) {
      wc.lastCacheMetricsAt = performance.now();
      loadCacheMetrics({ signal: timeoutSignal(8000) }).catch(() => {});
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      wc.errors++;
      wc.consecutiveErrors++;
      wc.skipped++;
      setWebcamStatus("Running", "Inference error");
      appendWebcamLog(`Inference error: ${err.message}`, "error");
      toastOnce(`Webcam inference failed: ${err.message}`, "error", 6000);
      if (wc.consecutiveErrors >= 5) {
        wc.paused = true;
        setWebcamStatus("Running", "Auto-paused after errors");
        toast("Webcam inference auto-paused after 5 consecutive errors.", "error", 8000);
      }
    }
  } finally {
    wc.inFlight = false;
    wc.abort = null;
    updateWebcamTelemetry();
    scheduleNextWebcamFrame();
  }
}
function captureVideoFrameFile(maxDim) {
  return new Promise((resolve, reject) => {
    const video = el.webcamVideo;
    const canvas = el.webcamCaptureCanvas;
    if (!video || !canvas) { reject(new Error("Webcam capture elements are missing")); return; }
    const vw = video.videoWidth || 0, vh = video.videoHeight || 0;
    if (!vw || !vh) { reject(new Error("Webcam metadata is unavailable")); return; }
    const scale = Math.min(1, maxDim / Math.max(vw, vh));
    canvas.width = Math.max(1, Math.round(vw * scale));
    canvas.height = Math.max(1, Math.round(vh * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) { reject(new Error("Canvas 2D context is unavailable")); return; }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
      if (!blob) { reject(new Error("Unable to encode webcam frame")); return; }
      resolve(new File([blob], `webcam-frame-${Date.now()}.jpg`, { type: "image/jpeg" }));
    }, "image/jpeg", 0.75);
  });
}
function loadImageForCanvas(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Unable to decode depth image"));
    img.src = src;
    if (img.decode) img.decode().then(() => resolve(img)).catch(() => {});
  });
}
async function applyDepthSmoothing(rawDataUrl, alpha) {
  if (!alpha || !state.webcam.previousDepthImageData) {
    const first = await imageDataFromUrl(rawDataUrl);
    state.webcam.previousDepthImageData = first;
    return rawDataUrl;
  }
  const current = await imageDataFromUrl(rawDataUrl);
  const previous = state.webcam.previousDepthImageData;
  if (previous.width !== current.width || previous.height !== current.height) {
    state.webcam.previousDepthImageData = current;
    return rawDataUrl;
  }
  const out = new ImageData(current.width, current.height);
  for (let i = 0; i < current.data.length; i += 4) {
    out.data[i] = previous.data[i] * alpha + current.data[i] * (1 - alpha);
    out.data[i+1] = previous.data[i+1] * alpha + current.data[i+1] * (1 - alpha);
    out.data[i+2] = previous.data[i+2] * alpha + current.data[i+2] * (1 - alpha);
    out.data[i+3] = 255;
  }
  state.webcam.previousDepthImageData = out;
  const canvas = document.createElement("canvas");
  canvas.width = out.width; canvas.height = out.height;
  canvas.getContext("2d").putImageData(out, 0, 0);
  return canvas.toDataURL("image/png");
}
async function imageDataFromUrl(src) {
  const img = await loadImageForCanvas(src);
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext("2d");
  if (!ctx || !canvas.width || !canvas.height) throw new Error("Unable to read depth pixels");
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}
async function setWebcamDepthPreview(result) {
  const rawDataUrl = safeDataImagePng(result?.depth_map);
  if (!rawDataUrl) throw new Error("Backend response did not include a depth map");
  state.webcam.lastDepthBase64 = result.depth_map;
  const alpha = getWebcamSmoothingAlpha();
  state.webcam.smoothingAlpha = alpha;
  let displayUrl = rawDataUrl;
  if (alpha > 0) {
    try { displayUrl = await applyDepthSmoothing(rawDataUrl, alpha); }
    catch (err) { console.debug(`[DepthLens] Webcam smoothing skipped: ${err.message}`); displayUrl = rawDataUrl; }
  } else {
    state.webcam.previousDepthImageData = null;
  }
  state.webcam.lastDepthDataUrl = displayUrl;
  if (el.webcamDepthImg) el.webcamDepthImg.src = displayUrl;
  if (el.webcamDepthPlaceholder) el.webcamDepthPlaceholder.hidden = true;
  syncWebcamControls();
}
function webcamTimestamp() {
  const d = new Date();
  const pad = n => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}
function downloadWebcamDepth() {
  const name = `webcam_depth_${webcamTimestamp()}.png`;
  if (state.webcam.lastDepthBase64) { dlB64(name, state.webcam.lastDepthBase64); return; }
  if (!state.webcam.lastDepthDataUrl) return;
  const a = Object.assign(document.createElement("a"), { href: state.webcam.lastDepthDataUrl, download: name });
  document.body.appendChild(a); a.click(); a.remove();
}
function handleWebcamVisibilityChange() {
  if (!state.webcam.running) return;
  if (document.hidden) {
    state.webcam.hiddenPaused = true;
    setWebcamStatus("Running", "Paused while hidden");
  } else if (state.webcam.hiddenPaused) {
    state.webcam.hiddenPaused = false;
    setWebcamStatus("Running", state.webcam.paused ? "Paused" : "Scheduled");
    scheduleNextWebcamFrame(0);
  }
  updateWebcamTelemetry();
}
el.webcamStartBtn?.addEventListener("click", startWebcam);
el.webcamStopBtn?.addEventListener("click", () => stopWebcam());
el.webcamPauseBtn?.addEventListener("click", toggleWebcamPause);
el.webcamCaptureBtn?.addEventListener("click", downloadWebcamDepth);
el.webcamTargetFps?.addEventListener("change", () => { appendWebcamLog(`Target FPS set to ${getWebcamTargetFps()}`); updateWebcamTelemetry(); scheduleNextWebcamFrame(); });
el.webcamMaxDim?.addEventListener("change", () => { appendWebcamLog(`Frame max dimension set to ${getWebcamMaxDim()}px`); updateWebcamTelemetry(); });
el.webcamSmoothing?.addEventListener("change", () => { state.webcam.smoothingAlpha = getWebcamSmoothingAlpha(); state.webcam.previousDepthImageData = null; appendWebcamLog(`Smoothing alpha set to ${state.webcam.smoothingAlpha}`); updateWebcamTelemetry(); });


// ══════════════════════════════════════════════════════════════
// METRICS DASHBOARD
// ══════════════════════════════════════════════════════════════
function updateMetrics() {
  const s=state.session, lats=s.latencies, cache=state.cacheMetrics;
  el.metricTotal.textContent=s.total;
  el.metricCached.textContent=cache ? cache.totalHits : s.cached;
  if (cache) {
    const cacheCell = el.metricCached.closest(".metric-cell");
    if (cacheCell) {
      cacheCell.title = `Cache hits: ${cache.totalHits} · Misses: ${cache.cacheMisses} · Keys: ${cache.keyspaceSize} · Backend: ${cache.backend} · TTL: ${cache.ttlSeconds}s`;
    }
  }
  el.metricErrors.textContent=s.errors;
  el.metricTotalTime.textContent=`${(s.totalInferenceMs/1000).toFixed(1)} s`;
  if (lats.length) {
    const avg=lats.reduce((a,b)=>a+b,0)/lats.length;
    el.metricAvgLatency.textContent=avg.toFixed(0);
    el.metricMinLat.textContent=Math.min(...lats).toFixed(0);
    el.metricMaxLat.textContent=Math.max(...lats).toFixed(0);
    el.metricThroughput.textContent=(60000/avg).toFixed(1);
  }
}

// ══════════════════════════════════════════════════════════════
// GALLERY
// ══════════════════════════════════════════════════════════════
function appendGalleryItem(r) {
  const m=r.metrics||{};
  const item=Object.assign(document.createElement("div"),{className:"gallery-item"});
  item.setAttribute("role","listitem"); item.setAttribute("tabindex","0");
  item.innerHTML = `
    <div class="gallery-img-wrap">
      <img src="${safeDataImagePng(r.depth_map)}" alt="Depth map — ${esc(r.filename)}" loading="lazy"/>
      <div class="gallery-overlay">◍</div>
    </div>
    <div class="gallery-meta">
      <div class="gallery-filename" title="${esc(r.filename)}">${esc(r.filename)}</div>
      <div class="gallery-tags">
        <span class="gallery-tag">${esc(r.model?.replace("MiDaS_","").replace("DPT_","DPT "))}</span>
        <span class="gallery-tag">${esc(r.colormap)}</span>
        <span class="gallery-tag">${esc(r.device_used||"")}</span>
        ${r.cached?'<span class="gallery-tag">cached</span>':""}
      </div>
      <div class="gallery-stats-row">
        <span>Latency <strong>${esc(Number.isFinite(Number(r.latency_ms)) ? `${Number(r.latency_ms)}ms` : "—")}</strong></span>
        <span>Proxy <strong>${Number.isFinite(Number(m.ssim)) ? Number(m.ssim).toFixed(3) : "—"}</strong></span>
        <span>${esc(r.resolution?.width ?? "?")}×${esc(r.resolution?.height ?? "?")}</span>
      </div>
    </div>`;
  item.addEventListener("click",()=>openLightbox(r));
  item.addEventListener("keydown",e=>{ if (e.key==="Enter"||e.key===" ") openLightbox(r); });
  el.gallery.appendChild(item);
}

el.clearResultsBtn?.addEventListener("click",()=>{ state.results=[]; el.gallery.innerHTML=""; el.resultsCard.hidden=true; });
el.downloadAllBtn?.addEventListener("click",()=>{
  if (!state.results.length) return;
  state.results.forEach(r=>dlB64(`depth_${r.filename}`,r.depth_map));
  toast(`Downloading ${state.results.length} depth maps…`);
});

// ══════════════════════════════════════════════════════════════
// LIGHTBOX METRICS ACCORDION
// ══════════════════════════════════════════════════════════════
const METRIC_GROUPS = [
  { id:"error", icon:"⨯", label:"Core Error Metrics",
    note:"Computed from predicted depth distribution only (no ground truth required). Values reflect self-consistency of the depth map.",
    metrics:[
      {key:"mae",label:"Mean Absolute Deviation from Predicted Mean",unit:"",desc:"Prediction-only mean absolute deviation from the predicted depth mean. Proxy statistic, not GT MAE.",needsGT:false},
      {key:"rmse",label:"RMS Deviation from Predicted Mean",unit:"",desc:"Prediction-only RMS deviation from the predicted depth mean. Proxy statistic, not GT RMSE.",needsGT:false},
      {key:"log_rmse",label:"Log-Depth Deviation",unit:"",desc:"Prediction-only log-depth deviation around the predicted mean; not benchmark Log RMSE.",needsGT:false},
      {key:"abs_rel",label:"Absolute Relative Error (Abs Rel)",unit:"",desc:"Mean of |pred−GT|/GT. Standard MDE benchmark metric. Requires ground-truth depth.",needsGT:true},
      {key:"sq_rel",label:"Squared Relative Error (Sq Rel)",unit:"",desc:"Mean of (pred−GT)²/GT. Penalises large relative errors. Requires ground-truth depth.",needsGT:true},
      {key:"gt_mae",label:"True MAE vs GT",unit:"",desc:"Mean absolute error after median-scale alignment to valid GT pixels.",needsGT:true},
      {key:"gt_rmse",label:"True RMSE vs GT",unit:"",desc:"Root mean squared error after median-scale alignment to valid GT pixels.",needsGT:true},
      {key:"gt_log_rmse",label:"True Log RMSE vs GT",unit:"",desc:"Log-depth RMSE after median-scale alignment to valid GT pixels.",needsGT:true},
    ]},
  { id:"accuracy", icon:"◔", label:"Threshold Accuracy",
    note:"δ metrics require ground-truth depth maps and cannot be computed here.",
    metrics:[
      {key:"delta_1",label:"δ < 1.25¹",unit:"%",desc:"Fraction of pixels within 25% scale of ground truth. Requires GT.",needsGT:true},
      {key:"delta_2",label:"δ < 1.25²",unit:"%",desc:"Looser threshold (56%). Requires GT.",needsGT:true},
      {key:"delta_3",label:"δ < 1.25³",unit:"%",desc:"Loosest threshold (95%). Requires GT.",needsGT:true},
    ]},
  { id:"scaleinv", icon:"⇲", label:"Scale-Invariant Metrics",
    metrics:[
      {key:"silog",label:"Log-Depth Dispersion Proxy",unit:"",desc:"Prediction-only log-depth dispersion proxy. True SILog requires GT.",needsGT:false},
      {key:"dynamic_range",label:"Dynamic Range",unit:" bits",desc:"Log₂ ratio of max/min non-zero depth. Larger = more depth variation captured.",needsGT:false},
      {key:"entropy",label:"Shannon Entropy",unit:" bits",desc:"Entropy of the depth histogram. Higher = more uniformly distributed depth values.",needsGT:false},
      {key:"coverage",label:"Depth Coverage",unit:"%",desc:"Fraction of histogram bins with ≥1% of peak count. Higher = depth values spread across the full range.",needsGT:false,pct:true},
    ]},
  { id:"structural", icon:"◬", label:"Structural & Geometric Metrics",
    metrics:[
      {key:"ssim",label:"RGB–Depth Structural Proxy",unit:"",desc:"Proxy comparing predicted depth structure to grayscale RGB input; not reference SSIM.",needsGT:false},
      {key:"gradient_mean",label:"Gradient Mean",unit:"",desc:"Mean Sobel gradient magnitude over the depth map. Higher = more depth edges/transitions.",needsGT:false},
      {key:"gradient_std",label:"Gradient Std Dev",unit:"",desc:"Variation in gradient strength. High std means some regions have sharp edges while others are smooth.",needsGT:false},
      {key:"gradient_error",label:"Depth Edge Proxy",unit:"",desc:"Prediction-only edge-detail proxy equal to mean depth gradient; not GT gradient error.",needsGT:false},
      {key:"edge_density",label:"Edge Density",unit:"%",desc:"Fraction of pixels with gradient > mean+std. Indicates how richly detailed the depth edges are.",needsGT:false,pct:true},
      {key:"surface_normal_error",label:"Surface Normal Error",unit:"",desc:"Requires ground-truth normals derived from GT depth. Not computable without GT.",needsGT:true},
    ]},
  { id:"perceptual", icon:"◉", label:"Perceptual & Consistency Metrics",
    metrics:[
      {key:"psnr",label:"Depth Variance PSNR Proxy",unit:" dB",desc:"Prediction-only PSNR-like variance proxy; true PSNR requires a reference depth map.",needsGT:false},
      {key:"lpips",label:"LPIPS (Perceptual Similarity)",unit:"",desc:"Learned perceptual metric. Requires a reference depth map. Not computable without GT.",needsGT:true},
    ]},
  { id:"ranking", icon:"≋", label:"Ranking / Relative Depth Metrics",
    metrics:[
      {key:"ordinal_error",label:"Ordinal Error",unit:"",desc:"Fraction of pixel pairs where relative ordering of pred depth disagrees with GT. Requires GT.",needsGT:true},
    ]},
];

function valColor(key,val) {
  if (val===null||val===undefined) return "na";
  if (key==="ssim")  return val>0.7?"good":val>0.4?"warn":"bad";
  if (key==="silog") return val<10?"good":val<25?"warn":"bad";
  if (key==="psnr")  return val>30?"good":val>15?"warn":"bad";
  return "";
}

function metricUnavailableReason(metrics, key, mode, needsGT) {
  const unavailable = metrics?.unavailable || {};
  if (unavailable[key] === "not_requested_fast_mode") return "Not requested in fast mode";
  if (unavailable[key] === "needs_gt_depth_upload" || needsGT) return "Needs GT depth upload";
  if (unavailable[key] === "not_implemented") return "Not implemented yet";
  if (unavailable[key]) return String(unavailable[key]).replace(/_/g," ");
  if (mode === "fast") return "Not requested in fast mode";
  return "—";
}

function renderMetricsAccordion(resultOrMetrics) {
  const metrics = resultOrMetrics?.metrics || resultOrMetrics || {};
  const mode = resultOrMetrics?.metrics_mode || "full";
  const hasGt = Boolean(resultOrMetrics?.gt_metadata?.provided);
  el.lightboxMetrics.innerHTML = "";
  METRIC_GROUPS.forEach((group,gi) => {
    const div=document.createElement("div");
    div.className="metric-group"; div.id=`mg-${group.id}`;
    const hdr=document.createElement("div");
    hdr.className="metric-group-header"; hdr.setAttribute("role","button");
    hdr.setAttribute("tabindex","0"); hdr.setAttribute("aria-expanded","false");
    hdr.innerHTML=`<span><span class="mg-icon">${esc(group.icon)}</span>${esc(group.label)}</span><span class="mg-toggle">▾</span>`;
    hdr.addEventListener("click",()=>toggleAccordion(div));
    hdr.addEventListener("keydown",e=>{ if (e.key==="Enter"||e.key===" "){e.preventDefault();toggleAccordion(div);} });
    const body=document.createElement("div");
    body.className="metric-group-body"; body.setAttribute("role","region");
    const content=document.createElement("div"); content.className="metric-group-content";
    if (group.note) {
      const noteEl=document.createElement("p");
      noteEl.style.cssText="font-family:var(--ff-mono);font-size:.6rem;color:var(--text-dim);margin-bottom:.4rem;line-height:1.4;";
      noteEl.textContent=group.note; content.appendChild(noteEl);
    }
    const seen=new Set();
    group.metrics.forEach(m => {
      if (seen.has(m.key)) return; seen.add(m.key);
      const raw=metrics?.[m.key];
      const isNull=raw===null||raw===undefined;
      let valText,cls;
      if (m.needsGT && isNull) { valText=hasGt ? metricUnavailableReason(metrics,m.key,mode,true) : "Needs GT depth upload"; cls="na"; }
      else if (isNull)         { valText=metricUnavailableReason(metrics,m.key,mode,false); cls="na"; }
      else if (m.pct)          { valText=`${(raw*100).toFixed(1)}%`; cls=valColor(m.key,raw); }
      else                     { valText=`${raw}${m.unit||""}`; cls=valColor(m.key,raw); }
      const badge = m.needsGT ? '<span class="metric-badge gt">GT</span>' : (metrics?.proxy_metrics && Object.prototype.hasOwnProperty.call(metrics.proxy_metrics,m.key) ? '<span class="metric-badge">Proxy</span>' : '');
      const row=document.createElement("div"); row.className="metric-row";
      row.innerHTML=`
        <div class="metric-row-left">
          <span class="metric-row-name">${esc(m.label)}${badge}</span>
          <span class="metric-row-desc">${esc(m.desc)}</span>
        </div>
        <span class="metric-row-val ${esc(cls)}">${esc(valText)}</span>`;
      content.appendChild(row);
    });
    body.appendChild(content); div.appendChild(hdr); div.appendChild(body);
    el.lightboxMetrics.appendChild(div);
    if (gi===0) toggleAccordion(div);
  });
}

function toggleAccordion(div) {
  const isOpen=div.classList.contains("open");
  div.classList.toggle("open",!isOpen);
  div.querySelector(".metric-group-header").setAttribute("aria-expanded",String(!isOpen));
}

function updateBlendPreview() {
  const v=Number(el.lbSlider.value||50)/100;
  el.lbOrigImg.style.opacity=(1-v*0.85).toFixed(2);
  el.lbDepthImg.style.opacity=(0.2+v*0.8).toFixed(2);
  if (el.lbRangeValue) el.lbRangeValue.textContent=`${Math.round(v*100)}%`;
}

let lightboxCloseTimer=null, lightboxTransitionCleanup=null, bodyScrollY=0;
function lockBodyScroll() {
  bodyScrollY=window.scrollY||window.pageYOffset||0;
  document.body.classList.add("modal-open");
  document.body.style.top=`-${bodyScrollY}px`;
}
function unlockBodyScroll() {
  document.body.classList.remove("modal-open");
  document.body.style.top="";
  window.scrollTo(0,bodyScrollY);
}

function openLightbox(r) {
  if (lightboxTransitionCleanup) { lightboxTransitionCleanup(); lightboxTransitionCleanup=null; }
  state.lb.current=r;
  el.lbOrigImg.src=r.originalSrc||"";
  el.lbDepthImg.src=safeDataImagePng(r.depth_map);
  el.lbSlider.value=50; updateBlendPreview();
  el.lbTags.innerHTML=[
    r.model?.replace("MiDaS_","").replace("DPT_","DPT "),
    r.colormap, r.device_used, `${r.latency_ms} ms`,
    r.cached?"cached":null, `${r.resolution?.width}×${r.resolution?.height}`,
  ].filter(Boolean).map(t=>`<span class="lb-tag">${esc(t)}</span>`).join("");
  renderMetricsAccordion(r);
  if (lightboxCloseTimer) { clearTimeout(lightboxCloseTimer); lightboxCloseTimer=null; }
  el.lightboxBackdrop.hidden=false;
  el.lightboxBackdrop.classList.remove("is-closing","is-open");
  el.lightboxBackdrop.classList.add("is-mounted");
  requestAnimationFrame(()=>{ requestAnimationFrame(()=>{ el.lightboxBackdrop.classList.add("is-open"); }); });
  el.lightboxMetrics.scrollTop=0; lockBodyScroll(); el.lightboxClose.focus();
}

function closeLightbox() {
  if (el.lightboxBackdrop.hidden) return;
  if (el.lightboxBackdrop.classList.contains("is-closing")) return;
  if (lightboxTransitionCleanup) { lightboxTransitionCleanup(); lightboxTransitionCleanup=null; }
  el.lightboxBackdrop.classList.remove("is-open");
  el.lightboxBackdrop.classList.add("is-closing");
  let finalized = false;
  const finalize=()=>{
    if (finalized) return;
    finalized = true;
    if (lightboxCloseTimer) { clearTimeout(lightboxCloseTimer); lightboxCloseTimer=null; }
    if (lightboxTransitionCleanup) { lightboxTransitionCleanup(); lightboxTransitionCleanup=null; }
    el.lightboxBackdrop.hidden=true;
    el.lightboxBackdrop.classList.remove("is-mounted","is-closing");
    unlockBodyScroll(); state.lb.current=null;
  };
  const onDone=(ev)=>{
    if (ev.target!==el.lightboxBackdrop) return;
    finalize();
  };
  lightboxTransitionCleanup = () => el.lightboxBackdrop.removeEventListener("transitionend",onDone);
  el.lightboxBackdrop?.addEventListener("transitionend",onDone, { once: true });
  lightboxCloseTimer=setTimeout(finalize,420);
}

el.lightboxClose?.addEventListener("click",closeLightbox);
el.lightboxBackdrop?.addEventListener("click",e=>{ if (e.target===el.lightboxBackdrop) closeLightbox(); });
document.addEventListener("keydown",e=>{ if (e.key==="Escape") closeLightbox(); });
el.lbSlider?.addEventListener("input",()=>{ updateBlendPreview(); });
el.lbDlDepth?.addEventListener("click",()=>{ const r=state.lb.current; if (r) dlB64(`depth_${r.filename}`,r.depth_map); });
el.lbDlGray?.addEventListener("click",()=>{ const r=state.lb.current; if (r) dlB64(`gray_${r.filename}`,r.grayscale); });

// ══════════════════════════════════════════════════════════════
// COMPARE PANEL
// ══════════════════════════════════════════════════════════════
el.compareFileInput?.addEventListener("change",()=>{
  state.compareFile=el.compareFileInput.files[0];
  if (state.compareFile) { el.compareFileName.textContent=state.compareFile.name; el.compareRunBtn.disabled=false; toast(`Loaded: ${state.compareFile.name}`); }
});
el.compareDropZone?.addEventListener("dragover",e=>{ e.preventDefault(); el.compareDropZone.classList.add("drag-over"); });
el.compareDropZone?.addEventListener("dragleave",e=>{ if (!el.compareDropZone.contains(e.relatedTarget)) el.compareDropZone.classList.remove("drag-over"); });
el.compareDropZone?.addEventListener("drop",e=>{
  e.preventDefault(); el.compareDropZone.classList.remove("drag-over");
  const f=e.dataTransfer.files[0];
  if (f?.type.startsWith("image/")) { state.compareFile=f; el.compareFileName.textContent=f.name; el.compareRunBtn.disabled=false; toast(`Loaded: ${f.name}`); }
});
el.compareDropZone?.addEventListener("click",()=>el.compareFileInput.click());
el.compareRunBtn?.addEventListener("click",runComparison);
el.compareCancelBtn?.addEventListener("click",()=>{ state.compareAbort?.abort(); toast("Comparison cancelled","warning"); });

async function runComparison() {
  if (!state.compareFile) return;
  const cmap=el.compareCmap.value, device=el.compareDevice.value;
  const models=["midas_small","dpt_hybrid","dpt_large"];
  state.compareAbort=new AbortController();
  el.compareRunBtn.disabled=true; el.compareCancelBtn.hidden=false;
  state.compareView.results=[];
  el.compareProgressBlock.hidden=false; el.compareResults.innerHTML="";
  el.compareChartCard.hidden=true; el.compareMetricGrid.innerHTML="";
  if (!engineReady()) {
    const ok = await checkHealth();
    if (!ok) {
      toast(`Depth engine is unavailable at ${API || DEFAULT_API_BASE_URL}`, "error", 6000);
      el.compareRunBtn.disabled=false; el.compareCancelBtn.hidden=true; state.compareAbort=null;
      return;
    }
  }
  const results=[], t0=Date.now();
  try {
  for (let i=0;i<models.length;i++) {
    if (state.compareAbort.signal.aborted) break;
    const model=models[i];
    const estCurrent=getEstimate("compare",model,device);
    const estRemainingStatic=models.slice(i+1).reduce((acc,m)=>acc+getEstimate("compare",m,device),0);
    const modelStart=Date.now();
    const tick=setInterval(()=>{
      const elapsedCurrent=Date.now()-modelStart;
      const unitDone=i+Math.min(elapsedCurrent/estCurrent,0.98);
      const pct=Math.min((unitDone/models.length)*100,99);
      const rem=Math.max(0,estCurrent-elapsedCurrent+estRemainingStatic);
      el.compareProgressFill.style.width=`${pct}%`;
      el.compareProgressPct.textContent=`${Math.round(pct)}%`;
      el.compareProgressText.textContent=`Running ${model}…`;
      el.compareProgressEta.textContent=`ETA: ${fmtDuration(rem)}`;
    },120);
    try {
      const r=await inferOne(state.compareFile,model,cmap,device,state.compareAbort.signal,"full","color,gray");
      clearInterval(tick); updateEstimate("compare",model,device,r.latency_ms);
      results.push(r); renderCompareCard(r);
    } catch(err) {
      clearInterval(tick);
      if (err.name!=="AbortError") toast(`${model} failed: ${err.message}`,"error");
    }
  }
  el.compareProgressFill.style.width="100%"; el.compareProgressPct.textContent="100%";
  el.compareProgressText.textContent=state.compareAbort.signal.aborted?"Cancelled":"Done!";
  el.compareProgressEta.textContent=`Total: ${fmtDuration(Date.now()-t0)}`;
  if (results.length) {
    state.compareView.results=results;
    renderCompareSummary(results);
    renderCompareChart(results,state.compareView.metricKey);
    toast("Comparison complete!","success");
  }
  } catch (err) {
    el.compareProgressText.textContent="Comparison failed";
    el.compareProgressEta.textContent=err.message || "Inference failed";
    toast(`Comparison failed: ${err.message}`, "error", 6000);
  } finally {
    setTimeout(()=>{ el.compareProgressBlock.hidden=true; },2500);
    el.compareRunBtn.disabled=false; el.compareCancelBtn.hidden=true; state.compareAbort=null;
  }
}

function renderCompareCard(r) {
  $(".compare-placeholder")?.remove();
  const card=document.createElement("div"); card.className="compare-card";
  const lbl=esc(r.model_display_name || r.model?.replace("midas_","MiDaS ").replace("dpt_","DPT ").replace("MiDaS_","").replace("DPT_","DPT ") || "Model");
  const latency = Number.isFinite(Number(r.latency_ms)) ? `${Number(r.latency_ms)} ms` : "—";
  const warning = r.fallback_used ? `<div class="compare-warning">ONNX unavailable · ${esc(r.engine_used || "PyTorch")} fallback · ${esc(r.device_used || "")}</div>` : `<div class="compare-warning">${esc(r.engine_used || "")} · ${esc(r.device_used || "")}</div>`;
  card.innerHTML=`
    <div class="compare-card-header">${lbl} <span class="latency-badge">${esc(latency)}</span></div>
    ${warning}
    <img src="${safeDataImagePng(r.depth_map)}" alt="Depth map — ${lbl}" loading="lazy"/>`;
  el.compareResults.appendChild(card);
}


// ══════════════════════════════════════════════════════════════
// BENCHMARK PANEL
// ══════════════════════════════════════════════════════════════
function benchmarkResult(data, engine) {
  return (data?.results || []).find(r => r.engine === engine) || {};
}

function fmtBenchLatency(result) {
  const avg = result?.latency_ms?.avg;
  return Number.isFinite(Number(avg)) ? `${Number(avg).toFixed(1)} ms` : "—";
}

function renderBenchmarkChart(data) {
  const results = data?.results || [];
  const values = results.map(r => Number(r?.latency_ms?.avg));
  const c = chartColors();
  const ctx = $("#benchmarkChart")?.getContext("2d");
  if (!ctx) return;
  const labels = results.map(r => r.engine === "onnxruntime" ? "ONNX Runtime" : "PyTorch");
  const dataValues = values.map(v => Number.isFinite(v) ? v : null);
  if (benchmarkChart) {
    benchmarkChart.data.labels = labels;
    benchmarkChart.data.datasets[0].data = dataValues;
    benchmarkChart.data.datasets[0].backgroundColor = dataValues.map(v => v === null ? "rgba(127,140,153,.45)" : c.bar);
    benchmarkChart.data.datasets[0].borderColor = dataValues.map(v => v === null ? "#5e6f81" : c.barBrd);
    applyChartPalette(benchmarkChart, c);
    benchmarkChart.update("none");
    return;
  }
  benchmarkChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Average latency (ms)",
        data: dataValues,
        backgroundColor: dataValues.map(v => v === null ? "rgba(127,140,153,.45)" : c.bar),
        borderColor: dataValues.map(v => v === null ? "#5e6f81" : c.barBrd),
        borderWidth: 1.5,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: c.ttTitle, font: { family:"JetBrains Mono", size:10 } } },
        tooltip: {
          backgroundColor: c.tooltip, borderColor: c.ttBrd, borderWidth: 1,
          titleColor: c.ttTitle, bodyColor: c.ttBody,
          callbacks: { label: ctx => ctx.raw === null ? "Unavailable" : `${Number(ctx.raw).toFixed(1)} ms` },
        },
      },
      scales: {
        x: { ticks: { color: c.ttTitle, font: { family:"Rajdhani", size:12, weight:"600" } }, grid: { color: c.grid } },
        y: { ticks: { color: c.tick, font: { family:"JetBrains Mono", size:9 } }, grid: { color: c.grid } },
      },
    },
  });
}

function renderBenchmark(data) {
  const torch = benchmarkResult(data, "pytorch");
  const onnx = benchmarkResult(data, "onnxruntime");
  el.benchTorchLatency.textContent = fmtBenchLatency(torch);
  el.benchOnnxLatency.textContent = fmtBenchLatency(onnx);
  el.benchSpeedup.textContent = data?.comparison?.speedup ? `${data.comparison.speedup}×` : "—";
  const onnxThroughput = onnx?.throughput_fps ?? data?.onnx?.throughput_fps;
  el.benchThroughput.textContent = Number.isFinite(Number(onnxThroughput)) ? `${Number(onnxThroughput).toFixed(2)} fps` : "Unavailable";
  el.benchMemory.textContent = data?.memory_snapshot?.process_rss_mb ? `${data.memory_snapshot.process_rss_mb} MB` : "—";
  const provider = onnx?.provider || onnx?.diagnostics?.runtime?.selected_provider || data?.onnx_diagnostics?.runtime?.selected_provider;
  const cpuFallback = onnx?.uses_cpu_fallback || onnx?.diagnostics?.runtime?.uses_cpu_fallback || data?.onnx_diagnostics?.runtime?.uses_cpu_fallback;
  const onnxStatus = data?.onnx?.status || onnx?.state || "unavailable";
  el.benchProvider.textContent = onnx?.status === "ok" || data?.onnx?.status === "ok" ? `ONNX Ready${provider ? ` · ${provider}` : ""}${cpuFallback ? " · CPU fallback" : ""}` : `ONNX ${onnxStatus}`;
  const diag = data?.onnx_diagnostics || onnx?.diagnostics || {};
  el.benchStatus.textContent = onnx?.status === "ok" || data?.onnx?.status === "ok" ? `Weights: ${data.weights?.onnx_path || data?.onnx?.onnx_path}` : `${data?.onnx?.message || onnx?.reason || "PyTorch benchmark completed; ONNX unavailable"}${diag.expected_path ? ` · expected: ${diag.expected_path}` : ""}${diag.recommended_export_command ? ` · run: ${diag.recommended_export_command}` : ""}`;
  renderBenchmarkChart(data);
}

async function runBenchmark() {
  if (!el.benchmarkRunBtn || el.benchmarkRunBtn.disabled) return;
  state.benchmarkAbort?.abort();
  state.benchmarkAbort = new AbortController();
  const model = el.benchmarkModel?.value || "MiDaS_small";
  const device = el.benchmarkDevice?.value || "auto";
  el.benchmarkRunBtn.disabled = true;
  el.benchmarkRunBtn.textContent = "Running…";
  el.benchStatus.textContent = "Loading models and measuring latency…";
  setStatus("online", "Depth Engine: Busy", "Benchmark running · /live remains available", deviceBadge(state.devices, state.primaryDevice));
  try {
    if (!engineReady()) {
      const ok = await checkHealth();
      if (!ok) throw new Error(`Depth engine is unavailable at ${API || DEFAULT_API_BASE_URL}`);
    }
    const res = await apiFetch(`/api/benchmark?model=${encodeURIComponent(model)}&device=${encodeURIComponent(device)}&iterations=3`, {
      signal: anySignal([state.benchmarkAbort.signal, timeoutSignal(240_000)]),
    });
    renderBenchmark(await res.json());
    toast("Benchmark complete","success");
  } catch (err) {
    el.benchProvider.textContent = "Failed";
    el.benchStatus.textContent = err.message;
    checkLive({ quiet: true }).catch(() => {});
    if (err.name !== "AbortError") toastOnce(`Benchmark failed: ${err.message}`,"error");
  } finally {
    el.benchmarkRunBtn.disabled = false;
    el.benchmarkRunBtn.textContent = "Run Benchmark";
    state.benchmarkAbort = null;
  }
}

el.benchmarkRunBtn?.addEventListener("click", () => {
  clearTimeout(window.__depthlensBenchmarkTimer);
  window.__depthlensBenchmarkTimer = setTimeout(runBenchmark, 250);
});


// ══════════════════════════════════════════════════════════════
// EXPERIMENT WORKSPACE
// ══════════════════════════════════════════════════════════════
function scalarMetric(result, key) {
  const metrics = result?.metrics || {};
  return metrics[key] ?? metrics.gt_metrics?.[key] ?? metrics.proxy_metrics?.[key] ?? metrics.prediction_stats?.[key] ?? null;
}
function experimentRows() {
  return state.experiment.results.map((r, i) => ({
    index: i + 1,
    filename: r.filename,
    model: r.model,
    device: r.device_used,
    latency_ms: r.latency_ms,
    engine: r.engine_used || "pytorch",
    fallback: Boolean(r.fallback_used),
    abs_rel: scalarMetric(r, "abs_rel"),
    rmse: scalarMetric(r, "gt_rmse") ?? scalarMetric(r, "rmse"),
    delta_1: scalarMetric(r, "delta_1"),
    gt: Boolean(r.gt_metadata?.provided),
    warnings: [...(r.warnings || []), ...(r.metrics?.warnings || []), ...(r.gt_metadata?.warnings || [])].join(" | "),
  }));
}
function renderExperiment() {
  const rows = experimentRows();
  el.experimentCount.textContent = rows.length;
  const latencies = rows.map(r => Number(r.latency_ms)).filter(Number.isFinite);
  el.experimentAvgLatency.textContent = latencies.length ? `${(latencies.reduce((a,b)=>a+b,0)/latencies.length).toFixed(0)} ms` : "—";
  const abs = rows.map(r => Number(r.abs_rel)).filter(Number.isFinite);
  el.experimentBestAbsRel.textContent = abs.length ? Math.min(...abs).toFixed(4) : "—";
  el.experimentStatusMetric.textContent = rows.length ? "Ready" : "Idle";
  el.experimentExportJsonBtn.disabled = !rows.length;
  el.experimentExportCsvBtn.disabled = !rows.length;
  el.experimentTableBody.innerHTML = rows.length ? rows.map(r => `
    <tr><td>${esc(r.filename)}</td><td>${esc(r.model)}</td><td>${esc(r.engine)} · ${esc(r.device)}</td><td>${esc(r.latency_ms)} ms</td><td>${esc(r.abs_rel ?? "Requires GT")}</td><td>${esc(r.rmse ?? "Requires GT")}</td><td>${esc(r.delta_1 ?? "Requires GT")}</td><td>${r.gt ? "GT" : "Image-only"}${r.warnings ? ` · ${esc(r.warnings)}` : ""}</td></tr>
  `).join("") : '<tr><td colspan="8">No experiment results yet.</td></tr>';
  el.experimentPreviews.innerHTML = state.experiment.results.map(r => `
    <article class="experiment-card">
      <div class="experiment-card-head"><span>${esc(r.filename)}</span><span>${esc(r.latency_ms)} ms</span></div>
      <div class="experiment-preview-grid">
        <div class="experiment-preview-tile"><img src="${String(r.originalSrc || '').startsWith('data:image/') ? esc(r.originalSrc) : ''}" alt="RGB input"><span>RGB</span></div>
        <div class="experiment-preview-tile"><img src="${safeDataImagePng(r.depth_map)}" alt="Predicted depth"><span>Predicted</span></div>
        ${r.gt_depth_map ? `<div class="experiment-preview-tile"><img src="${safeDataImagePng(r.gt_depth_map)}" alt="GT depth"><span>GT</span></div>` : `<div class="experiment-preview-tile"><span>GT unavailable</span></div>`}
        ${r.error_heatmap ? `<div class="experiment-preview-tile"><img src="${safeDataImagePng(r.error_heatmap)}" alt="Error heatmap"><span>Error</span></div>` : `<div class="experiment-preview-tile"><span>Error map unavailable</span></div>`}
      </div>
    </article>
  `).join("");
}
async function runExperiment() {
  if (!engineReady()) {
    const ok = await checkHealth();
    if (!ok) { toast("Inference runtime is not ready", "error"); return; }
  }
  if (!state.files.length) { toast("Add images in the Workspace queue before running an experiment", "warning"); return; }
  if (state.gtMode && (state.files.length !== 1 || !state.gtFile)) { toast("GT experiments require exactly one image and one GT file", "warning"); return; }
  const model=selModel(), colormap=selCmap(), device=selDevice();
  state.experiment = { name: el.experimentName?.value || "DepthLens validation run", results: [], startedAt: new Date().toISOString() };
  el.experimentStatus.textContent = "Running experiment…";
  el.experimentRunBtn.disabled = true;
  try {
    for (const entry of state.files) {
      const result = await inferOne(entry.file, model, colormap, device, null, state.gtMode ? "full" : "fast", "color,gray", state.gtMode ? state.gtFile : null, state.gtMode);
      state.experiment.results.push({...result, originalSrc: entry.thumb, filename: entry.file.name, experiment_name: state.experiment.name});
      renderExperiment();
    }
    el.experimentStatus.textContent = `Completed ${state.experiment.results.length} result(s) for ${state.experiment.name}.`;
    toast("Experiment complete", "success");
  } catch (err) {
    el.experimentStatus.textContent = err.message;
    toast(`Experiment failed: ${err.message}`, "error", 6000);
  } finally {
    el.experimentRunBtn.disabled = false;
  }
}
const activeBlobUrls = new Set();
function revokeBlobUrl(url) {
  if (!activeBlobUrls.has(url)) return;
  activeBlobUrls.delete(url);
  URL.revokeObjectURL(url);
}
function exportBlob(name, type, content) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  activeBlobUrls.add(url);
  while (activeBlobUrls.size > 20) revokeBlobUrl(activeBlobUrls.values().next().value);
  const a = Object.assign(document.createElement("a"), { href:url, download:name });
  document.body.appendChild(a);
  requestAnimationFrame(() => {
    a.click();
    a.remove();
    setTimeout(() => revokeBlobUrl(url), 60_000);
  });
}
window.addEventListener("pagehide", () => {
  stopWebcam({ quiet: true });
  [...activeBlobUrls].forEach(revokeBlobUrl);
});
function exportExperimentJson() {
  const payload = { run_name: state.experiment.name, started_at: state.experiment.startedAt, exported_at: new Date().toISOString(), results: state.experiment.results };
  exportBlob(`${state.experiment.name.replace(/[^a-z0-9_-]+/gi,"_")}.json`, "application/json", JSON.stringify(payload,null,2));
}
function exportExperimentCsv() {
  const rows = experimentRows();
  const header = ["filename","model","engine","device","latency_ms","abs_rel","rmse","delta_1","gt","fallback","warnings"];
  const csv = [header.join(","), ...rows.map(r => header.map(k => `"${String(r[k] ?? "").replace(/"/g,'""')}"`).join(","))].join("\n");
  exportBlob(`${(state.experiment.name||"experiment").replace(/[^a-z0-9_-]+/gi,"_")}.csv`, "text/csv", csv);
}
el.experimentRunBtn?.addEventListener("click", runExperiment);
el.experimentExportJsonBtn?.addEventListener("click", exportExperimentJson);
el.experimentExportCsvBtn?.addEventListener("click", exportExperimentCsv);

// ══════════════════════════════════════════════════════════════
// TOAST
// ══════════════════════════════════════════════════════════════
function toastOnce(msg, type="info", dur=3500) {
  const now = Date.now();
  if (state.lastToast.message === msg && now - state.lastToast.at < 15000) return;
  state.lastToast = { message: msg, at: now };
  toast(msg, type, dur);
}

function toast(msg, type="info", dur=3500) {
  const t=document.createElement("div");
  t.className=`toast ${type}`;
  t.innerHTML=`<span class="toast-dot"></span><span>${esc(msg)}</span>`;
  el.toastContainer.appendChild(t);
  setTimeout(()=>{
    t.style.animation="toastOut .28s ease forwards";
    setTimeout(()=>t.remove(),280);
  },dur);
}

// ══════════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════════
function dlB64(name,b64) {
  const a=Object.assign(document.createElement("a"),{
    href:safeDataImagePng(b64),
    download:String(name).match(/\.[^.]+$/)?String(name):`${String(name)}.png`,
  });
  if (!a.href) return;
  document.body.appendChild(a); a.click(); a.remove();
}

function esc(s) {
  return String(s ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

function escText(s) {
  return String(s ?? "");
}

function safeDataImagePng(b64) {
  const value = String(b64 || "");
  return /^[A-Za-z0-9+/=\r\n]+$/.test(value) ? `data:image/png;base64,${value.replace(/\s/g,"")}` : "";
}


function stopPollingLoops({ abort = true } = {}) {
  Object.values(state.pollTimers).forEach(clearInterval);
  state.pollTimers = {};
  state.pollMode = null;
  state.pollEpoch += 1;
  if (abort) {
    Object.values(state.pollControllers).forEach(controller => controller?.abort());
    state.pollControllers = {};
    state.pollInFlight = {};
  }
}

function runPollingTask(name, fn) {
  if (state.pollInFlight[name]) return;
  const controller = new AbortController();
  const epoch = state.pollEpoch;
  state.pollControllers[name]?.abort();
  state.pollControllers[name] = controller;
  state.pollInFlight[name] = true;
  Promise.resolve(fn(controller.signal))
    .catch(err => {
      if (err.name !== "AbortError" && epoch === state.pollEpoch) console.debug(`[DepthLens] ${name} poll failed: ${err.message}`);
    })
    .finally(() => {
      if (state.pollControllers[name] === controller) {
        delete state.pollControllers[name];
        state.pollInFlight[name] = false;
      }
    });
}

function addPollingInterval(name, intervalMs, fn) {
  state.pollTimers[name] = setInterval(() => runPollingTask(name, fn), intervalMs);
}

function startPollingLoops() {
  const mode = document.hidden ? "hidden" : "visible";
  if (state.pollMode === mode && Object.keys(state.pollTimers).length) return;
  stopPollingLoops();
  state.pollMode = mode;
  state.pollEpoch += 1;
  if (document.hidden) {
    addPollingInterval("live", 30_000, signal => checkLive({ quiet: true, signal }));
    return;
  }
  addPollingInterval("live", 10_000, signal => checkLive({ quiet: true, signal }));
  addPollingInterval("diagnostics", 60_000, signal => checkDiagnostics({ quiet: true, signal }));
  addPollingInterval("cacheMetrics", 45_000, signal => loadCacheMetrics({ signal }));
}

document.addEventListener("visibilitychange", () => {
  startPollingLoops();
  handleWebcamVisibilityChange();
  if (!document.hidden) {
    runPollingTask("live", signal => checkLive({ quiet: true, signal }));
  }
});

// ══════════════════════════════════════════════════════════════
// INITIALIZATION
// ══════════════════════════════════════════════════════════════
async function init() {
  if (el.appShell) el.appShell.classList.remove("ready");
  state.initializingBackend = true;
  loadPrefs();
  setStatus("connecting", "Starting engine…", DEFAULT_API_BASE_URL);
  syncQueueControls();
  initReconstructionPanel();
  initScrollableNav();
  initGuideAccordion();
  bindPointerGlow(".logo-group", { tilt: 3 });
  bindPointerGlow(".nav-btn", { tilt: 5 });
  bindPointerGlow(".guide-section-toggle", { tilt: 3 });
  bindPointerGlow(".guide-card, .guide-section", { tilt: 1.5 });
  try {
    await resolveApiBaseUrl();
    if (!runningInElectron) toastOnce("Browser/file mode detected — start the backend manually for inference.", "warning", 7000);
    initLatencyChart();
    initCompareControls();
    initEngineStatusPanel();
    syncWebcamControls();
    updateWebcamTelemetry();
    switchPanel("main");
    await checkLive();
    await checkReadiness({ quiet: false });
    state.initializingBackend = false;
    syncQueueControls();
    syncReconstructControls();
    syncWebcamControls();
    updateWebcamTelemetry();
    startPollingLoops();
    Promise.allSettled([checkDiagnostics({ quiet: true }), loadCacheMetrics()]);
  } catch (err) {
    backendOnline = false;
    setStatus("offline", "Depth Engine: Offline", `Backend URL resolution failed: ${err.message}`);
    toast(`Backend initialization failed: ${err.message}`, "error", 6000);
  } finally {
    state.initializingBackend = false;
    syncQueueControls();
    syncReconstructControls();
    syncWebcamControls();
    updateWebcamTelemetry();
  }
}

init().catch(err => {
  console.error("[DepthLens] Fatal init error", err);
  state.initializingBackend = false;
  backendOnline = false;
  inferenceReady = false;
  syncQueueControls();
  syncReconstructControls();
  syncWebcamControls();
  updateWebcamTelemetry();
});
