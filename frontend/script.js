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
  gtMode: false,
  gtFile: null,
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
  if (["model","colormap","device"].includes(e.target.name)) savePrefs();
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
  el.statusDot.className = `status-dot ${cls}`;
  el.statusLabel.textContent = label;
  el.statusSub.textContent = sub || "";
  if (deviceText) { el.deviceBadge.textContent = deviceText; el.deviceBadge.hidden = false; }
  else { el.deviceBadge.hidden = true; }
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
    return inferenceReady;
  } catch (err) {
    logEndpointTiming("/ready", started, false, err.message);
    readinessDetails = null;
    inferenceReady = false;
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
    backendOnline = data.status === "ok";
    if (!backendOnline) throw new Error("Unexpected /live response");
    if (data.busy) {
      setStatus("online", "Depth Engine: Busy", "Benchmark/model load running · liveness OK", deviceBadge(state.devices, state.primaryDevice));
    } else {
      setStatus("online","Depth Engine: Detected",`Backend live · checking inference readiness`, deviceBadge(state.devices, state.primaryDevice));
    }
    logEndpointTiming("/live", started, true);
    syncQueueControls();
    return true;
  } catch (err) {
    logEndpointTiming("/live", started, false, err.message);
    backendOnline = false;
    inferenceReady = false;
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
// PANEL NAVIGATION
// ══════════════════════════════════════════════════════════════
function switchPanel(name) {
  el.navBtns.forEach(b => b.classList.toggle("active", b.dataset.panel===name));
  el.panels.forEach(p => { p.hidden = p.id !== `panel-${name}`; });
}
el.navBtns.forEach(b => b.addEventListener("click", () => switchPanel(b.dataset.panel)));

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

async function inferOne(file,model,colormap,device,signal,metrics="fast",outputs="color",gtFile=null,gtRequired=false) {
  const fd=new FormData();
  fd.append("file",file); fd.append("model",model);
  fd.append("colormap",colormap); fd.append("device",device);
  fd.append("metrics", metrics);
  fd.append("outputs", outputs);
  if (gtFile) fd.append("gt_file", gtFile);
  if (gtRequired) fd.append("gt_required", "true");
  if (!engineReady()) throw new Error(`Depth engine is unavailable at ${API || DEFAULT_API_BASE_URL}`);
  const res=await apiFetch("/estimate",{
    method:"POST", body:fd,
    signal: requestSignal(signal, 180_000),
  });
  return res.json();
}

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
  try {
    await resolveApiBaseUrl();
    if (!runningInElectron) toastOnce("Browser/file mode detected — start the backend manually for inference.", "warning", 7000);
    initLatencyChart();
    initCompareControls();
    switchPanel("main");
    await checkLive();
    await checkReadiness({ quiet: false });
    state.initializingBackend = false;
    syncQueueControls();
    startPollingLoops();
    Promise.allSettled([checkDiagnostics({ quiet: true }), loadCacheMetrics()]);
  } catch (err) {
    backendOnline = false;
    setStatus("offline", "Depth Engine: Offline", `Backend URL resolution failed: ${err.message}`);
    toast(`Backend initialization failed: ${err.message}`, "error", 6000);
  } finally {
    state.initializingBackend = false;
    syncQueueControls();
  }
}

init().catch(err => {
  console.error("[DepthLens] Fatal init error", err);
  state.initializingBackend = false;
  backendOnline = false;
  inferenceReady = false;
  syncQueueControls();
});
