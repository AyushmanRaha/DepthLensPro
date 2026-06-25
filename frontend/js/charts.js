"use strict";

// ══════════════════════════════════════════════════════════════
// CHARTS
// ══════════════════════════════════════════════════════════════
let latencyChart = null;
let compareChart = null;
let benchmarkChart = null;

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

// ══════════════════════════════════════════════════════════════
// BACKGROUND CANVAS (workspace)
// ══════════════════════════════════════════════════════════════
function safePrefersReducedMotion() {
  try {
    return typeof prefersReducedMotion === "function" ? prefersReducedMotion() : false;
  } catch (err) {
    console.warn("[DepthLens] Reduced motion preference unavailable; using animated canvas fallback", err);
    return false;
  }
}

(function startBackgroundCanvas() {
  try {
    bgCanvas();
  } catch (err) {
    console.warn("[DepthLens] Background canvas initialization skipped", err);
  }
})();

function bgCanvas() {
  const cv = el?.bgCanvas;
  if (!cv?.getContext) return;
  const ctx = cv.getContext("2d");
  if (!ctx) return;
  let W, H, pts = [];
  const N = 50;
  const reduce = safePrefersReducedMotion();
  function accentRgb() {
    const c = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
    const m = c.match(/^#?([0-9a-f]{6})$/i);
    if (!m) return [0,200,255];
    return [parseInt(m[1].slice(0,2),16), parseInt(m[1].slice(2,4),16), parseInt(m[1].slice(4,6),16)];
  }

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
    const [ar,ag,ab] = accentRgb();
    ctx.strokeStyle = `rgba(${ar},${ag},${ab},.055)`; ctx.lineWidth=1;
    for (let x=0;x<W;x+=64) { ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke(); }
    for (let y=0;y<H;y+=64) { ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke(); }
    for (let i=0;i<pts.length;i++) {
      const p=pts[i];
      p.x=(p.x+p.vx+W)%W; p.y=(p.y+p.vy+H)%H;
      for (let j=i+1;j<pts.length;j++) {
        const q=pts[j], d=Math.hypot(p.x-q.x,p.y-q.y);
        if (d<130) {
          ctx.strokeStyle=`rgba(${ar},${ag},${ab},${0.11*(1-d/130)})`;
          ctx.lineWidth=0.55; ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke();
        }
      }
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(${ar},${ag},${ab},${p.a*0.55})`; ctx.fill();
    }
    if (!reduce) requestAnimationFrame(draw);
  }
  window.addEventListener("resize",resize);
  reset(); draw();
}

function chartColors() {
  const isDark = document.documentElement.getAttribute("data-theme") !== "light";
  return {
    grid:    isDark ? "rgba(0,200,255,.07)"  : "rgba(0,100,180,.08)",
    tick:    isDark ? "#3a5a72"               : "#5a7a99",
    line:    isDark ? "#00c8ff"               : "#0070cc",
    fill:    isDark ? "rgba(0,200,255,.07)"   : "rgba(0,112,204,.06)",
    bar:     isDark ? "rgba(0,200,255,.55)"   : "rgba(0,112,204,.55)",
    barBrd:  isDark ? "#00c8ff"               : "#0070cc",
    text:    isDark ? "#7faac8"               : "#2d4a66",
    body:    isDark ? "#dff0ff"               : "#0d1f33",
    muted:   isDark ? "rgba(127,140,153,.45)" : "rgba(127,140,153,.55)",
    mutedBrd:isDark ? "#5e6f81"               : "#6c7f91",
  };
}

const chartWarned = new Set();
const chartRegistry = new Map();

function warnChartOnce(key, message) {
  if (chartWarned.has(key)) return;
  chartWarned.add(key);
  console.warn(message);
}

function getCanvasContext(canvas, label = "chart") {
  if (!canvas?.getContext) {
    warnChartOnce(`${label}:missing`, `DepthLens Pro: ${label} chart canvas is unavailable.`);
    return null;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) warnChartOnce(`${label}:context`, `DepthLens Pro: ${label} chart canvas context is unavailable.`);
  return ctx;
}

function validDimension(value) {
  return Number.isFinite(value) && value > 0;
}

function isChartMeasurable(canvas) {
  const parent = canvas?.parentElement;
  if (!canvas?.isConnected || !parent) return false;
  const rect = parent.getBoundingClientRect?.();
  return Boolean(rect && rect.width > 0 && rect.height > 0);
}

function resizeCanvasToDisplaySize(canvas) {
  const parentRect = canvas?.parentElement?.getBoundingClientRect?.() || {};
  const rect = canvas?.getBoundingClientRect?.() || {};
  const sourceWidth = validDimension(parentRect.width) ? parentRect.width : (rect.width || canvas?.clientWidth || 320);
  const sourceHeight = validDimension(parentRect.height) ? parentRect.height : (rect.height || canvas?.clientHeight || 180);
  const cssWidth = Math.max(1, Math.round(sourceWidth));
  const cssHeight = Math.max(1, Math.round(sourceHeight));
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  const width = Math.max(1, Math.round(cssWidth * dpr));
  const height = Math.max(1, Math.round(cssHeight * dpr));
  const changed = canvas.width !== width || canvas.height !== height;
  if (changed) { canvas.width = width; canvas.height = height; }
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  const ctx = canvas.getContext?.("2d");
  ctx?.setTransform?.(dpr, 0, 0, dpr, 0, 0);
  return { width: cssWidth, height: cssHeight, dpr, changed };
}

function clearCanvas(canvas) {
  const ctx = getCanvasContext(canvas, "canvas");
  if (!ctx) return null;
  const size = resizeCanvasToDisplaySize(canvas);
  ctx.clearRect(0, 0, size.width, size.height);
  return { ctx, ...size };
}

function formatTickValue(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  const digits = abs === 0 || abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function measureTickWidth(ctx, maxValue, formatter = formatTickValue) {
  const values = [0, maxValue / 2, maxValue].map((v) => formatter(v));
  return Math.ceil(Math.max(...values.map((label) => ctx.measureText?.(label)?.width || String(label).length * 6), 24));
}

function chartArea(width, height, options = {}) {
  const legendHeight = options.legendHeight || 0;
  const xAxisHeight = options.showXAxis === false ? 8 : 24;
  const tickWidth = Math.max(26, options.tickWidth || 34);
  const left = Math.max(32, tickWidth + 10);
  const right = options.right ?? 10;
  const top = (options.top ?? 10) + legendHeight;
  const bottom = options.bottom ?? xAxisHeight;
  return {
    left,
    right,
    top,
    bottom,
    width: Math.max(20, width - left - right),
    height: Math.max(20, height - top - bottom),
  };
}

function drawCenteredLegend(ctx, area, width, label, colors, options = {}) {
  if (!label) return 0;
  const swatch = options.swatchSize || 8;
  const gap = 6;
  ctx.font = options.font || "10px Rajdhani, sans-serif";
  const textWidth = ctx.measureText?.(label)?.width || String(label).length * 6;
  const total = swatch + gap + textWidth;
  const x = Math.max(area.left, (width - total) / 2);
  const y = options.y ?? 9;
  ctx.fillStyle = options.fill || colors.bar;
  ctx.strokeStyle = options.stroke || colors.barBrd;
  ctx.lineWidth = 1;
  ctx.fillRect(x, y, swatch, swatch);
  ctx.strokeRect?.(x, y, swatch, swatch);
  ctx.fillStyle = colors.text;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x + swatch + gap, y + swatch / 2);
  return swatch + 8;
}

function drawGrid(ctx, area, maxValue, colors, options = {}) {
  const formatter = options.tickFormatter || formatTickValue;
  ctx.strokeStyle = colors.grid; ctx.lineWidth = 1; ctx.fillStyle = colors.tick;
  ctx.font = "9px JetBrains Mono, monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i += 1) {
    const y = area.top + (area.height * i / 4);
    ctx.beginPath(); ctx.moveTo(area.left, y); ctx.lineTo(area.left + area.width, y); ctx.stroke();
    const value = maxValue - (maxValue * i / 4);
    ctx.fillText(formatter(value), area.left - 6, y);
  }
}

function numericValues(values) {
  return (values || []).map(Number).filter(Number.isFinite);
}

function drawNoDataState(canvas, message = "No chart data yet") {
  const drawing = clearCanvas(canvas); if (!drawing) return false;
  const { ctx, width, height } = drawing; const c = chartColors();
  ctx.font = "9px JetBrains Mono, monospace";
  const area = chartArea(width, height, { tickWidth: measureTickWidth(ctx, 100), showXAxis: false });
  drawGrid(ctx, area, 100, c);
  ctx.fillStyle = c.text; ctx.font = "12px Rajdhani, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(message, width / 2, height / 2);
  return true;
}

function linePoints(data, area, maxValue) {
  const xFor = (i) => area.left + (data.length === 1 ? area.width / 2 : (area.width * i / (data.length - 1)));
  const yFor = (v) => area.top + area.height - (Math.max(0, v) / maxValue) * area.height;
  return data.map((v, i) => ({ x: xFor(i), y: yFor(v) }));
}

function drawSmoothLinePath(ctx, points, tension = 0.38) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  if (points.length < 3 || !ctx.bezierCurveTo) {
    points.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
    return;
  }
  for (let i = 0; i < points.length - 1; i += 1) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(points.length - 1, i + 2)];
    const cp1x = p1.x + (p2.x - p0.x) * tension / 6;
    const cp1y = p1.y + (p2.y - p0.y) * tension / 6;
    const cp2x = p2.x - (p3.x - p1.x) * tension / 6;
    const cp2y = p2.y - (p3.y - p1.y) * tension / 6;
    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
  }
}

function drawLineChart(canvas, values, options = {}) {
  const data = numericValues(values);
  if (!data.length) return drawNoDataState(canvas, options.emptyMessage || "No chart data yet");
  const drawing = clearCanvas(canvas); if (!drawing) return false;
  const { ctx, width, height } = drawing; const c = chartColors();
  const maxValue = Math.max(1, options.maxValue || Math.max(...data) * 1.15);
  ctx.font = "9px JetBrains Mono, monospace";
  const area = chartArea(width, height, { tickWidth: measureTickWidth(ctx, maxValue, options.tickFormatter), showXAxis: false, top: 8, bottom: 10 });
  drawGrid(ctx, area, maxValue, c, { tickFormatter: options.tickFormatter });
  const points = linePoints(data, area, maxValue);
  drawSmoothLinePath(ctx, points, options.tension ?? 0.38);
  ctx.lineTo(points[points.length - 1].x, area.top + area.height);
  ctx.lineTo(points[0].x, area.top + area.height);
  ctx.closePath();
  ctx.fillStyle = options.fillColor || c.fill; ctx.fill();
  drawSmoothLinePath(ctx, points, options.tension ?? 0.38);
  ctx.strokeStyle = options.lineColor || c.line; ctx.lineWidth = options.lineWidth || 1.8; ctx.stroke();
  ctx.fillStyle = options.lineColor || c.line;
  points.forEach((p) => { ctx.beginPath(); ctx.arc(p.x, p.y, options.pointRadius ?? 2, 0, Math.PI * 2); ctx.fill(); });
  if (options.showLegend && options.label) drawCenteredLegend(ctx, area, width, options.label, c, { fill: c.fill, stroke: c.line });
  return true;
}

function drawSparkline(canvas, values, options = {}) { return drawLineChart(canvas, values, options); }

function drawBarChart(canvas, rows, options = {}) {
  const entries = (rows || []).map((row) => ({ label: String(row.label || ""), value: Number(row.value) }));
  const drawable = entries.filter((row) => Number.isFinite(row.value));
  if (!drawable.length) return drawNoDataState(canvas, options.emptyMessage || "No chart data yet");
  const drawing = clearCanvas(canvas); if (!drawing) return false;
  const { ctx, width, height } = drawing; const c = chartColors();
  const maxValue = Math.max(1, options.maxValue || Math.max(...drawable.map((row) => row.value)) * 1.15);
  const tickFormatter = options.tickFormatter || formatTickValue;
  ctx.font = "9px JetBrains Mono, monospace";
  const legendHeight = options.label ? 20 : 0;
  const area = chartArea(width, height, { legendHeight, tickWidth: measureTickWidth(ctx, maxValue, tickFormatter), showXAxis: true, top: 8, bottom: 24 });
  if (options.label) drawCenteredLegend(ctx, area, width, options.label, c, { y: 6 });
  drawGrid(ctx, area, maxValue, c, { tickFormatter });
  const slot = area.width / Math.max(1, entries.length); const barW = Math.max(10, Math.min(slot * 0.56, 46));
  entries.forEach((row, i) => {
    const x = area.left + slot * i + (slot - barW) / 2;
    const ok = Number.isFinite(row.value); const h = ok ? Math.max(1, (Math.max(0, row.value) / maxValue) * area.height) : 2;
    const y = area.top + area.height - h;
    ctx.fillStyle = ok ? c.bar : c.muted; ctx.strokeStyle = ok ? c.barBrd : c.mutedBrd; ctx.lineWidth = 1;
    ctx.fillRect(x, y, barW, h); ctx.strokeRect?.(x, y, barW, h);
    ctx.fillStyle = ok ? c.text : c.mutedBrd; ctx.font = "10px Rajdhani, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(row.label, x + barW / 2, area.top + area.height + 6, Math.max(38, slot - 4));
    if (options.showValueLabels) {
      ctx.font = "9px JetBrains Mono, monospace"; ctx.textBaseline = "bottom";
      ctx.fillText(ok ? (options.formatValue ? options.formatValue(row.value) : formatTickValue(row.value)) : "—", x + barW / 2, y - 3);
    }
  });
  return true;
}

function drawGroupedBarChart(canvas, rows, options = {}) { return drawBarChart(canvas, rows, options); }

function rememberChart(name, canvas, draw) {
  if (!canvas) return;
  chartRegistry.set(name, { canvas, draw });
}

function redrawVisibleCharts() {
  chartRegistry.forEach((entry) => {
    if (isChartMeasurable(entry.canvas)) entry.draw?.();
  });
}

let chartResizeTimer = null;
function scheduleChartResize() {
  clearTimeout(chartResizeTimer);
  chartResizeTimer = setTimeout(redrawVisibleCharts, 80);
}
window.addEventListener?.("resize", scheduleChartResize);

function initLatencyChart() {
  const canvas = $("#latencyChart");
  if (!canvas) { warnChartOnce("latency:missing", "DepthLens Pro: latency chart canvas is unavailable."); return null; }
  latencyChart = { canvas, values: [] };
  rememberChart("latency", canvas, () => drawSparkline(canvas, latencyChart.values, { label: "Inference ms", emptyMessage: "No latency samples yet" }));
  drawSparkline(canvas, [], { label: "Inference ms", emptyMessage: "No latency samples yet" });
  return latencyChart;
}

function updateChartTheme() {
  if (latencyChart) drawSparkline(latencyChart.canvas, latencyChart.values, { label: "Inference ms", emptyMessage: "No latency samples yet" });
  if (compareChart) compareChart.draw?.();
  if (benchmarkChart) benchmarkChart.draw?.();
  if (state?.observability?.chart) state.observability.chart.draw?.();
}

function pushLatency(ms) {
  const value = Number(ms);
  if (!Number.isFinite(value)) { warnChartOnce("latency:bad-data", "DepthLens Pro: latency chart skipped because no drawable latency data was provided."); return; }
  if (!latencyChart && !initLatencyChart()) return;
  const d = (state?.session?.latencies || []).map(Number).filter(Number.isFinite).slice(-20);
  latencyChart.values = d.length ? d : [value];
  drawSparkline(latencyChart.canvas, latencyChart.values, { label: "Inference ms", emptyMessage: "No latency samples yet" });
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
  const rows = (results || []).map(r => ({
    label: escText(r.model).replace("MiDaS_","").replace("DPT_","DPT "),
    value: Number(compareMetricValue(r, metric)),
  }));
  if (el.compareChartCard) el.compareChartCard.hidden = false;
  const canvas = $("#compareChart");
  if (!canvas) { warnChartOnce("compare:missing", "DepthLens Pro: compare chart canvas is unavailable."); return; }
  compareChart = {
    canvas,
    rows,
    metricKey: metric.key,
    draw: () => drawBarChart(canvas, rows, {
      label: metric.label,
      emptyMessage: "No comparison chart data yet",
      formatValue: (value) => metric.fmt(value),
      tickFormatter: (value) => metric.fmt(value),
    }),
  };
  rememberChart("compare", canvas, compareChart.draw);
  compareChart.draw();
}

function initCompareControls() {
  if (!el.compareMetricSelect || !el.compareChartToggle || !el.compareChartBody) return;
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
    if (state.compareView.open) requestAnimationFrame(() => scheduleChartResize());
  });
}

const DepthLensCharts = {
  getCanvasContext,
  resizeCanvasToDisplaySize,
  isChartMeasurable,
  clearCanvas,
  drawNoDataState,
  drawLineChart,
  drawBarChart,
  drawGroupedBarChart,
  drawSparkline,
  scheduleChartResize,
  redrawVisibleCharts,
  renderCompareChart,
};
window.DepthLensCharts = DepthLensCharts;

function logEndpointTiming(label, started, ok, extra = "") {
  if (!settings.endpointTimingLogs && !settings.verboseConsoleLogs) return;
  const ms = Math.round(performance.now() - started);
  console.debug(`[DepthLens] ${label} ${ok ? "ok" : "failed"} in ${ms}ms${extra ? ` · ${extra}` : ""}`);
}
