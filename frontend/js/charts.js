"use strict";

// ══════════════════════════════════════════════════════════════
// BACKGROUND CANVAS (workspace)
// ══════════════════════════════════════════════════════════════
(function bgCanvas() {
  const cv = el.bgCanvas, ctx = cv.getContext("2d");
  let W, H, pts = [];
  const N = 50;
  const reduce = prefersReducedMotion();
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
  if (state.observability.chart) { applyChartPalette(state.observability.chart, c); state.observability.chart.update("none"); }
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
  if (!settings.endpointTimingLogs && !settings.verboseConsoleLogs) return;
  const ms = Math.round(performance.now() - started);
  console.debug(`[DepthLens] ${label} ${ok ? "ok" : "failed"} in ${ms}ms${extra ? ` · ${extra}` : ""}`);
}
