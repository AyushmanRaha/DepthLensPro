/**
 * DepthLens Pro — Frontend Logic v2.0
 * Handles: panel navigation, file queue, batch inference,
 *          gallery, lightbox, model comparison, metrics, and bg canvas.
 */

"use strict";

/* ═══════════════════════════════════════════════════════════
   CONFIG
═══════════════════════════════════════════════════════════ */
const API_BASE    = "http://127.0.0.1:8000";
const POLL_DELAY  = 800; // ms between health polls

/* ═══════════════════════════════════════════════════════════
   STATE
═══════════════════════════════════════════════════════════ */
const state = {
  files:       [],          // { id, file, thumb, status, result }
  results:     [],          // accumulated results
  latencies:   [],          // last N latency readings
  session: {
    total:    0,
    cached:   0,
    errors:   0,
    latencies: [],
  },
  lightbox: {
    current: null,
  },
};

/* ═══════════════════════════════════════════════════════════
   ELEMENT SHORTCUTS
═══════════════════════════════════════════════════════════ */
const $  = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

const el = {
  // Header
  statusDot:     $("#statusDot"),
  statusLabel:   $("#statusLabel"),
  navBtns:       $$(".nav-btn"),
  panels:        $$(".panel"),

  // Upload
  dropZone:      $("#dropZone"),
  fileInput:     $("#fileInput"),
  browseBtn:     $("#browseBtn"),
  fileQueue:     $("#fileQueue"),
  clearBtn:      $("#clearBtn"),
  runBtn:        $("#runBtn"),
  progressWrap:  $("#progressWrap"),
  progressFill:  $("#progressFill"),
  progressLabel: $("#progressLabel"),
  progressBar:   $("#progressBar"),

  // Results
  resultsCard:       $("#resultsCard"),
  gallery:           $("#gallery"),
  downloadAllBtn:    $("#downloadAllBtn"),
  clearResultsBtn:   $("#clearResultsBtn"),

  // Metrics
  metricTotal:      $("#metricTotal"),
  metricAvgLatency: $("#metricAvgLatency"),
  metricCached:     $("#metricCached"),
  metricErrors:     $("#metricErrors"),

  // Compare
  compareDropZone:       $("#compareDropZone"),
  compareFileInput:      $("#compareFileInput"),
  compareBrowseBtn:      $("#compareBrowseBtn"),
  compareRunBtn:         $("#compareRunBtn"),
  compareResults:        $("#compareResults"),
  compareCmap:           $("#compareCmap"),
  compareProgressWrap:   $("#compareProgressWrap"),
  compareProgressFill:   $("#compareProgressFill"),
  compareProgressLabel:  $("#compareProgressLabel"),
  compareChartCard:      $("#compareChartCard"),

  // Lightbox
  lightbox:           $("#lightbox"),
  lightboxBackdrop:   $("#lightboxBackdrop"),
  lightboxClose:      $("#lightboxClose"),
  lbOrigImg:          $("#lbOrigImg"),
  lbDepthImg:         $("#lbDepthImg"),
  lbStats:            $("#lbStats"),
  lbSlider:           $("#lbSlider"),
  lbDownloadDepth:    $("#lbDownloadDepth"),
  lbDownloadGray:     $("#lbDownloadGray"),

  // Toast
  toastContainer: $("#toastContainer"),

  // Background canvas
  bgCanvas: $("#bgCanvas"),
};

/* ═══════════════════════════════════════════════════════════
   BACKGROUND CANVAS — animated grid + particles
═══════════════════════════════════════════════════════════ */
(function initCanvas() {
  const canvas = el.bgCanvas;
  const ctx    = canvas.getContext("2d");
  let W, H, particles = [];
  const N_PARTICLES = 55;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function mkParticle() {
    return {
      x:  Math.random() * W,
      y:  Math.random() * H,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      r:  Math.random() * 1.5 + 0.4,
      a:  Math.random(),
    };
  }

  function reset() {
    resize();
    particles = Array.from({ length: N_PARTICLES }, mkParticle);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "rgba(0,200,255,0.06)";
    ctx.lineWidth   = 1;
    const CELL = 60;
    for (let x = 0; x < W; x += CELL) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += CELL) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Particles + connecting lines
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;

      for (let j = i + 1; j < particles.length; j++) {
        const q = particles[j];
        const d = Math.hypot(p.x - q.x, p.y - q.y);
        if (d < 140) {
          ctx.strokeStyle = `rgba(0,200,255,${0.12 * (1 - d / 140)})`;
          ctx.lineWidth   = 0.6;
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
        }
      }

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,200,255,${p.a * 0.6})`;
      ctx.fill();
    }

    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  reset();
  draw();
})();

/* ═══════════════════════════════════════════════════════════
   LATENCY CHART
═══════════════════════════════════════════════════════════ */
let latencyChart;

function initLatencyChart() {
  const ctx = $("#latencyChart").getContext("2d");
  latencyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels:   [],
      datasets: [{
        label:           "Latency (ms)",
        data:            [],
        borderColor:     "#00c8ff",
        backgroundColor: "rgba(0,200,255,0.08)",
        borderWidth:     1.5,
        pointRadius:     2,
        tension:         0.4,
        fill:            true,
      }],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           { duration: 300 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#111927",
          borderColor:     "#00c8ff",
          borderWidth:     1,
          titleColor:      "#7fa8c8",
          bodyColor:       "#e8f4ff",
          callbacks: { label: ctx => `${ctx.raw} ms` },
        },
      },
      scales: {
        x: {
          display:   false,
          grid:      { display: false },
        },
        y: {
          display:   true,
          grid:      { color: "rgba(0,200,255,0.07)" },
          ticks: {
            color:     "#3d607a",
            font:      { family: "JetBrains Mono", size: 9 },
            maxTicksLimit: 4,
          },
        },
      },
    },
  });
}

function pushLatency(ms) {
  state.session.latencies.push(ms);
  const MAX = 20;
  const data = state.session.latencies.slice(-MAX);
  latencyChart.data.labels   = data.map((_, i) => i + 1);
  latencyChart.data.datasets[0].data = data;
  latencyChart.update("none");
}

/* ═══════════════════════════════════════════════════════════
   COMPARE CHART
═══════════════════════════════════════════════════════════ */
let compareChart;

function renderCompareChart(results) {
  el.compareChartCard.hidden = false;
  const ctx = $("#compareChart").getContext("2d");

  const labels  = results.map(r => r.model.replace("MiDaS_", "").replace("DPT_", "DPT "));
  const latency = results.map(r => r.latency_ms);

  if (compareChart) compareChart.destroy();

  compareChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label:           "Latency (ms)",
          data:            latency,
          backgroundColor: ["rgba(0,200,255,0.6)", "rgba(123,92,248,0.6)", "rgba(255,107,107,0.6)"],
          borderColor:     ["#00c8ff", "#7b5cf8", "#ff6b6b"],
          borderWidth:     1.5,
          borderRadius:    4,
        },
      ],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           { duration: 400 },
      plugins: {
        legend: {
          labels: { color: "#7fa8c8", font: { family: "JetBrains Mono", size: 10 } },
        },
        tooltip: {
          backgroundColor: "#111927",
          borderColor:     "#00c8ff",
          borderWidth:     1,
          titleColor:      "#7fa8c8",
          bodyColor:       "#e8f4ff",
        },
      },
      scales: {
        x: {
          ticks: { color: "#7fa8c8", font: { family: "Rajdhani", size: 12, weight: "600" } },
          grid:  { color: "rgba(0,200,255,0.07)" },
        },
        y: {
          ticks: {
            color: "#3d607a",
            font:  { family: "JetBrains Mono", size: 9 },
            callback: v => `${v} ms`,
          },
          grid: { color: "rgba(0,200,255,0.07)" },
        },
      },
    },
  });
}

/* ═══════════════════════════════════════════════════════════
   HEALTH / SERVER STATUS
═══════════════════════════════════════════════════════════ */
async function checkHealth() {
  setStatus("loading", "Connecting…");
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const device = data.device === "cuda" ? "GPU" : "CPU";
    setStatus("online", `Online · ${device}`);
    return true;
  } catch {
    setStatus("offline", "Server offline");
    return false;
  }
}

function setStatus(state, label) {
  el.statusDot.className   = `status-dot ${state}`;
  el.statusLabel.textContent = label;
}

/* ═══════════════════════════════════════════════════════════
   PANEL NAVIGATION
═══════════════════════════════════════════════════════════ */
function switchPanel(name) {
  el.navBtns.forEach(b => b.classList.toggle("active", b.dataset.panel === name));
  el.panels.forEach(p => {
    const match = p.id === `panel-${name}`;
    p.hidden = !match;
    if (match) p.classList.add("active");
    else        p.classList.remove("active");
  });
}

el.navBtns.forEach(btn => {
  btn.addEventListener("click", () => switchPanel(btn.dataset.panel));
});

/* ═══════════════════════════════════════════════════════════
   FILE HANDLING
═══════════════════════════════════════════════════════════ */
function uid() {
  return Math.random().toString(36).slice(2, 9);
}

function formatSize(bytes) {
  if (bytes < 1024)       return `${bytes} B`;
  if (bytes < 1048576)    return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function addFiles(fileList) {
  const allowed = ["image/png","image/jpeg","image/webp","image/bmp","image/tiff","image/gif"];
  let added = 0;

  for (const file of fileList) {
    if (!allowed.some(t => file.type === t || file.type.startsWith("image/"))) {
      showToast(`Skipped "${file.name}" — unsupported format`, "warning");
      continue;
    }
    if (file.size > 20 * 1024 * 1024) {
      showToast(`Skipped "${file.name}" — exceeds 20 MB`, "warning");
      continue;
    }
    // Avoid duplicates by name+size
    if (state.files.some(f => f.file.name === file.name && f.file.size === file.size)) continue;

    const entry = { id: uid(), file, thumb: null, status: "pending", result: null };
    state.files.push(entry);
    renderFileItem(entry);

    // Generate thumbnail
    const reader = new FileReader();
    reader.onload = e => {
      entry.thumb = e.target.result;
      const img = $(`#thumb-${entry.id}`);
      if (img) img.src = e.target.result;
    };
    reader.readAsDataURL(file);
    added++;
  }

  if (added > 0) {
    updateQueueControls();
    showToast(`${added} file${added > 1 ? "s" : ""} added to queue`);
  }
}

function renderFileItem(entry) {
  const li = document.createElement("li");
  li.className = "file-item";
  li.id        = `file-item-${entry.id}`;
  li.innerHTML = `
    <img class="file-thumb" id="thumb-${entry.id}" src="" alt="" />
    <div class="file-meta">
      <div class="file-name" title="${escapeHtml(entry.file.name)}">${escapeHtml(entry.file.name)}</div>
      <div class="file-size">${formatSize(entry.file.size)}</div>
    </div>
    <span class="file-status pending" id="status-${entry.id}">Pending</span>
    <button class="file-remove" data-id="${entry.id}" aria-label="Remove ${escapeHtml(entry.file.name)}">✕</button>
  `;
  el.fileQueue.appendChild(li);
  li.querySelector(".file-remove").addEventListener("click", () => removeFile(entry.id));
}

function removeFile(id) {
  state.files = state.files.filter(f => f.id !== id);
  const item = $(`#file-item-${id}`);
  if (item) item.remove();
  updateQueueControls();
}

function updateQueueControls() {
  const hasFiles = state.files.length > 0;
  el.clearBtn.disabled = !hasFiles;
  el.runBtn.disabled   = !hasFiles;
}

function setFileStatus(id, status, label) {
  const entry = state.files.find(f => f.id === id);
  if (entry) entry.status = status;

  const el_ = $(`#status-${id}`);
  if (!el_) return;
  el_.className       = `file-status ${status}`;
  el_.textContent     = label;
}

el.browseBtn.addEventListener("click", () => el.fileInput.click());
el.fileInput.addEventListener("change", () => { addFiles(el.fileInput.files); el.fileInput.value = ""; });

// Drag & drop
el.dropZone.addEventListener("dragover", e => { e.preventDefault(); el.dropZone.classList.add("drag-over"); });
el.dropZone.addEventListener("dragleave", e => { if (!el.dropZone.contains(e.relatedTarget)) el.dropZone.classList.remove("drag-over"); });
el.dropZone.addEventListener("drop", e => {
  e.preventDefault();
  el.dropZone.classList.remove("drag-over");
  addFiles(e.dataTransfer.files);
});
el.dropZone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") el.fileInput.click(); });

el.clearBtn.addEventListener("click", () => {
  state.files = [];
  el.fileQueue.innerHTML = "";
  updateQueueControls();
});

/* ═══════════════════════════════════════════════════════════
   INFERENCE
═══════════════════════════════════════════════════════════ */
function selectedModel()    { return $('input[name="model"]:checked')?.value   || "MiDaS_small"; }
function selectedColormap() { return $('input[name="colormap"]:checked')?.value || "inferno"; }

function setProgress(pct, label) {
  el.progressFill.style.width      = `${pct}%`;
  el.progressBar.setAttribute("aria-valuenow", pct);
  el.progressLabel.textContent     = label;
}

el.runBtn.addEventListener("click", runInference);

async function runInference() {
  const pending = state.files.filter(f => f.status === "pending" || f.status === "error");
  if (!pending.length) return;

  const model    = selectedModel();
  const colormap = selectedColormap();

  el.runBtn.disabled   = true;
  el.clearBtn.disabled = true;
  el.progressWrap.hidden = false;
  setProgress(0, `0 / ${pending.length} — preparing…`);

  for (let i = 0; i < pending.length; i++) {
    const entry = pending[i];
    setFileStatus(entry.id, "running", "Running…");
    setProgress(
      Math.round((i / pending.length) * 100),
      `${i + 1} / ${pending.length} — ${entry.file.name}`
    );

    try {
      const result = await inferSingle(entry.file, model, colormap);
      entry.result = result;
      entry.status = "done";
      setFileStatus(entry.id, "done", `✓ ${result.latency_ms}ms`);
      state.results.push({ ...result, originalSrc: entry.thumb, filename: entry.file.name });
      state.session.total++;
      if (result.cached) state.session.cached++;
      pushLatency(result.latency_ms);
      updateMetrics();
      appendGalleryItem(state.results[state.results.length - 1]);
      el.resultsCard.hidden = false;
    } catch (err) {
      entry.status = "error";
      setFileStatus(entry.id, "error", "Error");
      state.session.errors++;
      updateMetrics();
      showToast(`Error processing "${entry.file.name}": ${err.message}`, "error");
    }
  }

  setProgress(100, `Done — ${pending.length} image${pending.length > 1 ? "s" : ""} processed`);
  setTimeout(() => { el.progressWrap.hidden = true; }, 2000);
  el.runBtn.disabled   = false;
  el.clearBtn.disabled = false;

  showToast(`Batch complete — ${pending.filter(e => e.status === "done").length} succeeded`, "success");
}

async function inferSingle(file, model, colormap) {
  const fd = new FormData();
  fd.append("file",     file);
  fd.append("model",    model);
  fd.append("colormap", colormap);

  const res = await fetch(`${API_BASE}/estimate`, {
    method: "POST",
    body:   fd,
    signal: AbortSignal.timeout(120_000),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/* ═══════════════════════════════════════════════════════════
   METRICS
═══════════════════════════════════════════════════════════ */
function updateMetrics() {
  el.metricTotal.textContent      = state.session.total;
  el.metricCached.textContent     = state.session.cached;
  el.metricErrors.textContent     = state.session.errors;

  if (state.session.latencies.length) {
    const avg = state.session.latencies.reduce((a, b) => a + b, 0) / state.session.latencies.length;
    el.metricAvgLatency.textContent = avg.toFixed(0);
  } else {
    el.metricAvgLatency.textContent = "—";
  }
}

/* ═══════════════════════════════════════════════════════════
   GALLERY
═══════════════════════════════════════════════════════════ */
function appendGalleryItem(result) {
  const item = document.createElement("div");
  item.className = "gallery-item";
  item.setAttribute("role", "listitem");
  item.setAttribute("tabindex", "0");
  item.setAttribute("aria-label", `Depth map for ${result.filename}`);

  const stats = result.stats || {};
  item.innerHTML = `
    <div class="gallery-img-wrap">
      <img src="data:image/png;base64,${result.depth_map}" alt="Depth map of ${escapeHtml(result.filename)}" loading="lazy" />
      <div class="gallery-overlay">🔍</div>
    </div>
    <div class="gallery-meta">
      <div class="gallery-filename" title="${escapeHtml(result.filename)}">${escapeHtml(result.filename)}</div>
      <div class="gallery-tags">
        <span class="gallery-tag">${result.model?.replace("MiDaS_","").replace("DPT_","DPT ")}</span>
        <span class="gallery-tag">${result.colormap}</span>
        ${result.cached ? '<span class="gallery-tag">cached</span>' : ""}
      </div>
      <div class="gallery-stats-row">
        <span>Latency <strong>${result.latency_ms} ms</strong></span>
        <span>Mean <strong>${(stats.mean ?? 0).toFixed(3)}</strong></span>
        <span>${result.resolution?.width}×${result.resolution?.height}</span>
      </div>
    </div>
  `;

  item.addEventListener("click",  () => openLightbox(result));
  item.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") openLightbox(result); });
  el.gallery.appendChild(item);
}

el.clearResultsBtn.addEventListener("click", () => {
  state.results = [];
  el.gallery.innerHTML = "";
  el.resultsCard.hidden = true;
});

el.downloadAllBtn.addEventListener("click", () => {
  if (!state.results.length) return;
  state.results.forEach(r => downloadBase64(`depth_${r.filename}`, r.depth_map));
  showToast(`Downloading ${state.results.length} depth map${state.results.length > 1 ? "s" : ""}…`);
});

/* ═══════════════════════════════════════════════════════════
   LIGHTBOX
═══════════════════════════════════════════════════════════ */
function openLightbox(result) {
  state.lightbox.current = result;

  el.lbOrigImg.src   = result.originalSrc || "";
  el.lbDepthImg.src  = `data:image/png;base64,${result.depth_map}`;

  // Stats
  const s = result.stats || {};
  el.lbStats.innerHTML = [
    `<strong>Model:</strong> ${result.model}`,
    `<strong>Colormap:</strong> ${result.colormap}`,
    `<strong>Latency:</strong> ${result.latency_ms} ms`,
    `<strong>Min:</strong> ${s.min ?? "—"}`,
    `<strong>Max:</strong> ${s.max ?? "—"}`,
    `<strong>Mean:</strong> ${s.mean ?? "—"}`,
    `<strong>Std:</strong> ${s.std ?? "—"}`,
    `<strong>Median:</strong> ${s.median ?? "—"}`,
    `<strong>Cached:</strong> ${result.cached ? "Yes" : "No"}`,
  ].join(" &nbsp;·&nbsp; ");

  el.lbSlider.value = 50;

  el.lightbox.hidden         = false;
  el.lightboxBackdrop.hidden = false;
  document.body.style.overflow = "hidden";
  el.lightboxClose.focus();
}

function closeLightbox() {
  el.lightbox.hidden         = true;
  el.lightboxBackdrop.hidden = true;
  document.body.style.overflow = "";
  state.lightbox.current = null;
}

el.lightboxClose.addEventListener("click", closeLightbox);
el.lightboxBackdrop.addEventListener("click", closeLightbox);
document.addEventListener("keydown", e => { if (e.key === "Escape") closeLightbox(); });

// Slider — blend original/depth by changing opacity
el.lbSlider.addEventListener("input", () => {
  const v = el.lbSlider.value / 100;
  el.lbOrigImg.style.opacity  = 1 - v * 0.6;
  el.lbDepthImg.style.opacity = 0.4 + v * 0.6;
});

el.lbDownloadDepth.addEventListener("click", () => {
  const r = state.lightbox.current;
  if (r) downloadBase64(`depth_${r.filename}`, r.depth_map);
});
el.lbDownloadGray.addEventListener("click", () => {
  const r = state.lightbox.current;
  if (r) downloadBase64(`gray_${r.filename}`, r.grayscale);
});

/* ═══════════════════════════════════════════════════════════
   MODEL COMPARISON PANEL
═══════════════════════════════════════════════════════════ */
let compareFile = null;

el.compareBrowseBtn.addEventListener("click", () => el.compareFileInput.click());
el.compareDropZone.addEventListener("click", () => el.compareFileInput.click());
el.compareDropZone.addEventListener("dragover", e => { e.preventDefault(); el.compareDropZone.classList.add("drag-over"); });
el.compareDropZone.addEventListener("dragleave", e => { if (!el.compareDropZone.contains(e.relatedTarget)) el.compareDropZone.classList.remove("drag-over"); });
el.compareDropZone.addEventListener("drop", e => {
  e.preventDefault();
  el.compareDropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) { compareFile = f; el.compareRunBtn.disabled = false; showToast(`Loaded: ${f.name}`); }
});

el.compareFileInput.addEventListener("change", () => {
  compareFile = el.compareFileInput.files[0];
  if (compareFile) { el.compareRunBtn.disabled = false; showToast(`Loaded: ${compareFile.name}`); }
});

el.compareRunBtn.addEventListener("click", runComparison);

async function runComparison() {
  if (!compareFile) return;

  const colormap = el.compareCmap.value;
  const models   = ["MiDaS_small", "DPT_Hybrid", "DPT_Large"];

  el.compareRunBtn.disabled       = true;
  el.compareProgressWrap.hidden   = false;
  el.compareResults.innerHTML     = "";

  const results = [];

  for (let i = 0; i < models.length; i++) {
    const model = models[i];
    el.compareProgressFill.style.width     = `${Math.round((i / models.length) * 100)}%`;
    el.compareProgressLabel.textContent    = `Running ${model}…`;

    try {
      const r = await inferSingle(compareFile, model, colormap);
      results.push(r);
      renderCompareCard(r);
    } catch (err) {
      showToast(`${model} failed: ${err.message}`, "error");
    }
  }

  el.compareProgressFill.style.width   = "100%";
  el.compareProgressLabel.textContent  = "Done!";
  setTimeout(() => { el.compareProgressWrap.hidden = true; }, 1500);
  el.compareRunBtn.disabled = false;

  if (results.length) {
    renderCompareChart(results);
    showToast("Comparison complete!", "success");
  }
}

function renderCompareCard(result) {
  // Remove placeholder
  const ph = $(".compare-placeholder");
  if (ph) ph.remove();

  const card = document.createElement("div");
  card.className = "compare-card";
  const label = result.model.replace("MiDaS_", "").replace("DPT_", "DPT ");
  card.innerHTML = `
    <div class="compare-card-header">
      ${label}
      <span class="latency-badge">${result.latency_ms} ms</span>
    </div>
    <img src="data:image/png;base64,${result.depth_map}" alt="Depth map — ${label}" loading="lazy" />
  `;
  el.compareResults.appendChild(card);
}

/* ═══════════════════════════════════════════════════════════
   TOAST NOTIFICATIONS
═══════════════════════════════════════════════════════════ */
function showToast(message, type = "info", duration = 3500) {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span class="toast-dot"></span><span>${escapeHtml(message)}</span>`;
  el.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = "toastOut 0.3s ease forwards";
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

/* ═══════════════════════════════════════════════════════════
   UTILS
═══════════════════════════════════════════════════════════ */
function downloadBase64(filename, b64) {
  const ext = filename.match(/\.[^.]+$/) ? "" : ".png";
  const a   = Object.assign(document.createElement("a"), {
    href:     `data:image/png;base64,${b64}`,
    download: filename + ext,
  });
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ═══════════════════════════════════════════════════════════
   INIT
═══════════════════════════════════════════════════════════ */
async function init() {
  initLatencyChart();
  switchPanel("main");
  updateQueueControls();

  // Initial health check + periodic polling
  await checkHealth();
  setInterval(checkHealth, 30_000);
}

init();