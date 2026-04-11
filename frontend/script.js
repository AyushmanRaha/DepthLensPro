/**
 * DepthLens Pro — Frontend v3.0
 * Fixes: browse button, cancel, timer ETA, server status clarity,
 *        device selector, full MDE metrics accordion, persistence.
 */
"use strict";

/* ═══════════════════════════════════════════
   CONFIG & STATE
═══════════════════════════════════════════ */
const API = "http://127.0.0.1:8000";

const state = {
  files:   [],        // { id, file, thumb, status, result }
  results: [],        // completed results
  session: { total: 0, cached: 0, errors: 0, latencies: [], totalInferenceMs: 0 },
  lb:      { current: null },
  abort:   null,      // AbortController for current batch
  compareAbort: null,
  compareFile:  null,
};

/* ═══════════════════════════════════════════
   DOM
═══════════════════════════════════════════ */
const $  = (s, ctx = document) => ctx.querySelector(s);
const $$ = (s, ctx = document) => [...ctx.querySelectorAll(s)];

const el = {
  // header
  statusDot:    $("#statusDot"),
  statusLabel:  $("#statusLabel"),
  statusSub:    $("#statusSub"),
  deviceBadge:  $("#deviceBadge"),
  navBtns:      $$(".nav-btn"),
  panels:       $$(".panel"),

  // device card
  deviceInfoBanner: $("#deviceInfoBanner"),
  deviceSelector:   $("#deviceSelector"),

  // upload
  dropZone:     $("#dropZone"),
  fileInput:    $("#fileInput"),
  fileQueue:    $("#fileQueue"),
  clearBtn:     $("#clearBtn"),
  cancelBtn:    $("#cancelBtn"),
  runBtn:       $("#runBtn"),

  // progress
  progressBlock:       $("#progressBlock"),
  progressFill:        $("#progressFill"),
  progressPct:         $("#progressPct"),
  progressBar:         $("#progressBar"),
  progressStatusText:  $("#progressStatusText"),
  progressEta:         $("#progressEta"),
  progressCurrentFile: $("#progressCurrentFile"),
  progressItemCount:   $("#progressItemCount"),

  // results
  resultsCard:     $("#resultsCard"),
  gallery:         $("#gallery"),
  downloadAllBtn:  $("#downloadAllBtn"),
  clearResultsBtn: $("#clearResultsBtn"),

  // metrics
  metricTotal:      $("#metricTotal"),
  metricAvgLatency: $("#metricAvgLatency"),
  metricCached:     $("#metricCached"),
  metricErrors:     $("#metricErrors"),
  metricMinLat:     $("#metricMinLat"),
  metricMaxLat:     $("#metricMaxLat"),
  metricThroughput: $("#metricThroughput"),
  metricTotalTime:  $("#metricTotalTime"),

  // compare
  compareDropZone:      $("#compareDropZone"),
  compareFileInput:     $("#compareFileInput"),
  compareFileName:      $("#compareFileName"),
  compareRunBtn:        $("#compareRunBtn"),
  compareCancelBtn:     $("#compareCancelBtn"),
  compareCmap:          $("#compareCmap"),
  compareDevice:        $("#compareDevice"),
  compareResults:       $("#compareResults"),
  compareProgressBlock: $("#compareProgressBlock"),
  compareProgressFill:  $("#compareProgressFill"),
  compareProgressPct:   $("#compareProgressPct"),
  compareProgressText:  $("#compareProgressText"),
  compareProgressEta:   $("#compareProgressEta"),
  compareChartCard:     $("#compareChartCard"),

  // lightbox
  lightbox:        $("#lightbox"),
  lightboxBackdrop:$("#lightboxBackdrop"),
  lightboxClose:   $("#lightboxClose"),
  lbOrigImg:       $("#lbOrigImg"),
  lbDepthImg:      $("#lbDepthImg"),
  lightboxMetrics: $("#lightboxMetrics"),
  lbSlider:        $("#lbSlider"),
  lbTags:          $("#lbTags"),
  lbDlDepth:       $("#lbDlDepth"),
  lbDlGray:        $("#lbDlGray"),

  toastContainer: $("#toastContainer"),
  bgCanvas:       $("#bgCanvas"),
};

/* ═══════════════════════════════════════════
   PERSISTENCE (localStorage)
═══════════════════════════════════════════ */
const PREFS_KEY = "depthlens_prefs_v3";

function savePrefs() {
  const prefs = {
    model:    $('input[name="model"]:checked')?.value    || "MiDaS_small",
    colormap: $('input[name="colormap"]:checked')?.value || "inferno",
    device:   $('input[name="device"]:checked')?.value   || "auto",
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
    const cmapEl  = $(`input[name="colormap"][value="${p.colormap}"]`);
    if (cmapEl)  cmapEl.checked  = true;
    // device restored after device list loads
    window._savedDevice = p.device;
  } catch {}
}

// Persist on every relevant input change
document.addEventListener("change", e => {
  if (["model","colormap","device"].includes(e.target.name)) savePrefs();
});

/* ═══════════════════════════════════════════
   BACKGROUND CANVAS
═══════════════════════════════════════════ */
(function bgCanvas() {
  const cv = el.bgCanvas, ctx = cv.getContext("2d");
  let W, H, pts = [];
  const N = 50;

  function mkP() {
    return { x: Math.random()*W, y: Math.random()*H,
             vx: (Math.random()-.5)*.28, vy: (Math.random()-.5)*.28,
             r: Math.random()*1.3+.3, a: Math.random() };
  }
  function resize() { W = cv.width = window.innerWidth; H = cv.height = window.innerHeight; }
  function reset()  { resize(); pts = Array.from({length:N}, mkP); }

  function draw() {
    ctx.clearRect(0,0,W,H);
    ctx.strokeStyle = "rgba(0,200,255,.055)";
    ctx.lineWidth   = 1;
    for (let x=0; x<W; x+=64) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
    for (let y=0; y<H; y+=64) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

    for (let i=0; i<pts.length; i++) {
      const p = pts[i];
      p.x = (p.x+p.vx+W)%W; p.y = (p.y+p.vy+H)%H;
      for (let j=i+1; j<pts.length; j++) {
        const q=pts[j], d=Math.hypot(p.x-q.x,p.y-q.y);
        if (d<130) {
          ctx.strokeStyle=`rgba(0,200,255,${.11*(1-d/130)})`;
          ctx.lineWidth=.55;
          ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke();
        }
      }
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(0,200,255,${p.a*.55})`; ctx.fill();
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  reset(); draw();
})();

/* ═══════════════════════════════════════════
   CHARTS
═══════════════════════════════════════════ */
let latencyChart, compareChart;

function initLatencyChart() {
  latencyChart = new Chart($("#latencyChart").getContext("2d"), {
    type: "line",
    data: { labels:[], datasets:[{
      label: "Inference ms",
      data: [], borderColor:"#00c8ff", backgroundColor:"rgba(0,200,255,.07)",
      borderWidth: 1.5, pointRadius: 2, tension: .4, fill: true,
    }]},
    options: {
      responsive: true, maintainAspectRatio: false, animation: {duration:280},
      plugins: { legend:{display:false}, tooltip:{
        backgroundColor:"#101e2e", borderColor:"#00c8ff", borderWidth:1,
        titleColor:"#7faac8", bodyColor:"#dff0ff",
        callbacks:{ label: c=>`${c.raw} ms` },
      }},
      scales: {
        x:{display:false},
        y:{display:true, grid:{color:"rgba(0,200,255,.07)"},
           ticks:{color:"#3a5a72", font:{family:"JetBrains Mono",size:9}, maxTicksLimit:4}},
      },
    },
  });
}

function pushLatency(ms) {
  const d = state.session.latencies.slice(-20);
  latencyChart.data.labels   = d.map((_,i)=>i+1);
  latencyChart.data.datasets[0].data = d;
  latencyChart.update("none");
}

function renderCompareChart(results) {
  el.compareChartCard.hidden = false;
  const ctx = $("#compareChart").getContext("2d");
  if (compareChart) compareChart.destroy();
  compareChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: results.map(r=>r.model.replace("MiDaS_","").replace("DPT_","DPT ")),
      datasets:[{
        label:"Latency (ms)", data: results.map(r=>r.latency_ms),
        backgroundColor:["rgba(0,200,255,.55)","rgba(123,92,248,.55)","rgba(255,107,107,.55)"],
        borderColor:["#00c8ff","#7b5cf8","#ff6b6b"], borderWidth:1.5, borderRadius:4,
      }],
    },
    options:{
      responsive:true, maintainAspectRatio:false, animation:{duration:380},
      plugins:{legend:{labels:{color:"#7faac8",font:{family:"JetBrains Mono",size:10}}},
               tooltip:{backgroundColor:"#101e2e",borderColor:"#00c8ff",borderWidth:1,titleColor:"#7faac8",bodyColor:"#dff0ff"}},
      scales:{
        x:{ticks:{color:"#7faac8",font:{family:"Rajdhani",size:12,weight:"600"}}, grid:{color:"rgba(0,200,255,.07)"}},
        y:{ticks:{color:"#3a5a72",font:{family:"JetBrains Mono",size:9},callback:v=>`${v}ms`}, grid:{color:"rgba(0,200,255,.07)"}},
      },
    },
  });
}

/* ═══════════════════════════════════════════
   SERVER HEALTH + DEVICE LIST
   Status indicator is clearly about the LOCAL
   FastAPI inference engine (not a remote server).
═══════════════════════════════════════════ */
function setStatus(cls, label, sub, deviceText) {
  el.statusDot.className   = `status-dot ${cls}`;
  el.statusLabel.textContent = label;
  el.statusSub.textContent   = sub || "";
  if (deviceText) {
    el.deviceBadge.textContent = deviceText;
    el.deviceBadge.hidden = false;
  } else {
    el.deviceBadge.hidden = true;
  }
}

async function checkHealth() {
  setStatus("connecting", "Connecting…", "localhost:8000");
  try {
    const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Build device badge text
    const devs    = data.devices || {};
    const primary = data.primary_device || "cpu";
    const info    = devs[primary];
    let badge = "CPU";
    if (info?.type === "cuda") badge = `GPU: ${info.name} (${info.memory_gb} GB)`;
    else if (info?.type === "mps") badge = `Apple ${info.chip} · Neural Engine`;

    setStatus("online", "Depth Engine: Online", `PyTorch ${data.torch_version} · ${primary}`, badge);

    // Populate device selector
    await loadDevices(devs);
    updateDeviceInfoBanner(badge);
    return true;
  } catch {
    setStatus("offline", "Depth Engine: Offline", "Run: uvicorn app:app --reload");
    el.deviceInfoBanner.querySelector("span:last-child").textContent =
      "Start the FastAPI engine to detect available devices";
    return false;
  }
}

async function loadDevices(devs) {
  if (!devs || !Object.keys(devs).length) {
    try {
      const r = await fetch(`${API}/devices`, { signal: AbortSignal.timeout(3000) });
      devs = (await r.json()).devices;
    } catch { return; }
  }

  // Workspace device selector
  el.deviceSelector.innerHTML = "";
  const all = { auto: { name: "Auto (Best Available)", type: "auto" }, ...devs };
  Object.entries(all).forEach(([key, info]) => {
    const isCuda = info.type === "cuda";
    const isMps  = info.type === "mps";
    const icon   = key === "auto" ? "🔀" : isCuda ? "🎮" : isMps ? "🍎" : "🖥";
    const sub    = key === "auto"  ? "Picks best: MPS > CUDA > CPU"
             : isCuda ? `CUDA · ${info.memory_gb} GB VRAM`
             : isMps  ? `Apple ${info.chip} · Metal Performance Shaders + ANE`
             : "System processor";
    const saved  = window._savedDevice;
    const checked = (saved && saved === key) || (!saved && key === "auto");

    const lbl = document.createElement("label");
    lbl.className = "device-opt" + (checked ? " selected" : "");
    lbl.dataset.device = key;
    lbl.innerHTML = `
      <input type="radio" name="device" value="${key}" ${checked ? "checked" : ""} />
      <div class="device-opt-inner">
        <span class="device-opt-icon">${icon}</span>
        <div>
          <span class="device-opt-name">${esc(info.name || key)}</span>
          <span class="device-opt-sub">${sub}</span>
        </div>
      </div>`;
    el.deviceSelector.appendChild(lbl);
  });

  // Highlight selected
  $$(".device-opt input", el.deviceSelector).forEach(inp => {
    inp.addEventListener("change", () => {
      $$(".device-opt", el.deviceSelector).forEach(l => l.classList.remove("selected"));
      inp.closest(".device-opt").classList.add("selected");
      savePrefs();
    });
  });

  // Compare panel device dropdown
  el.compareDevice.innerHTML = "";
  Object.entries(all).forEach(([key, info]) => {
    const opt = document.createElement("option");
    opt.value = key; opt.textContent = info.name || key;
    el.compareDevice.appendChild(opt);
  });
}

function updateDeviceInfoBanner(text) {
  el.deviceInfoBanner.querySelector("span:last-child").textContent =
    `Detected: ${text}. Choose compute target below.`;
}

/* ═══════════════════════════════════════════
   PANEL NAVIGATION
═══════════════════════════════════════════ */
function switchPanel(name) {
  el.navBtns.forEach(b => b.classList.toggle("active", b.dataset.panel === name));
  el.panels.forEach(p  => { p.hidden = p.id !== `panel-${name}`; });
}
el.navBtns.forEach(b => b.addEventListener("click", () => switchPanel(b.dataset.panel)));

/* ═══════════════════════════════════════════
   FILE HANDLING  — browse button fixed via
   <label for="fileInput"> approach in HTML.
   No programmatic .click() needed.
═══════════════════════════════════════════ */
const ALLOWED = /^image\//;

function uid() { return Math.random().toString(36).slice(2,9); }
function fmtSize(b) {
  if (b < 1024)      return `${b} B`;
  if (b < 1048576)   return `${(b/1024).toFixed(1)} KB`;
  return `${(b/1048576).toFixed(1)} MB`;
}

function addFiles(list) {
  let added = 0;
  for (const file of list) {
    if (!ALLOWED.test(file.type)) { toast(`Skipped "${file.name}" — not an image`, "warning"); continue; }
    if (file.size > 20*1024*1024) { toast(`Skipped "${file.name}" — exceeds 20 MB`, "warning"); continue; }
    if (state.files.some(f=>f.file.name===file.name && f.file.size===file.size)) continue;

    const entry = { id:uid(), file, thumb:null, status:"pending", result:null };
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
  li.className = "file-item"; li.id = `fitem-${entry.id}`;
  li.innerHTML = `
    <img class="file-thumb" id="fthumb-${entry.id}" src="" alt="" />
    <div class="file-meta">
      <div class="file-name" title="${esc(entry.file.name)}">${esc(entry.file.name)}</div>
      <div class="file-size">${fmtSize(entry.file.size)}</div>
    </div>
    <span class="file-status pending" id="fst-${entry.id}">Pending</span>
    <button class="file-remove" data-id="${entry.id}" aria-label="Remove ${esc(entry.file.name)}">✕</button>`;
  el.fileQueue.appendChild(li);
  $(".file-remove", li).addEventListener("click", () => removeFile(entry.id));
}

function removeFile(id) {
  if (state.files.find(f=>f.id===id)?.status === "running") return;
  state.files = state.files.filter(f=>f.id!==id);
  $(`#fitem-${id}`)?.remove();
  syncQueueControls();
}

function syncQueueControls() {
  const has = state.files.length > 0;
  el.clearBtn.disabled = !has;
  el.runBtn.disabled   = !has;
}

function setFileSt(id, cls, txt) {
  const s = $(`#fst-${id}`);
  if (!s) return;
  s.className = `file-status ${cls}`; s.textContent = txt;
}

// File input change (handles both label-click and drag-drop onto the input)
el.fileInput.addEventListener("change", () => { addFiles(el.fileInput.files); el.fileInput.value=""; });

// Drag & Drop onto drop zone
el.dropZone.addEventListener("dragover", e=>{ e.preventDefault(); el.dropZone.classList.add("drag-over"); });
el.dropZone.addEventListener("dragleave", e=>{ if(!el.dropZone.contains(e.relatedTarget)) el.dropZone.classList.remove("drag-over"); });
el.dropZone.addEventListener("drop", e=>{ e.preventDefault(); el.dropZone.classList.remove("drag-over"); addFiles(e.dataTransfer.files); });
// Keyboard accessibility
el.dropZone.addEventListener("keydown", e=>{ if(e.key==="Enter"||e.key===" "){ e.preventDefault(); el.fileInput.click(); } });

el.clearBtn.addEventListener("click", ()=>{ state.files=[]; el.fileQueue.innerHTML=""; syncQueueControls(); });

/* ═══════════════════════════════════════════
   INFERENCE  — with cancel + ETA timer
═══════════════════════════════════════════ */
function selModel()   { return $('input[name="model"]:checked')?.value || "MiDaS_small"; }
function selCmap()    { return $('input[name="colormap"]:checked')?.value || "inferno"; }
function selDevice()  { return $('input[name="device"]:checked')?.value || "auto"; }

function setProgress(pct, status, eta, currentFile, countStr) {
  el.progressFill.style.width = `${pct}%`;
  el.progressPct.textContent  = `${Math.round(pct)}%`;
  if (el.progressBar) el.progressBar.setAttribute("aria-valuenow", pct);
  el.progressStatusText.textContent = status;
  el.progressEta.textContent        = eta || "";
  if (currentFile !== undefined) el.progressCurrentFile.textContent = currentFile;
  if (countStr    !== undefined) el.progressItemCount.textContent   = countStr;
}

function fmtDuration(ms) {
  const s = Math.round(ms / 1000);
  if (s < 60)  return `${s}s`;
  if (s < 3600) return `${Math.floor(s/60)}m ${s%60}s`;
  return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m ${s%60}s`;
}

el.runBtn.addEventListener("click", runBatch);
el.cancelBtn.addEventListener("click", cancelBatch);

function cancelBatch() {
  if (state.abort) { state.abort.abort(); state.abort = null; }
  toast("Batch cancelled", "warning");
}

async function runBatch() {
  const pending = state.files.filter(f=>f.status==="pending"||f.status==="error");
  if (!pending.length) return;

  state.abort = new AbortController();

  el.runBtn.disabled   = true;
  el.clearBtn.disabled = true;
  el.cancelBtn.hidden  = false;
  el.progressBlock.hidden = false;
  setProgress(0, "Initialising…", "", "", `0 / ${pending.length}`);

  const batchStart = Date.now();
  const model    = selModel();
  const colormap = selCmap();
  const device   = selDevice();

  for (let i = 0; i < pending.length; i++) {
    if (state.abort?.signal?.aborted) break;

    const entry = pending[i];
    setFileSt(entry.id, "running", "Running…");
    const pct = (i / pending.length) * 100;

    // ETA calculation
    let eta = "";
    if (i > 0) {
      const elapsed = Date.now() - batchStart;
      const rate    = elapsed / i;
      const rem     = rate * (pending.length - i);
      eta = `ETA: ${fmtDuration(rem)}`;
    }

    setProgress(
      pct, `Processing ${i+1} of ${pending.length}…`,
      eta,
      esc(entry.file.name),
      `${i+1} / ${pending.length}`
    );

    try {
      const result = await inferOne(entry.file, model, colormap, device, state.abort.signal);
      entry.result = result; entry.status = "done";
      setFileSt(entry.id, "done", `✓ ${result.latency_ms}ms`);

      state.session.total++;
      state.session.totalInferenceMs += result.latency_ms;
      if (result.cached) state.session.cached++;
      state.session.latencies.push(result.latency_ms);
      updateMetrics();
      pushLatency(result.latency_ms);

      state.results.push({ ...result, originalSrc: entry.thumb, filename: entry.file.name });
      appendGalleryItem(state.results.at(-1));
      el.resultsCard.hidden = false;

    } catch (err) {
      if (err.name === "AbortError") { setFileSt(entry.id, "pending", "Cancelled"); break; }
      entry.status = "error";
      setFileSt(entry.id, "error", "Error");
      state.session.errors++;
      updateMetrics();
      toast(`"${entry.file.name}": ${err.message}`, "error");
    }
  }

  const elapsed = Date.now() - batchStart;
  const done    = pending.filter(e=>e.status==="done").length;
  setProgress(100, `Done — ${done} image${done!==1?"s":""} in ${fmtDuration(elapsed)}`, "");

  setTimeout(() => { el.progressBlock.hidden = true; }, 3000);
  el.runBtn.disabled   = false;
  el.clearBtn.disabled = false;
  el.cancelBtn.hidden  = true;
  state.abort = null;

  if (done > 0) toast(`Batch complete — ${done} succeeded`, "success");
}

async function inferOne(file, model, colormap, device, signal) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("model", model);
  fd.append("colormap", colormap);
  fd.append("device", device);

  const res = await fetch(`${API}/estimate`, {
    method: "POST", body: fd,
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(180_000)]) : AbortSignal.timeout(180_000),
  });
  if (!res.ok) {
    const err = await res.json().catch(()=>({detail:`HTTP ${res.status}`}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/* ═══════════════════════════════════════════
   METRICS DASHBOARD
═══════════════════════════════════════════ */
function updateMetrics() {
  const s   = state.session;
  const lats = s.latencies;
  el.metricTotal.textContent    = s.total;
  el.metricCached.textContent   = s.cached;
  el.metricErrors.textContent   = s.errors;
  el.metricTotalTime.textContent = `${(s.totalInferenceMs/1000).toFixed(1)} s`;

  if (lats.length) {
    const avg = lats.reduce((a,b)=>a+b,0) / lats.length;
    el.metricAvgLatency.textContent = avg.toFixed(0);
    el.metricMinLat.textContent     = Math.min(...lats).toFixed(0);
    el.metricMaxLat.textContent     = Math.max(...lats).toFixed(0);
    el.metricThroughput.textContent = (60000 / avg).toFixed(1);
  }
}

/* ═══════════════════════════════════════════
   GALLERY
═══════════════════════════════════════════ */
function appendGalleryItem(r) {
  const m = r.metrics || {};
  const item = Object.assign(document.createElement("div"), {
    className: "gallery-item",
  });
  item.setAttribute("role","listitem");
  item.setAttribute("tabindex","0");
  item.innerHTML = `
    <div class="gallery-img-wrap">
      <img src="data:image/png;base64,${r.depth_map}" alt="Depth map — ${esc(r.filename)}" loading="lazy"/>
      <div class="gallery-overlay">🔍</div>
    </div>
    <div class="gallery-meta">
      <div class="gallery-filename" title="${esc(r.filename)}">${esc(r.filename)}</div>
      <div class="gallery-tags">
        <span class="gallery-tag">${r.model?.replace("MiDaS_","").replace("DPT_","DPT ")}</span>
        <span class="gallery-tag">${r.colormap}</span>
        <span class="gallery-tag">${r.device_used || ""}</span>
        ${r.cached?'<span class="gallery-tag">cached</span>':""}
      </div>
      <div class="gallery-stats-row">
        <span>Latency <strong>${r.latency_ms}ms</strong></span>
        <span>SSIM <strong>${(m.ssim??0).toFixed(3)}</strong></span>
        <span>${r.resolution?.width}×${r.resolution?.height}</span>
      </div>
    </div>`;
  item.addEventListener("click",  ()=>openLightbox(r));
  item.addEventListener("keydown", e=>{ if(e.key==="Enter"||e.key===" ") openLightbox(r); });
  el.gallery.appendChild(item);
}

el.clearResultsBtn.addEventListener("click", ()=>{ state.results=[]; el.gallery.innerHTML=""; el.resultsCard.hidden=true; });
el.downloadAllBtn.addEventListener("click",  ()=>{
  if(!state.results.length) return;
  state.results.forEach(r=>dlB64(`depth_${r.filename}`, r.depth_map));
  toast(`Downloading ${state.results.length} depth maps…`);
});

/* ═══════════════════════════════════════════
   LIGHTBOX + MDE METRICS ACCORDION
═══════════════════════════════════════════ */

/**
 * Metric definitions — label, description, unit, whether it needs GT.
 * Organised into display groups for the accordion.
 */
const METRIC_GROUPS = [
  {
    id: "error",
    icon: "📏",
    label: "Core Error Metrics",
    note: "Computed from predicted depth distribution only (no ground truth required). Values reflect self-consistency of the depth map.",
    metrics: [
      { key:"mae",      label:"MAE (Mean Absolute Error)",        unit:"",   desc:"Average magnitude of deviation from depth mean. Lower = more uniform depth field. High MAE suggests a wide depth range (which can be desirable).", needsGT:false },
      { key:"rmse",     label:"RMSE (Root Mean Squared Error)",   unit:"",   desc:"Square-root of mean squared deviation from depth mean. Penalises large outliers more than MAE.", needsGT:false },
      { key:"log_rmse", label:"Log RMSE",                         unit:"",   desc:"RMSE computed in log-depth space — more sensitive to relative errors at small depth values.", needsGT:false },
      { key:"abs_rel",  label:"Absolute Relative Error (Abs Rel)",unit:"",   desc:"Mean of |pred−GT|/GT. Standard MDE benchmark metric. Requires ground-truth depth.", needsGT:true },
      { key:"sq_rel",   label:"Squared Relative Error (Sq Rel)",  unit:"",   desc:"Mean of (pred−GT)²/GT. Penalises large relative errors. Requires ground-truth depth.", needsGT:true },
    ],
  },
  {
    id: "accuracy",
    icon: "📐",
    label: "Threshold Accuracy",
    note: "δ metrics require ground-truth depth maps and cannot be computed here. They measure the fraction of pixels where max(pred/GT, GT/pred) < threshold.",
    metrics: [
      { key:"delta_1", label:"δ < 1.25¹",   unit:"%", desc:"Fraction of pixels within 25% scale of ground truth. Typical MDE benchmark metric. Requires GT.", needsGT:true },
      { key:"delta_2", label:"δ < 1.25²",   unit:"%", desc:"Looser threshold (56%). Requires GT.", needsGT:true },
      { key:"delta_3", label:"δ < 1.25³",   unit:"%", desc:"Loosest threshold (95%). Requires GT.", needsGT:true },
    ],
  },
  {
    id: "scaleinv",
    icon: "🧭",
    label: "Scale-Invariant Metrics",
    metrics: [
      { key:"silog",         label:"SILog (Scale-Invariant Log Error)", unit:"",   desc:"Measures log-depth variance after removing global scale. Lower = better structure preservation. Commonly used on KITTI / NYU benchmarks.", needsGT:false },
      { key:"dynamic_range", label:"Dynamic Range",                      unit:" bits", desc:"Log₂ ratio of max/min non-zero depth. Larger = more depth variation captured. ~8–14 bits is typical for natural scenes.", needsGT:false },
      { key:"entropy",       label:"Shannon Entropy",                    unit:" bits", desc:"Entropy of the depth histogram (bits). Higher = more uniformly distributed depth values. Low entropy indicates most pixels cluster at a single depth.", needsGT:false },
      { key:"coverage",      label:"Depth Coverage",                     unit:"%",     desc:"Fraction of histogram bins with ≥1% of peak count. Higher = depth values spread across the full range.", needsGT:false, pct:true },
    ],
  },
  {
    id: "structural",
    icon: "🧱",
    label: "Structural & Geometric Metrics",
    metrics: [
      { key:"ssim",           label:"SSIM (Structural Similarity)",  unit:"",  desc:"Compares depth map structure to grayscale input. Range 0–1; higher means the depth map preserves the luminance structure of the scene (edges, textures).", needsGT:false },
      { key:"gradient_mean",  label:"Gradient Mean",                 unit:"",  desc:"Mean Sobel gradient magnitude over the depth map. Higher = more depth edges / transitions. Low values indicate blurry or featureless depth maps.", needsGT:false },
      { key:"gradient_std",   label:"Gradient Std Dev",              unit:"",  desc:"Variation in gradient strength. High std means some regions have sharp edges while others are smooth.", needsGT:false },
      { key:"gradient_error", label:"Gradient Error",                unit:"",  desc:"Proxy for edge detail in the depth map (mean gradient). Higher = more preserved depth boundaries.", needsGT:false },
      { key:"edge_density",   label:"Edge Density",                  unit:"%", desc:"Fraction of pixels with gradient > mean+std. Indicates how richly detailed the depth edges are.", needsGT:false, pct:true },
      { key:"surface_normal_error", label:"Surface Normal Error",    unit:"",  desc:"Requires ground-truth normals derived from GT depth. Not computable without GT.", needsGT:true },
    ],
  },
  {
    id: "perceptual",
    icon: "👁",
    label: "Perceptual & Consistency Metrics",
    metrics: [
      { key:"psnr",  label:"PSNR (Peak Signal-to-Noise Ratio)", unit:" dB", desc:"Signal quality of the depth map relative to its own mean. Higher = less self-noise. Computed in depth space; not directly comparable to image PSNR.", needsGT:false },
      { key:"lpips", label:"LPIPS (Perceptual Similarity)",     unit:"",    desc:"Learned perceptual metric based on deep features. Requires a reference depth map or image. Not computable without GT.", needsGT:true },
    ],
  },
  {
    id: "ranking",
    icon: "🧪",
    label: "Ranking / Relative Depth Metrics",
    metrics: [
      { key:"ordinal_error", label:"Ordinal Error",          unit:"", desc:"Fraction of pixel pairs where the relative ordering of pred depth disagrees with GT. Key metric for monocular methods. Requires GT.", needsGT:true },
      { key:"abs_rel",       label:"Relative Depth Accuracy",unit:"", desc:"Accuracy of relative depth relationships. Requires GT reference depth map.", needsGT:true },
    ],
  },
];

function valColor(key, val) {
  // Heuristic colouring for known metrics
  if (val === null || val === undefined) return "na";
  if (key === "ssim")    return val > .7 ? "good" : val > .4 ? "warn" : "bad";
  if (key === "silog")   return val < 10 ? "good" : val < 25 ? "warn" : "bad";
  if (key === "psnr")    return val > 30 ? "good" : val > 15 ? "warn" : "bad";
  if (key === "edge_density" || key === "coverage") return "";
  return "";
}

function renderMetricsAccordion(metrics) {
  el.lightboxMetrics.innerHTML = "";

  METRIC_GROUPS.forEach((group, gi) => {
    const div = document.createElement("div");
    div.className = "metric-group";
    div.id = `mg-${group.id}`;

    const hdr = document.createElement("div");
    hdr.className = "metric-group-header";
    hdr.setAttribute("role","button");
    hdr.setAttribute("tabindex","0");
    hdr.setAttribute("aria-expanded","false");
    hdr.innerHTML = `
      <span><span class="mg-icon">${group.icon}</span>${group.label}</span>
      <span class="mg-toggle">▾</span>`;
    hdr.addEventListener("click", () => toggleAccordion(div));
    hdr.addEventListener("keydown", e=>{ if(e.key==="Enter"||e.key===" "){ e.preventDefault(); toggleAccordion(div); } });

    const body = document.createElement("div");
    body.className = "metric-group-body";
    body.setAttribute("role","region");

    const content = document.createElement("div");
    content.className = "metric-group-content";

    if (group.note) {
      const noteEl = document.createElement("p");
      noteEl.style.cssText = "font-family:var(--ff-mono);font-size:.6rem;color:var(--text-dim);margin-bottom:.4rem;line-height:1.4;";
      noteEl.textContent = group.note;
      content.appendChild(noteEl);
    }

    // De-duplicate keys per group
    const seen = new Set();
    group.metrics.forEach(m => {
      if (seen.has(m.key)) return; seen.add(m.key);
      const raw = metrics?.[m.key];
      const isNull = raw === null || raw === undefined;
      let valText, cls;
      if (m.needsGT && isNull) { valText="N/A (needs GT)"; cls="na"; }
      else if (isNull)          { valText="—"; cls="na"; }
      else if (m.pct)           { valText=`${(raw*100).toFixed(1)}%`; cls=valColor(m.key,raw); }
      else                      { valText=`${raw}${m.unit||""}`; cls=valColor(m.key,raw); }

      const row = document.createElement("div"); row.className="metric-row";
      row.innerHTML=`
        <div class="metric-row-left">
          <span class="metric-row-name">${m.label}</span>
          <span class="metric-row-desc">${m.desc}</span>
        </div>
        <span class="metric-row-val ${cls}">${valText}</span>`;
      content.appendChild(row);
    });

    body.appendChild(content);
    div.appendChild(hdr);
    div.appendChild(body);
    el.lightboxMetrics.appendChild(div);

    // Open first group by default
    if (gi === 0) toggleAccordion(div);
  });
}

function toggleAccordion(div) {
  const isOpen = div.classList.contains("open");
  div.classList.toggle("open", !isOpen);
  div.querySelector(".metric-group-header").setAttribute("aria-expanded", String(!isOpen));
}

function openLightbox(r) {
  state.lb.current = r;
  el.lbOrigImg.src  = r.originalSrc || "";
  el.lbDepthImg.src = `data:image/png;base64,${r.depth_map}`;
  el.lbSlider.value = 50;

  // Tags
  el.lbTags.innerHTML = [
    r.model?.replace("MiDaS_","").replace("DPT_","DPT "),
    r.colormap, r.device_used,
    `${r.latency_ms} ms`,
    r.cached ? "cached" : null,
    `${r.resolution?.width}×${r.resolution?.height}`,
  ].filter(Boolean).map(t=>`<span class="lb-tag">${esc(t)}</span>`).join("");

  renderMetricsAccordion(r.metrics);

  el.lightbox.hidden = false;
  el.lightboxBackdrop.hidden = false;
  document.body.style.overflow = "hidden";
  el.lightboxClose.focus();
}

function closeLightbox() {
  el.lightbox.hidden = true;
  el.lightboxBackdrop.hidden = true;
  document.body.style.overflow = "";
  state.lb.current = null;
}

el.lightboxClose.addEventListener("click", closeLightbox);
el.lightboxBackdrop.addEventListener("click", closeLightbox);
document.addEventListener("keydown", e => { if(e.key==="Escape") closeLightbox(); });

el.lbSlider.addEventListener("input", () => {
  const v = el.lbSlider.value / 100;
  el.lbOrigImg.style.opacity  = 1 - v*.65;
  el.lbDepthImg.style.opacity = .35 + v*.65;
});

el.lbDlDepth.addEventListener("click", () => { const r=state.lb.current; if(r) dlB64(`depth_${r.filename}`, r.depth_map); });
el.lbDlGray.addEventListener("click",  () => { const r=state.lb.current; if(r) dlB64(`gray_${r.filename}`,  r.grayscale);  });

/* ═══════════════════════════════════════════
   COMPARE PANEL
═══════════════════════════════════════════ */
el.compareFileInput.addEventListener("change", () => {
  state.compareFile = el.compareFileInput.files[0];
  if (state.compareFile) {
    el.compareFileName.textContent = state.compareFile.name;
    el.compareRunBtn.disabled = false;
    toast(`Loaded: ${state.compareFile.name}`);
  }
});

el.compareDropZone.addEventListener("dragover", e=>{e.preventDefault(); el.compareDropZone.classList.add("drag-over");});
el.compareDropZone.addEventListener("dragleave", e=>{ if(!el.compareDropZone.contains(e.relatedTarget)) el.compareDropZone.classList.remove("drag-over");});
el.compareDropZone.addEventListener("drop", e=>{
  e.preventDefault(); el.compareDropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f?.type.startsWith("image/")) { state.compareFile=f; el.compareFileName.textContent=f.name; el.compareRunBtn.disabled=false; toast(`Loaded: ${f.name}`); }
});
el.compareDropZone.addEventListener("click", ()=>el.compareFileInput.click());

el.compareRunBtn.addEventListener("click",   runComparison);
el.compareCancelBtn.addEventListener("click", ()=>{ state.compareAbort?.abort(); toast("Comparison cancelled","warning"); });

async function runComparison() {
  if (!state.compareFile) return;
  const cmap   = el.compareCmap.value;
  const device = el.compareDevice.value;
  const models = ["MiDaS_small","DPT_Hybrid","DPT_Large"];

  state.compareAbort = new AbortController();
  el.compareRunBtn.disabled     = true;
  el.compareCancelBtn.hidden    = false;
  el.compareProgressBlock.hidden = false;
  el.compareResults.innerHTML   = "";

  const results = [];
  const t0 = Date.now();

  for (let i=0; i<models.length; i++) {
    if (state.compareAbort.signal.aborted) break;
    const model = models[i];
    const pct   = Math.round((i/models.length)*100);
    el.compareProgressFill.style.width = `${pct}%`;
    el.compareProgressPct.textContent  = `${pct}%`;
    el.compareProgressText.textContent = `Running ${model}…`;
    if (i>0) {
      const elapsed = Date.now()-t0;
      const rem     = (elapsed/i)*(models.length-i);
      el.compareProgressEta.textContent = `ETA: ${fmtDuration(rem)}`;
    }
    try {
      const r = await inferOne(state.compareFile, model, cmap, device, state.compareAbort.signal);
      results.push(r);
      renderCompareCard(r);
    } catch(err) {
      if (err.name!=="AbortError") toast(`${model} failed: ${err.message}`, "error");
    }
  }

  el.compareProgressFill.style.width = "100%";
  el.compareProgressPct.textContent  = "100%";
  el.compareProgressText.textContent = state.compareAbort.signal.aborted ? "Cancelled" : "Done!";
  el.compareProgressEta.textContent  = `Total: ${fmtDuration(Date.now()-t0)}`;
  setTimeout(()=>{ el.compareProgressBlock.hidden=true; }, 2500);
  el.compareRunBtn.disabled  = false;
  el.compareCancelBtn.hidden = true;
  state.compareAbort = null;

  if (results.length) { renderCompareChart(results); toast("Comparison complete!","success"); }
}

function renderCompareCard(r) {
  $(".compare-placeholder")?.remove();
  const card = document.createElement("div"); card.className="compare-card";
  const lbl  = r.model.replace("MiDaS_","").replace("DPT_","DPT ");
  card.innerHTML=`
    <div class="compare-card-header">
      ${lbl} <span class="latency-badge">${r.latency_ms} ms</span>
    </div>
    <img src="data:image/png;base64,${r.depth_map}" alt="Depth map — ${lbl}" loading="lazy"/>`;
  el.compareResults.appendChild(card);
}

/* ═══════════════════════════════════════════
   TOAST
═══════════════════════════════════════════ */
function toast(msg, type="info", dur=3500) {
  const t = document.createElement("div");
  t.className = `toast ${type}`;
  t.innerHTML = `<span class="toast-dot"></span><span>${esc(msg)}</span>`;
  el.toastContainer.appendChild(t);
  setTimeout(()=>{ t.style.animation="toastOut .28s ease forwards"; setTimeout(()=>t.remove(),280); }, dur);
}

/* ═══════════════════════════════════════════
   UTILS
═══════════════════════════════════════════ */
function dlB64(name, b64) {
  const a = Object.assign(document.createElement("a"), {
    href: `data:image/png;base64,${b64}`,
    download: name.match(/\.[^.]+$/) ? name : name+".png",
  });
  document.body.appendChild(a); a.click(); a.remove();
}

function esc(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

/* ═══════════════════════════════════════════
   INIT
═══════════════════════════════════════════ */
async function init() {
  loadPrefs();
  initLatencyChart();
  switchPanel("main");
  syncQueueControls();
  await checkHealth();
  setInterval(checkHealth, 30_000);
}

init();