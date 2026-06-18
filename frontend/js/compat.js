"use strict";

window.addEventListener("pagehide", () => {
  if (settings.autoClearResultsOnClose) { state.results = []; if (el.gallery) el.gallery.innerHTML = ""; }
  stopWebcam({ quiet: true });
  [...activeBlobUrls].forEach(revokeBlobUrl);
});

function exportExperimentJson() {
  const payload = {
    run_name: state.experiment.name,
    started_at: state.experiment.startedAt,
    exported_at: new Date().toISOString(),
    results: settings.stripFilenamesFromExports ? state.experiment.results.map(r => ({ ...r, filename: undefined })) : state.experiment.results,
  };

  exportBlob(
    `${state.experiment.name.replace(/[^a-z0-9_-]+/gi, "_")}.json`,
    "application/json",
    JSON.stringify(payload, null, 2)
  );
}

function exportExperimentCsv() {
  const rows = settings.stripFilenamesFromExports ? experimentRows().map(r => ({ ...r, filename: "" })) : experimentRows();
  const header = ["filename", "model", "engine", "device", "latency_ms", "abs_rel", "rmse", "delta_1", "gt", "fallback", "warnings"];

  const csv = [
    header.join(","),
    ...rows.map(r => header.map(k => `"${String(r[k] ?? "").replace(/"/g, '""')}"`).join(",")),
  ].join("\n");

  exportBlob(
    `${(state.experiment.name || "experiment").replace(/[^a-z0-9_-]+/gi, "_")}.csv`,
    "text/csv",
    csv
  );
}

el.experimentRunBtn?.addEventListener("click", runExperiment);
el.experimentExportJsonBtn?.addEventListener("click", exportExperimentJson);
el.experimentExportCsvBtn?.addEventListener("click", exportExperimentCsv);
// ══════════════════════════════════════════════════════════════
// TOAST
// ══════════════════════════════════════════════════════════════
function toastOnce(msg, type="info", dur=3500) {
  const now = Date.now();
  if (settings.dedupeWarnings && state.lastToast.message === msg && now - state.lastToast.at < 15000) return;
  state.lastToast = { message: msg, at: now };
  toast(msg, type, dur);
}

function toast(msg, type="info", dur=3500) {
  if (!toastAllowed(type)) return;
  dur = toastDurationMs(typeof dur === "number" && dur !== 3500 ? dur : settings.toastDuration);
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
  addPollingInterval("diagnostics", diagnosticsIntervalMs(), signal => checkDiagnostics({ quiet: true, signal }));
  addPollingInterval("cacheMetrics", Math.max(30_000, diagnosticsIntervalMs() * 0.75), signal => loadCacheMetrics({ signal }));
  addPollingInterval("observability", 20_000, signal => { if (backendOnline && $(".nav-btn.active")?.dataset.panel === "performance" && state.performanceView === "observability" && !document.hidden) return loadObservability({ quiet: true, signal }); });
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
  applySettings({ notify: false });
  setStatus("connecting", "Starting depth engine", DEFAULT_API_BASE_URL);
  syncQueueControls();
  initReconstructionPanel();
  initSettingsPanel();
  initScrollableNav();
  initGuideAccordion();
  bindPointerGlow(".logo-group", { tilt: 3 });
  bindPointerGlow(".nav-btn", { tilt: 5 });
  bindPointerGlow(".guide-section-toggle", { tilt: 3 });
  bindPointerGlow(".guide-card, .guide-section", { tilt: 1.5 });
  try {
    await resolveApiBaseUrl();
    if (!runningInElectron) toastOnce("Browser mode detected — start the depth engine manually for inference", "warning", 7000);
    initLatencyChart();
    initCompareControls();
    initEngineStatusPanel();
    syncWebcamControls();
    updateWebcamTelemetry();
    const savedPanel = settings.rememberLastTab ? (() => { try { return localStorage.getItem(LAST_TAB_KEY) || "main"; } catch { return "main"; } })() : "main";
    switchPanel(savedPanel);
    if (settings.skipWelcome && el.welcomeScreen && el.appShell) { el.welcomeScreen.hidden = true; el.appShell.classList.add("ready"); el.themeToggleHeader?.appendChild(el.themeToggleBtn); el.themeToggleBtn?.classList.add("visible"); }
    if (settings.autoCheckEngine) {
      await checkLive();
      await checkReadiness({ quiet: false });
    } else {
      await checkLive({ quiet: true });
    }
    state.initializingBackend = false;
    syncQueueControls();
    syncReconstructControls();
    syncWebcamControls();
    updateWebcamTelemetry();
    startPollingLoops();
    if (settings.autoCheckEngine) Promise.allSettled([checkDiagnostics({ quiet: true }), loadCacheMetrics()]);
  } catch (err) {
    backendOnline = false;
    setStatus("offline", "Depth engine offline", `Depth engine URL resolution failed · ${err.message}`);
    toast(`Depth engine initialization failed · ${err.message}`, "error", 6000);
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
