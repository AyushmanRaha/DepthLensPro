"use strict";

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
    if (live?.busy) return "Depth engine busy";
    if (inferenceReady) return "Depth engine ready";
    return "Engine detected · checking runtime";
  }
  if (cls === "connecting") return "Connecting to depth engine";
  return "Depth engine unavailable";
}

function engineReadinessText() {
  const readiness = state.engineStatus.readiness || readinessDetails || state.engineStatus.health?.readiness;

  if (inferenceReady) return "Inference runtime ready";
  if (readiness?.required) return readinessSummary(readiness);
  if (state.engineStatus.cls === "offline") return "Inference runtime unavailable";
  return "Checking inference runtime";
}

function engineDiagnosticsText() {
  const health = state.engineStatus.health;
  if (!health) {
    return backendOnline ? "Waiting for diagnostics" : "Diagnostics unavailable";
  }

  const status = health.diagnostics_status || health.status || "unknown";
  const accel = health.acceleration_ok === false ? "accelerator degraded" : "accelerator ok";
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
  return "Runtime not reported";
}

function engineCacheText() {
  const cache = state.cacheMetrics;
  if (!cache) return "No metrics yet";

  return `${cache.backend} · ${cache.keyspaceSize} entries · ${cache.totalHits} hits`;
}

function engineLoadedModelsText() {
  const loaded = state.engineStatus.health?.loaded_models;
  if (!Array.isArray(loaded) || !loaded.length) return "No models loaded";
  return loaded.join(", ");
}

function engineSystemText() {
  const health = state.engineStatus.health;
  const system = health?.system;
  const telemetry = health?.telemetry;

  if (!system && !telemetry) return "Waiting for diagnostics";

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

  return parts.length ? parts.join(" · ") : "No system summary reported";
}

function engineModulePillsHtml() {
  const readiness = state.engineStatus.readiness || readinessDetails || state.engineStatus.health?.readiness;
  const required = readiness?.required || {};

  const entries = Object.entries(required);
  if (!entries.length) {
    return `<span class="engine-module-pill muted">Waiting for runtime check</span>`;
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
  if (!target || settings.motion === "off" || settings.reduceMotionOverride || (options.panel && settings.panelPhysics === false)) return;

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
    el.engineStatusRefresh.textContent = "Checking";

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
    if (!state.engineStatus.panelOpen || !settings.closePanelsOnOutsideClick) return;
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
  return failed.length ? failed.join(" · ") : "Inference runtime ready";
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
      setStatus("online", "Depth engine online", `Inference runtime ready · ${API || DEFAULT_API_BASE_URL}`, deviceBadge(state.devices, state.primaryDevice));
    } else {
      const summary = readinessSummary(data);
      setStatus("offline", "Depth engine degraded", summary);
      const bannerText = el.deviceInfoBanner?.querySelector("span:last-child");
      if (bannerText) bannerText.textContent = `Inference runtime degraded · ${summary}`;
      if (!quiet) toastOnce(`Inference runtime is not ready · ${summary}`, "error", 8000);
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
    setStatus("offline", "Depth engine degraded", `Runtime check failed · ${err.message}`);
    if (!quiet) toastOnce(`Runtime check failed · ${err.message}`, "error", 6000);
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
  if (!quiet) setStatus("connecting","Connecting to depth engine", API || DEFAULT_API_BASE_URL);
  const started = performance.now();
  try {
    const res = await apiFetch("/live", { signal: requestSignal(signal, 2500) });
    const data = await res.json();
    state.engineStatus.live = data;
    backendOnline = data.status === "ok";
    if (!backendOnline) throw new Error("Unexpected /live response");
    if (data.busy) {
      setStatus("online", "Depth engine busy", "Benchmark or model load running · /live ok", deviceBadge(state.devices, state.primaryDevice));
    } else {
      setStatus("online","Depth engine detected",`Engine detected · checking runtime`, deviceBadge(state.devices, state.primaryDevice));
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
    setStatus("offline","Depth engine offline",`No /live response from ${API || DEFAULT_API_BASE_URL}`);
    const bannerText = el.deviceInfoBanner?.querySelector("span:last-child");
    if (bannerText) bannerText.textContent = `Depth engine unavailable at ${API || DEFAULT_API_BASE_URL}`;
    if (!quiet) toastOnce(`Depth engine unavailable · ${err.message}`, "error", 6000);
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
    setStatus("online","Depth engine online",`PyTorch ${data.torch_version} · ${primary} · diagnostics ${data.status || "ok"} · ${accelText}`,badge);
    await loadDevices(devs, primary);
    updateDeviceInfoBanner(`${badge} · ${accelText}`);
    logEndpointTiming("/health", started, true, data.status || "ok");
    if (data.status === "degraded" && !quiet) toastOnce("Diagnostics degraded · inference remains available","warning");
    return true;
  } catch (err) {
    logEndpointTiming("/health", started, false, err.message);
    if (backendOnline) {
      setStatus("online","Depth engine online",`Diagnostics degraded · ${err.message}`, deviceBadge(state.devices, state.primaryDevice));
      if (!quiet) toastOnce(`Diagnostics unavailable · ${err.message}`, "warning", 5000);
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
      <span class="device-opt-icon" aria-hidden="true"></span>
      <div>
        <span class="device-opt-name">Auto Select</span>
        <span class="device-opt-sub">${esc(autoDeviceLabel(devs,primary))}</span>
      </div>
    </div>`;
  el.deviceSelector.appendChild(auto);

  deviceEntries.forEach(({key,info,kinds}) => {
    if (!matchesDeviceFilter(kinds, state.deviceFilter)) return;
    const isCuda=info.type==="cuda", isMps=info.type==="mps", isXpu=info.type==="xpu";
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
        <span class="device-opt-icon" aria-hidden="true"></span>
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
  if (bannerText) bannerText.textContent = `Detected ${text} · choose compute target below`;
}
