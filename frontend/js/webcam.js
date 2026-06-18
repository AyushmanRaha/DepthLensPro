"use strict";

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
  if (el.webcamTargetFpsMetric) el.webcamTargetFpsMetric.textContent = `${getWebcamTargetFps()} target / ${(wc.adaptiveFps || getWebcamTargetFps()).toFixed(1)} active`;
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
    el.webcamStartBtn.title = supported ? (engineReady() ? "Start webcam depth inference" : "Inference runtime will be checked before starting") : "Camera capture is not supported in this browser";
  }
  if (el.webcamStopBtn) el.webcamStopBtn.disabled = !wc.running;
  if (el.webcamPauseBtn) {
    el.webcamPauseBtn.disabled = !wc.running;
    el.webcamPauseBtn.textContent = wc.paused ? "Resume Inference" : "Pause Inference";
  }
  if (el.webcamCaptureBtn) el.webcamCaptureBtn.disabled = !(wc.lastDepthBase64 || wc.lastDepthDataUrl);
}
async function startWebcam() {
  if (!webcamSupported()) { toast("Camera capture is not supported in this browser", "error"); return; }
  if (state.webcam.running || state.webcam.starting) return;
  state.webcam.starting = true;
  syncWebcamControls();
  if (!engineReady()) {
    appendWebcamLog("Checking inference runtime");
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
      running: true, paused: false, hiddenPaused: false, inFlight: false, adaptiveFps: getWebcamTargetFps(),
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
  appendWebcamLog(state.webcam.paused ? "Webcam inference paused" : "Webcam inference resumed", state.webcam.paused ? "warning" : "success");
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
  const interval = 1000 / Math.max(0.5, state.webcam.adaptiveFps || getWebcamTargetFps());
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
    const latestLatency = Number(result.latency_ms) || e2e;
    const target = getWebcamTargetFps();
    if (latestLatency > (1000 / Math.max(target, 1)) * 1.4) wc.adaptiveFps = Math.max(0.5, (wc.adaptiveFps || target) * 0.75);
    else wc.adaptiveFps = Math.min(target, (wc.adaptiveFps || target) + 0.15);
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
        setWebcamStatus("Running", "Paused after errors");
        toast("Webcam paused after repeated errors", "error", 8000);
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
  state.webcam.lastDepthBase64 = null;
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
  if (!state.webcam.lastDepthDataUrl) return;
  const a = Object.assign(document.createElement("a"), { href: state.webcam.lastDepthDataUrl, download: name });
  document.body.appendChild(a); a.click(); a.remove();
}
function handleWebcamVisibilityChange() {
  if (!settings.pauseWebcamWhenHidden) return;
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
