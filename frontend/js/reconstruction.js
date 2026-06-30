"use strict";

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

function setReconstructSourceFile(file, source = "upload", previewDataUrl = "") {
  if (!file) return;
  if (!file.type?.startsWith("image/")) { toast("Image file required for point cloud generation", "warning"); return; }
  if (file.size > 20 * 1024 * 1024) { toast("Point cloud source image exceeds 20 MB", "warning"); return; }
  state.reconstruct.file = file;
  state.reconstruct.result = null;
  state.reconstruct.capture.lastCapturedFile = source === "camera" ? file : null;
  state.reconstruct.filePreview = previewDataUrl || "";
  const sourceLabel = source === "camera" ? "Camera capture" : source === "latest" ? "Latest result" : "Uploaded image";
  if (el.reconstructFileName) el.reconstructFileName.textContent = `${file.name || "Selected image"} · ${sourceLabel}`;
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
  reader.onerror = () => toast("Point cloud preview could not be read", "warning");
  reader.readAsDataURL(file);
  syncReconstructControls();
}

function setReconstructFile(file, previewDataUrl = "") {
  return setReconstructSourceFile(file, previewDataUrl ? "latest" : "upload", previewDataUrl);
}

function setCaptureStatus(message, type = "idle") {
  if (el.captureModalStatus) el.captureModalStatus.textContent = message;
  if (el.reconstructCaptureStatus) {
    el.reconstructCaptureStatus.textContent = message;
    el.reconstructCaptureStatus.dataset.state = type;
  }
}

function withCaptureModalOpen() {
  return Boolean(el.captureBackdrop && !el.captureBackdrop.hidden && state.reconstruct.capture.running);
}

function captureTimestamp(date = new Date()) {
  const pad = value => String(value).padStart(2, "0");
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}_${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}

function blobToFile(blob, name, type = "image/jpeg") {
  return new File([blob], name, { type, lastModified: Date.now() });
}

async function openReconstructCaptureModal() {
  if (!el.captureBackdrop || state.reconstruct.capture.starting || state.reconstruct.capture.running) return;
  state.reconstruct.capture.previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : el.reconstructOpenCaptureBtn;
  el.captureBackdrop.hidden = false;
  el.captureBackdrop.classList.remove("is-open");
  if (el.capturePlaceholder) { el.capturePlaceholder.hidden = false; el.capturePlaceholder.textContent = "Starting camera…"; }
  renderCaptureDetections({ status: "Warming detector…" });
  requestAnimationFrame(() => el.captureBackdrop?.classList.add("is-open"));
  el.captureCloseBtn?.focus();
  await startReconstructCaptureStream();
}

function closeReconstructCaptureModal({ quiet = false } = {}) {
  if (!el.captureBackdrop || el.captureBackdrop.hidden) return;
  stopReconstructCaptureStream();
  el.captureBackdrop.classList.remove("is-open");
  el.captureBackdrop.hidden = true;
  setCaptureStatus(quiet ? "Camera not opened" : "Camera closed", "idle");
  const previous = state.reconstruct.capture.previousFocus;
  state.reconstruct.capture.previousFocus = null;
  if (previous?.focus) previous.focus();
}

async function startReconstructCaptureStream() {
  const cap = state.reconstruct.capture;
  if (!navigator.mediaDevices?.getUserMedia) {
    setCaptureStatus("No camera API available", "error");
    if (el.capturePlaceholder) el.capturePlaceholder.textContent = "No camera API available";
    return;
  }
  cap.starting = true;
  setCaptureStatus("Starting camera…", "idle");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" }, audio: false });
    cap.stream = stream;
    cap.running = true;
    if (el.captureVideo) {
      el.captureVideo.srcObject = stream;
      await new Promise(resolve => {
        if (el.captureVideo.readyState >= 1) resolve();
        else el.captureVideo.onloadedmetadata = () => resolve();
      });
      await el.captureVideo.play?.();
    }
    if (el.capturePlaceholder) el.capturePlaceholder.hidden = true;
    setCaptureStatus("Camera ready", "ready");
    warmupCaptureDetector();
  } catch (err) {
    const name = err?.name || "";
    const message = name === "NotAllowedError" ? "Permission denied" : name === "NotFoundError" ? "No camera found" : "Camera unavailable";
    setCaptureStatus(message, "error");
    if (el.capturePlaceholder) el.capturePlaceholder.textContent = message;
    toast(`${message} · image capture remains available through upload`, "warning", 5000);
  } finally {
    cap.starting = false;
  }
}

function stopReconstructCaptureStream() {
  const cap = state.reconstruct.capture;
  clearTimeout(cap.detectTimer);
  cap.detectTimer = null;
  cap.detectAbort?.abort();
  cap.detectAbort = null;
  cap.detecting = false;
  cap.stream?.getTracks?.().forEach(track => track.stop());
  cap.stream = null;
  cap.running = false;
  if (el.captureVideo) {
    el.captureVideo.pause?.();
    el.captureVideo.srcObject = null;
    el.captureVideo.onloadedmetadata = null;
  }
}

async function captureFrameFile({ maxDim = 512, quality = 0.82 } = {}) {
  const video = el.captureVideo;
  const canvas = el.captureCanvas;
  if (!video || !canvas || !video.videoWidth || !video.videoHeight) throw new Error("Camera frame is not ready");
  const scale = Math.min(1, maxDim / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Image encoding unavailable");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", quality));
  if (!blob) throw new Error("Image encoding failed");
  return blobToFile(blob, `webcam-capture-${captureTimestamp()}.jpg`, "image/jpeg");
}

async function captureReconstructStill() {
  try {
    const file = await captureFrameFile({ maxDim: 1600, quality: 0.9 });
    state.reconstruct.capture.lastCapturedFile = file;
    await setReconstructSourceFile(file, "camera");
    closeReconstructCaptureModal({ quiet: true });
    toast("Camera image captured for 3D reconstruction", "success");
  } catch (err) {
    toast(`Camera capture failed · ${err.message || err}`, "error", 6000);
  }
}

function scheduleCaptureDetection(delay = 750) {
  const cap = state.reconstruct.capture;
  if (cap.detecting) return;
  clearTimeout(cap.detectTimer);
  if (!withCaptureModalOpen() || document.hidden) return;
  cap.detectTimer = setTimeout(runCaptureDetectionOnce, delay);
}

async function warmupCaptureDetector() {
  const cap = state.reconstruct.capture;
  if (!withCaptureModalOpen() || cap.detecting) return;
  cap.detecting = true;
  renderCaptureDetections({ status: "Warming detector…" });
  try {
    const controller = new AbortController();
    cap.detectAbort = controller;
    const res = await apiFetch(`/api/detect/status?warmup=true&device=${encodeURIComponent(selDevice?.() || "auto")}`, { signal: requestSignal(controller.signal, 60000) });
    const payload = await res.json();
    if (payload.available || payload.state === "ready") { renderCaptureDetections({ status: "Detector loaded · detecting…" }); scheduleCaptureDetection(250); }
    else { renderCaptureDetections({ error: payload.message || "Detector unavailable — capture still works" }); scheduleCaptureDetection(5000); }
  } catch (err) {
    if (err?.name !== "AbortError") renderCaptureDetections({ error: "Detector unavailable — capture still works" });
  } finally {
    cap.detecting = false;
  }
}

async function runCaptureDetectionOnce() {
  const cap = state.reconstruct.capture;
  if (!withCaptureModalOpen() || cap.detecting || document.hidden) return;
  cap.detecting = true;
  try {
    const file = await captureFrameFile({ maxDim: 512, quality: 0.76 });
    cap.detectAbort?.abort();
    const controller = new AbortController();
    cap.detectAbort = controller;
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("device", selDevice?.() || "auto");
    form.append("threshold", "0.35");
    form.append("max_detections", "5");
    const response = await apiFetch("/api/detect", { method: "POST", body: form, signal: requestSignal(controller.signal, 20000) });
    const payload = await response.json();
    cap.detectionFailures = 0;
    renderCaptureDetections(payload);
  } catch (err) {
    if (err?.name !== "AbortError") { cap.detectionFailures = (cap.detectionFailures || 0) + 1; renderCaptureDetections({ error: err?.payload?.detail || err?.message || "Detector unavailable — capture still works" }); }
  } finally {
    cap.detecting = false;
    if (withCaptureModalOpen()) scheduleCaptureDetection(cap.detectionFailures ? Math.min(8000, 1500 * cap.detectionFailures) : 900);
  }
}

function renderCaptureDetections(detections) {
  if (!el.captureDetections) return;
  if (detections && !Array.isArray(detections) && detections.status) { el.captureDetections.textContent = detections.status; return; }
  if (detections && !Array.isArray(detections) && detections.error) {
    const detail = typeof detections.error === "object" ? (detections.error.message || detections.error.error_code || "Detector unavailable") : detections.error;
    el.captureDetections.textContent = String(detail).includes("offline") ? "Backend offline" : detail;
    return;
  }
  if (detections === null) { el.captureDetections.textContent = "Detector unavailable"; return; }
  const threshold = Number(detections?.threshold || 0.35);
  const items = Array.isArray(detections) ? detections : (detections?.detections || []);
  const visible = (items || []).filter(d => Number(d.score) >= threshold).slice(0, 3);
  state.reconstruct.capture.lastDetections = visible;
  if (!visible.length) { el.captureDetections.textContent = withCaptureModalOpen() ? `No object detected · threshold ${threshold}` : "Detecting…"; return; }
  el.captureDetections.innerHTML = visible.map(d => `<div class="capture-detection-label"><span>${esc(d.label || "object")}</span><span>${Math.round(Number(d.score || 0) * 100)}%</span></div>`).join("");
}

function clearReconstruct() {
  cancelReconstruction({ quiet: true });
  state.reconstruct.file = null;
  state.reconstruct.filePreview = "";
  state.reconstruct.result = null;
  state.reconstruct.capture.lastCapturedFile = null;
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
    if (!latest) { toast("Run workspace inference first or upload an image", "warning"); return; }
    const file = dataUrlToFile(latest.originalSrc, latest.filename || "latest-depthlens-source.jpg");
    setReconstructSourceFile(file, "latest", latest.originalSrc);
    toast("Latest workspace source loaded for point cloud generation", "success");
  } catch (err) {
    toast(`Latest result unavailable · ${err.message}`, "error", 6000);
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
  updateReconstructProgress(pct, "Uploading image and options", "Estimating");
  clearInterval(state.reconstruct.progressTimer);
  state.reconstruct.progressTimer = setInterval(() => {
    const elapsed = performance.now() - started;
    const ceiling = elapsed < 15000 ? 70 : 95;
    pct = Math.min(ceiling, pct + Math.max(0.4, (ceiling - pct) * 0.045));
    updateReconstructProgress(pct, pct < 75 ? "Generating relative depth" : "Building point cloud artifact", `${Math.round(elapsed / 1000)}s elapsed`);
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
  if (!reconstructSupported()) { toast("Point cloud upload is not supported in this browser", "error"); return; }
  if (state.reconstruct.running) return;
  const ready = await ensureReconstructionBackendReady();
  if (!ready) { toast("Inference runtime is not ready for point cloud generation", "warning", 6000); syncReconstructControls(); return; }

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
      toast("Point cloud generation cancelled", "info");
    } else {
      stopReconstructProgress(0, "Failed");
      setReconstructStatus("Error", "error");
      toast(`Point cloud generation failed · ${err.message}`, "error", 7000);
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
  if (!result?.artifact_base64) { toast("No point cloud artifact available", "warning"); return; }
  try {
    const blob = b64ToBlob(result.artifact_base64, result.artifact_mime || "text/plain");
    downloadBlob(blob, result.artifact_filename || `depthlens_point_cloud.${result.artifact_format || "ply"}`);
  } catch (err) {
    toast(`Point cloud download failed · ${err.message}`, "error");
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
    toast("Point cloud metadata copied", "success");
  } catch (err) {
    toast(`Metadata copy failed · ${err.message}`, "error");
  }
}

function downloadReconstructionDepth() {
  const b64 = state.reconstruct.result?.depth_map;
  if (!b64) { toast("No depth preview available", "warning"); return; }
  dlB64("reconstruction_depth.png", b64);
}

function clearPointCloudViewer() {
  stopPointCloudViewer();
  const viewer = state.reconstruct.viewer;
  viewer.points = [];
  viewer.normalized = [];
  viewer.mode = "none";
  if (el.pointCloudPlaceholder) { el.pointCloudPlaceholder.hidden = false; el.pointCloudPlaceholder.textContent = "Generate a point cloud to preview points"; }
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
    el.pointCloudPlaceholder.textContent = viewer.normalized.length ? "" : "No preview points returned";
  }
  if (el.pointCloudModeMessage) {
    el.pointCloudModeMessage.textContent = pointCloudWebglAvailable()
      ? "2D canvas preview · drag to rotate · wheel to zoom · double-click to reset"
      : "WebGL unavailable — showing 2D preview";
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
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  const am = accent.match(/^#?([0-9a-f]{6})$/i);
  const fallback = am ? [parseInt(am[1].slice(0,2),16), parseInt(am[1].slice(2,4),16), parseInt(am[1].slice(4,6),16)] : (dark ? [0, 200, 255] : [0, 112, 204]);
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
  el.reconstructDropZone.addEventListener("drop", event => { event.preventDefault(); el.reconstructDropZone.classList.remove("drag-over"); setReconstructSourceFile(event.dataTransfer?.files?.[0], "upload"); });
  el.reconstructFileInput?.addEventListener("change", () => { setReconstructSourceFile(el.reconstructFileInput.files?.[0], "upload"); el.reconstructFileInput.value = ""; });
  el.reconstructUseLatestBtn?.addEventListener("click", useLatestReconstructionSource);
  el.reconstructOpenCaptureBtn?.addEventListener("click", openReconstructCaptureModal);
  el.captureCloseBtn?.addEventListener("click", () => closeReconstructCaptureModal());
  el.captureStillBtn?.addEventListener("click", captureReconstructStill);
  el.captureBackdrop?.addEventListener("click", event => { if (event.target === el.captureBackdrop) closeReconstructCaptureModal(); });
  document.addEventListener("keydown", event => { if (event.key === "Escape" && el.captureBackdrop && !el.captureBackdrop.hidden) closeReconstructCaptureModal(); });
  document.addEventListener("visibilitychange", () => { if (!el.captureBackdrop || el.captureBackdrop.hidden) return; document.hidden ? stopReconstructCaptureStream() : startReconstructCaptureStream(); });
  window.addEventListener("pagehide", () => closeReconstructCaptureModal({ quiet: true }));
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
