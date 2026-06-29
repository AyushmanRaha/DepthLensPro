"use strict";

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
  const rows = results.map(r => ({
    label: r.engine === "onnxruntime" ? "ONNX Runtime" : "PyTorch",
    value: Number(r?.latency_ms?.avg),
  }));
  const canvas = $("#benchmarkChart");
  if (!canvas) { warnChartOnce("benchmark:missing", "DepthLens Pro: benchmark chart canvas is unavailable."); return; }
  benchmarkChart = { canvas, rows, draw: () => drawBarChart(canvas, rows, { label: "Average latency (ms)", emptyMessage: "No benchmark latency data yet", formatValue: v => `${Number(v).toFixed(1)} ms` }) };
  rememberChart("benchmark", canvas, benchmarkChart.draw);
  benchmarkChart.draw();
}

function renderBenchmark(data) {
  const torch = benchmarkResult(data, "pytorch");
  const onnx = benchmarkResult(data, "onnxruntime");
  el.benchTorchLatency.textContent = fmtBenchLatency(torch);
  el.benchOnnxLatency.textContent = fmtBenchLatency(onnx);
  el.benchSpeedup.textContent = data?.comparison?.display_label || (data?.comparison?.speedup ? `${data.comparison.speedup}×` : "—");
  const onnxThroughput = onnx?.throughput_fps ?? data?.onnx?.throughput_fps;
  if (el.benchRecommendedEngine) el.benchRecommendedEngine.textContent = prettyEngineName(data?.comparison?.recommended_engine || "");
  if (el.benchRecommendationReason) el.benchRecommendationReason.textContent = data?.comparison?.recommendation_reason || "Awaiting benchmark run";
  el.benchThroughput.textContent = Number.isFinite(Number(onnxThroughput)) ? `${Number(onnxThroughput).toFixed(2)} fps` : "Unavailable";
  el.benchMemory.textContent = data?.memory_snapshot?.process_rss_mb ? `${data.memory_snapshot.process_rss_mb} MB` : "—";
  const provider = onnx?.provider || onnx?.diagnostics?.runtime?.selected_provider || data?.onnx_diagnostics?.runtime?.selected_provider;
  const cpuFallback = onnx?.uses_cpu_fallback || onnx?.diagnostics?.runtime?.uses_cpu_fallback || data?.onnx_diagnostics?.runtime?.uses_cpu_fallback;
  const onnxStatus = data?.onnx?.status || onnx?.state || "unavailable";
  el.benchProvider.textContent = onnx?.status === "ok" || data?.onnx?.status === "ok" ? `ONNX ready${provider ? ` · ${provider}` : ""}${cpuFallback ? " · CPU fallback" : ""}` : `ONNX ${onnxStatus}`;
  const diag = data?.onnx_diagnostics || onnx?.diagnostics || {};
  el.benchStatus.textContent = onnx?.status === "ok" || data?.onnx?.status === "ok" ? `Weights ${data.weights?.onnx_path || data?.onnx?.onnx_path}` : `${data?.onnx?.message || onnx?.reason || "PyTorch benchmark complete · ONNX unavailable"}${diag.expected_path ? ` · expected ${diag.expected_path}` : ""}${diag.recommended_export_command ? ` · run ${diag.recommended_export_command}` : ""}`;
  renderBenchmarkChart(data);
}

async function runBenchmark() {
  if (!el.benchmarkRunBtn || el.benchmarkRunBtn.disabled) return;
  state.benchmarkAbort?.abort();
  state.benchmarkAbort = new AbortController();
  const model = el.benchmarkModel?.value || "MiDaS_small";
  const device = el.benchmarkDevice?.value || "auto";
  const engine = el.benchmarkEngine?.value || "both";
  el.benchmarkRunBtn.disabled = true;
  el.benchmarkRunBtn.textContent = "Running";
  el.benchStatus.textContent = "Loading models and measuring latency";
  setStatus("online", "Depth engine busy", "Benchmark running · /live available", deviceBadge(state.devices, state.primaryDevice));
  try {
    if (!engineReady()) {
      const ok = await checkHealth();
      if (!ok) throw new Error(`Depth engine is unavailable at ${API || DEFAULT_API_BASE_URL}`);
    }
    const res = await apiFetch(`/api/benchmark?model=${encodeURIComponent(model)}&device=${encodeURIComponent(device)}&iterations=3&engine=${encodeURIComponent(engine)}`, {
      signal: anySignal([state.benchmarkAbort.signal, timeoutSignal(240_000)]),
    });
    renderBenchmark(await res.json());
    toast("Benchmark complete","success");
  } catch (err) {
    el.benchProvider.textContent = "Failed";
    el.benchStatus.textContent = err.message;
    checkLive({ quiet: true }).catch(() => {});
    if (err.name !== "AbortError") toastOnce(`Benchmark failed · ${err.message}`,"error");
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

function fmtObsMs(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${Math.round(n)} ms` : "—";
}
function obsRowsEmpty(text) { return `<tr><td class="observability-empty" colspan="7">${esc(text)}</td></tr>`; }
function switchPerformanceView(view) {
  const next = view === "observability" ? "observability" : "benchmark";
  state.performanceView = next;
  el.performanceSubnav?.forEach(btn => { const active = btn.dataset.performanceView === next; btn.classList.toggle("active", active); btn.setAttribute("aria-selected", active ? "true" : "false"); });
  if (el.performanceViewBenchmark) { el.performanceViewBenchmark.hidden = next !== "benchmark"; el.performanceViewBenchmark.classList.toggle("active", next === "benchmark"); }
  if (el.performanceViewObservability) { el.performanceViewObservability.hidden = next !== "observability"; el.performanceViewObservability.classList.toggle("active", next === "observability"); }
  const resizeCharts = window.DepthLensCharts?.scheduleChartResize
    || (typeof scheduleChartResize === "function" ? scheduleChartResize : null);
  if (resizeCharts) requestAnimationFrame(() => resizeCharts());
  if (next === "observability") loadObservability({ quiet: true });
}
async function loadObservability({ quiet = false, signal = null } = {}) {
  state.observability.abort?.abort();
  state.observability.abort = new AbortController();
  const sig = signal ? anySignal([signal, state.observability.abort.signal, timeoutSignal(10000)]) : anySignal([state.observability.abort.signal, timeoutSignal(10000)]);
  state.observability.loading = true;
  if (el.observabilityStatus) el.observabilityStatus.textContent = "Loading";
  try {
    const res = await apiFetch("/api/observability", { signal: sig });
    const snap = await res.json();
    state.observability.snapshot = snap; state.observability.lastUpdatedAt = new Date();
    renderObservability(snap);
  } catch (err) {
    if (!quiet && err.name !== "AbortError") toast(`Observability unavailable · ${err.message}`, "warning", 6000);
    if (el.observabilityStatus) el.observabilityStatus.textContent = "Offline";
  } finally { state.observability.loading = false; }
}
function renderObservability(snapshot) {
  el.observabilityTotalRequests.textContent = snapshot?.http?.total_requests ?? "0";
  el.observabilityHttpP95.textContent = fmtObsMs(snapshot?.http?.p95_latency_ms);
  el.observabilityInferenceRuns.textContent = snapshot?.inference?.total ?? "0";
  el.observabilityInferenceP95.textContent = fmtObsMs(snapshot?.inference?.p95_latency_ms);
  el.observabilityCacheHitRatio.textContent = snapshot?.cache?.hit_ratio_percent == null ? "—" : `${snapshot.cache.hit_ratio_percent}%`;
  el.observabilityCrashCount.textContent = snapshot?.crashes?.total ?? "0";
  if (el.observabilityUpdatedAt) el.observabilityUpdatedAt.textContent = snapshot?.generated_at ? `Updated ${new Date(snapshot.generated_at).toLocaleString()}` : "Not loaded";
  if (el.observabilityStatus) el.observabilityStatus.textContent = snapshot?.status || "ok";
  renderObservabilityChart(snapshot); renderTraceRows(snapshot); renderBenchmarkHistoryRows(snapshot); renderCrashRows(snapshot);
}
function renderObservabilityChart(snapshot) {
  const canvas = el.observabilityChart || $("#observabilityChart");
  if (!canvas) { warnChartOnce("observability:missing", "DepthLens Pro: observability chart canvas is unavailable."); return; }
  const recent = snapshot?.inference?.recent || [];
  const data = recent.map(e => Number(e.latency_ms)).filter(Number.isFinite).slice(-30);
  state.observability.chart = {
    canvas,
    values: data,
    draw: () => drawLineChart(canvas, data, {
      label: "Inference latency (ms)",
      emptyMessage: "No inference latency samples yet",
      tooltipLabel: "Inference latency (ms)",
      formatValue: (value) => Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 }),
    }),
  };
  rememberChart("observability", canvas, state.observability.chart.draw);
  state.observability.chart.draw();
}

function renderTraceRows(snapshot) {
  const rows = (snapshot?.traces?.recent || []).slice(-10).reverse();
  el.observabilityTraceBody.innerHTML = rows.length ? rows.map(r => `<tr><td>${esc(new Date(r.timestamp).toLocaleTimeString())}</td><td>${esc(r.component)}</td><td>${esc(r.span)}</td><td>${esc(fmtObsMs(r.duration_ms))}</td><td>${esc(r.outcome)}</td></tr>`).join("") : obsRowsEmpty("No traces yet");
}
function renderBenchmarkHistoryRows(snapshot) {
  const rows = (snapshot?.benchmarks?.history || []).slice(-10).reverse();
  el.observabilityBenchmarkBody.innerHTML = rows.length ? rows.map(r => `<tr><td>${esc(new Date(r.timestamp).toLocaleTimeString())}</td><td>${esc(r.display_name || r.model_id)}</td><td>${esc(r.device_type)}</td><td>${esc(fmtObsMs(r.pytorch_latency_ms))}</td><td>${esc(fmtObsMs(r.onnx_latency_ms))}</td><td>${esc(r.speedup ?? "—")}</td><td>${esc(r.onnx_status || r.outcome)}</td></tr>`).join("") : obsRowsEmpty("No benchmark history yet");
}
function renderCrashRows(snapshot) {
  const rows = (snapshot?.crashes?.recent || []).slice(-10).reverse();
  el.observabilityCrashBody.innerHTML = rows.length ? rows.map(r => `<tr><td>${esc(new Date(r.timestamp).toLocaleTimeString())}</td><td>${esc(r.component)}</td><td>${esc(r.error_code)}</td><td>${esc(r.message || "—")}</td></tr>`).join("") : obsRowsEmpty("No crashes recorded");
}
function exportObservabilitySnapshot() { const blob = new Blob([JSON.stringify(state.observability.snapshot || {}, null, 2)], { type:"application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "depthlens-observability.json"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }
async function copyMetricsEndpoint() { const value = `${API || DEFAULT_API_BASE_URL}/metrics`; try { await navigator.clipboard.writeText(value); toast("Metrics endpoint copied", "success"); } catch { toast(value, "info"); } }
el.performanceSubnav?.forEach(btn => btn.addEventListener("click", () => switchPerformanceView(btn.dataset.performanceView)));
el.observabilityRefreshBtn?.addEventListener("click", () => loadObservability());
el.observabilityExportBtn?.addEventListener("click", exportObservabilitySnapshot);
el.observabilityCopyMetricsBtn?.addEventListener("click", copyMetricsEndpoint);


// ══════════════════════════════════════════════════════════════
