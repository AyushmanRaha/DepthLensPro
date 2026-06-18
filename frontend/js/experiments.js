"use strict";

// EXPERIMENT WORKSPACE
// ══════════════════════════════════════════════════════════════

function scalarMetric(result, key) {
  const metrics = result?.metrics || {};
  return metrics[key] ?? metrics.gt_metrics?.[key] ?? metrics.proxy_metrics?.[key] ?? metrics.prediction_stats?.[key] ?? null;
}

function formatExperimentMetric(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "Requires GT";

  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);

  if (Math.abs(n) >= 100) return n.toFixed(1);
  if (Math.abs(n) >= 10) return n.toFixed(2);
  return n.toFixed(digits);
}

function formatExperimentLatency(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n)) return "—";
  return `${Math.round(n)} ms`;
}

function formatExperimentModel(model) {
  const raw = String(model || "—").trim();

  if (/^midas[_-]?small$/i.test(raw)) return "MiDaS Small";
  if (/^dpt[_-]?hybrid$/i.test(raw)) return "DPT Hybrid";
  if (/^dpt[_-]?large$/i.test(raw)) return "DPT Large";

  return raw.replace(/_/g, " ");
}

function formatExperimentEngine(engine) {
  const raw = String(engine || "").trim().toLowerCase();

  if (!raw) return "";
  if (raw.includes("onnx")) return "ONNX Runtime";
  if (raw.includes("torch") || raw.includes("pytorch")) return "PyTorch";

  return raw.replace(/_/g, " ");
}

function experimentRows() {
  return state.experiment.results.map((r, i) => {
    const warnings = [
      ...(r.warnings || []),
      ...(r.metrics?.warnings || []),
      ...(r.gt_metadata?.warnings || []),
    ]
      .filter(Boolean)
      .map(item => String(item).trim())
      .filter(Boolean)
      .join(" · ");

    return {
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
      warnings,
    };
  });
}

function experimentStatusForRow(row) {
  const details = [];

  if (row.fallback) {
    details.push("PyTorch fallback");
  } else {
    const engineLabel = formatExperimentEngine(row.engine);
    if (engineLabel) details.push(engineLabel);
  }

  if (row.warnings) details.push(row.warnings);

  const hasWarning = row.fallback || Boolean(row.warnings);

  return {
    label: row.gt ? "GT ready" : "Image-only",
    detail: details.join(" · "),
    tone: hasWarning ? "warning" : row.gt ? "gt" : "image",
  };
}

function renderExperimentEmptyRow() {
  const label = state.experiment.running ? "Experiment is running…" : "No experiment results yet.";

  return `
    <tr class="experiment-empty-row">
      <td class="experiment-empty" colspan="8">
        <span>${esc(label)}</span>
      </td>
    </tr>
  `;
}

function renderExperimentRow(row) {
  const status = experimentStatusForRow(row);
  const modelText = formatExperimentModel(row.model);
  const deviceText = String(row.device || "—").replace(/_/g, " ");
  const latencyText = formatExperimentLatency(row.latency_ms);
  const absRelText = formatExperimentMetric(row.abs_rel);
  const rmseText = formatExperimentMetric(row.rmse);
  const deltaText = formatExperimentMetric(row.delta_1);

  return `
    <tr class="experiment-result-row">
      <td class="experiment-col-image" data-label="Image">
        <span class="experiment-file-name" title="${esc(row.filename || "Untitled image")}">${esc(row.filename || "Untitled image")}</span>
      </td>

      <td class="experiment-col-model" data-label="Model">
        <span class="experiment-pill" title="${esc(modelText)}">${esc(modelText)}</span>
      </td>

      <td class="experiment-col-device" data-label="Device">
        <span class="experiment-mono" title="${esc(deviceText)}">${esc(deviceText)}</span>
      </td>

      <td class="experiment-col-latency" data-label="Latency">
        <span class="experiment-mono">${esc(latencyText)}</span>
      </td>

      <td class="experiment-col-absrel" data-label="Abs Rel">
        <span class="${row.abs_rel === null || row.abs_rel === undefined ? "experiment-metric-empty" : "experiment-mono"}">${esc(absRelText)}</span>
      </td>

      <td class="experiment-col-rmse" data-label="RMSE">
        <span class="${row.rmse === null || row.rmse === undefined ? "experiment-metric-empty" : "experiment-mono"}">${esc(rmseText)}</span>
      </td>

      <td class="experiment-col-delta" data-label="δ<1.25">
        <span class="${row.delta_1 === null || row.delta_1 === undefined ? "experiment-metric-empty" : "experiment-mono"}">${esc(deltaText)}</span>
      </td>

      <td class="experiment-col-status" data-label="Status">
        <div class="experiment-status-cell">
          <span class="experiment-status-chip ${esc(status.tone)}">${esc(status.label)}</span>
          ${status.detail ? `<span class="experiment-status-detail" title="${esc(status.detail)}">${esc(status.detail)}</span>` : ""}
        </div>
      </td>
    </tr>
  `;
}

function renderExperiment() {
  const rows = settings.stripFilenamesFromExports ? experimentRows().map(r => ({ ...r, filename: "" })) : experimentRows();

  el.experimentCount.textContent = rows.length;

  const latencies = rows.map(r => Number(r.latency_ms)).filter(Number.isFinite);
  el.experimentAvgLatency.textContent = latencies.length
    ? `${(latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(0)} ms`
    : "—";

  const abs = rows.map(r => Number(r.abs_rel)).filter(Number.isFinite);
  el.experimentBestAbsRel.textContent = abs.length ? Math.min(...abs).toFixed(4) : "—";

  el.experimentStatusMetric.textContent = state.experiment.running
    ? "Running"
    : rows.length
      ? "Ready"
      : "Idle";

  el.experimentExportJsonBtn.disabled = state.experiment.running || !rows.length;
  el.experimentExportCsvBtn.disabled = state.experiment.running || !rows.length;

  el.experimentTableBody.innerHTML = rows.length
    ? rows.map(renderExperimentRow).join("")
    : renderExperimentEmptyRow();

  el.experimentPreviews.innerHTML = state.experiment.results.map(r => `
    <article class="experiment-card">
      <div class="experiment-card-head">
        <span title="${esc(r.filename || "Untitled image")}">${esc(r.filename || "Untitled image")}</span>
        <span>${esc(formatExperimentLatency(r.latency_ms))}</span>
      </div>

      <div class="experiment-preview-grid">
        <div class="experiment-preview-tile">
          <img src="${String(r.originalSrc || "").startsWith("data:image/") ? esc(r.originalSrc) : ""}" alt="RGB input">
          <span>RGB</span>
        </div>

        <div class="experiment-preview-tile">
          <img src="${safeDataImagePng(r.depth_map)}" alt="Predicted depth">
          <span>Predicted</span>
        </div>

        ${
          r.gt_depth_map
            ? `<div class="experiment-preview-tile"><img src="${safeDataImagePng(r.gt_depth_map)}" alt="GT depth"><span>GT</span></div>`
            : `<div class="experiment-preview-tile experiment-preview-empty"><span>GT unavailable</span></div>`
        }

        ${
          r.error_heatmap
            ? `<div class="experiment-preview-tile"><img src="${safeDataImagePng(r.error_heatmap)}" alt="Error heatmap"><span>Error</span></div>`
            : `<div class="experiment-preview-tile experiment-preview-empty"><span>Error map unavailable</span></div>`
        }
      </div>
    </article>
  `).join("");
}

async function runExperiment() {
  if (!engineReady()) {
    const ok = await checkHealth();
    if (!ok) {
      toast("Inference runtime is not ready", "error");
      return;
    }
  }

  if (!state.files.length) {
    toast("Add images in the Workspace queue before running an experiment", "warning");
    return;
  }

  if (state.gtMode && (state.files.length !== 1 || !state.gtFile)) {
    toast("GT experiments require one image and one GT depth file", "warning");
    return;
  }

  const model = selModel();
  const colormap = selCmap();
  const device = selDevice();

  state.experiment = {
    name: el.experimentName?.value || "DepthLens validation run",
    results: [],
    startedAt: new Date().toISOString(),
    running: true,
  };

  el.experimentStatus.textContent = "Running experiment — waiting for inference results";
  el.experimentRunBtn.disabled = true;
  renderExperiment();

  try {
    for (let i = 0; i < state.files.length; i++) {
      const entry = state.files[i];

      el.experimentStatus.textContent = `Running experiment — ${i + 1} of ${state.files.length}`;

      const result = await inferOne(
        entry.file,
        model,
        colormap,
        device,
        null,
        state.gtMode ? "full" : "fast",
        "color,gray",
        state.gtMode ? state.gtFile : null,
        state.gtMode,
        getInteractiveMaxDim()
      );

      state.experiment.results.push({
        ...result,
        originalSrc: entry.thumb,
        filename: entry.file.name,
        experiment_name: state.experiment.name,
      });

      renderExperiment();
    }

    el.experimentStatus.textContent = `Experiment complete — ${state.experiment.results.length} result(s) for ${state.experiment.name}`;
    toast("Experiment complete", "success");
  } catch (err) {
    el.experimentStatus.textContent = err.message;
    toast(`Experiment failed · ${err.message}`, "error", 6000);
  } finally {
    state.experiment.running = false;
    el.experimentRunBtn.disabled = false;
    renderExperiment();
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

  while (activeBlobUrls.size > 20) {
    revokeBlobUrl(activeBlobUrls.values().next().value);
  }

  const a = Object.assign(document.createElement("a"), {
    href: url,
    download: name,
  });

  document.body.appendChild(a);

  requestAnimationFrame(() => {
    a.click();
    a.remove();
    setTimeout(() => revokeBlobUrl(url), 60_000);
  });
}
