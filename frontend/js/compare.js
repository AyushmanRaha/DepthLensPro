"use strict";

// ══════════════════════════════════════════════════════════════
// COMPARE PANEL
// ══════════════════════════════════════════════════════════════
el.compareFileInput?.addEventListener("change",()=>{
  state.compareFile=el.compareFileInput.files[0];
  if (state.compareFile) { el.compareFileName.textContent=state.compareFile.name; el.compareRunBtn.disabled=false; toast(`Loaded ${state.compareFile.name}`); }
});
el.compareDropZone?.addEventListener("dragover",e=>{ e.preventDefault(); el.compareDropZone.classList.add("drag-over"); });
el.compareDropZone?.addEventListener("dragleave",e=>{ if (!el.compareDropZone.contains(e.relatedTarget)) el.compareDropZone.classList.remove("drag-over"); });
el.compareDropZone?.addEventListener("drop",e=>{
  e.preventDefault(); el.compareDropZone.classList.remove("drag-over");
  const f=e.dataTransfer.files[0];
  if (f?.type.startsWith("image/")) { state.compareFile=f; el.compareFileName.textContent=f.name; el.compareRunBtn.disabled=false; toast(`Loaded ${f.name}`); }
});
el.compareDropZone?.addEventListener("click",()=>el.compareFileInput.click());
el.compareRunBtn?.addEventListener("click",runComparison);
el.compareCancelBtn?.addEventListener("click",()=>{ state.compareAbort?.abort(); toast("Comparison cancelled","warning"); });

async function runComparison() {
  if (!state.compareFile) return;
  const cmap=el.compareCmap.value, device=el.compareDevice.value;
  const engine = el.compareEngine?.value || selEngine?.() || "auto";
  const includeLarge = Boolean(el.compareIncludeLarge?.checked);
  const models=["midas_small","dpt_hybrid",...(includeLarge?["dpt_large"]:[])];
  state.compareAbort=new AbortController();
  el.compareRunBtn.disabled=true; el.compareCancelBtn.hidden=false;
  state.compareView.results=[];
  el.compareProgressBlock.hidden=false; el.compareResults.innerHTML="";
  el.compareChartCard.hidden=true; el.compareMetricGrid.innerHTML="";
  if (!engineReady()) {
    const ok = await checkHealth();
    if (!ok) {
      toast(`Depth engine is unavailable at ${API || DEFAULT_API_BASE_URL}`, "error", 6000);
      el.compareRunBtn.disabled=false; el.compareCancelBtn.hidden=true; state.compareAbort=null;
      return;
    }
  }
  const t0=Date.now();
  const tick=setInterval(()=>{
    const elapsed=Date.now()-t0;
    const pct=Math.min(98, (elapsed / Math.max(1, models.reduce((acc,m)=>acc+getEstimate("compare",m,device),0))) * 100);
    el.compareProgressFill.style.width=`${pct}%`;
    el.compareProgressPct.textContent=`${Math.round(pct)}%`;
    el.compareProgressText.textContent=`Running comparison · ${prettyEngineName(engine)}`;
    el.compareProgressEta.textContent=`Elapsed ${fmtDuration(elapsed)}`;
  },120);
  try {
    const fd=new FormData();
    fd.append("file", state.compareFile);
    fd.append("models", models.join(","));
    fd.append("colormap", cmap);
    fd.append("device", device);
    fd.append("metrics", "full");
    fd.append("outputs", "color,gray");
    fd.append("engine", engine);
    const res=await apiFetch("/api/compare", { method:"POST", body:fd, signal: requestSignal(state.compareAbort.signal, includeLarge ? 300000 : 180000) });
    const payload=await res.json();
    const results=payload.results || [];
    const errors=payload.errors || [];
    clearInterval(tick);
    el.compareProgressFill.style.width="100%"; el.compareProgressPct.textContent="100%";
    el.compareProgressText.textContent=errors.length ? "Comparison completed with failures" : "Comparison complete";
    el.compareProgressEta.textContent=`Total ${fmtDuration(Date.now()-t0)}`;
    results.forEach(r => { updateEstimate("compare", r.model_id || r.model, device, r.latency_ms || getEstimate("compare", r.model_id || r.model, device)); renderCompareCard(r); });
    errors.forEach(renderCompareErrorCard);
    state.compareView.results=results;
    if (results.length) { renderCompareSummary(results); renderCompareChart(results,state.compareView.metricKey); }
    if (errors.length) toast(`Comparison completed with failures · ${errors.length}/${payload.total || models.length} failed`,"warning",7000);
    else toast("Comparison complete","success");
  } catch (err) {
    clearInterval(tick);
    const aborted = err.name==="AbortError" || state.compareAbort?.signal.aborted;
    el.compareProgressText.textContent=aborted ? "Comparison cancelled" : "Comparison failed";
    el.compareProgressEta.textContent=aborted ? "Cancelled by user" : (err.message || "Depth inference failed");
    if (!aborted) toast(`Comparison failed · ${err.message}`, "error", 6000);
  } finally {
    setTimeout(()=>{ el.compareProgressBlock.hidden=true; },2500);
    el.compareRunBtn.disabled=false; el.compareCancelBtn.hidden=true; state.compareAbort=null;
  }
}

function renderCompareCard(r) {
  $(".compare-placeholder")?.remove();
  const card=document.createElement("div"); card.className="compare-card";
  const lbl=esc(r.model_display_name || r.model?.replace("midas_","MiDaS ").replace("dpt_","DPT ").replace("MiDaS_","").replace("DPT_","DPT ") || "Model");
  const latency = Number.isFinite(Number(r.latency_ms)) ? `${Number(r.latency_ms)} ms` : "—";
  const engineText = formatEngineStatus(r).replace(/^.*? · /, "");
  const warning = `<div class="compare-warning">${esc(engineText)} · ${esc(r.device_used || "")}</div>`;
  card.innerHTML=`
    <div class="compare-card-header">${lbl} <span class="latency-badge">${esc(latency)}</span></div>
    ${warning}
    <img src="${safeDataImagePng(r.depth_map)}" alt="Depth map — ${lbl}" loading="lazy"/>`;
  el.compareResults.appendChild(card);
}


function renderCompareErrorCard(error) {
  $(".compare-placeholder")?.remove();
  const card=document.createElement("div"); card.className="compare-card compare-card-error";
  const name=error.model_display_name || error.model || error.model_id || "Model";
  const reason=error.message || error.technical_detail || error.error || "Model failed";
  card.innerHTML=`<div class="compare-card-header">${esc(name)} <span class="latency-badge">failed</span></div><div class="compare-warning">${esc(reason)}</div>`;
  el.compareResults.appendChild(card);
}

// ══════════════════════════════════════════════════════════════
