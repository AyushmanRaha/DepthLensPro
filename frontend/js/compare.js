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
  const models=["midas_small","dpt_hybrid","dpt_large"];
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
  const results=[], t0=Date.now();
  try {
  for (let i=0;i<models.length;i++) {
    if (state.compareAbort.signal.aborted) break;
    const model=models[i];
    const estCurrent=getEstimate("compare",model,device);
    const estRemainingStatic=models.slice(i+1).reduce((acc,m)=>acc+getEstimate("compare",m,device),0);
    const modelStart=Date.now();
    const tick=setInterval(()=>{
      const elapsedCurrent=Date.now()-modelStart;
      const unitDone=i+Math.min(elapsedCurrent/estCurrent,0.98);
      const pct=Math.min((unitDone/models.length)*100,99);
      const rem=Math.max(0,estCurrent-elapsedCurrent+estRemainingStatic);
      el.compareProgressFill.style.width=`${pct}%`;
      el.compareProgressPct.textContent=`${Math.round(pct)}%`;
      el.compareProgressText.textContent=`Running ${model}`;
      el.compareProgressEta.textContent=`ETA ${fmtDuration(rem)}`;
    },120);
    try {
      const r=await inferOne(state.compareFile,model,cmap,device,state.compareAbort.signal,"full","color,gray",null,false,getInteractiveMaxDim());
      clearInterval(tick); updateEstimate("compare",model,device,r.latency_ms);
      results.push(r); renderCompareCard(r);
    } catch(err) {
      clearInterval(tick);
      if (err.name!=="AbortError") toast(`${model} failed · ${err.message}`,"error");
    }
  }
  el.compareProgressFill.style.width="100%"; el.compareProgressPct.textContent="100%";
  el.compareProgressText.textContent=state.compareAbort.signal.aborted?"Cancelled":"Comparison complete";
  el.compareProgressEta.textContent=`Total ${fmtDuration(Date.now()-t0)}`;
  if (results.length) {
    state.compareView.results=results;
    renderCompareSummary(results);
    renderCompareChart(results,state.compareView.metricKey);
    toast("Comparison complete","success");
  }
  } catch (err) {
    el.compareProgressText.textContent="Comparison failed";
    el.compareProgressEta.textContent=err.message || "Depth inference failed";
    toast(`Comparison failed · ${err.message}`, "error", 6000);
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
  const warning = r.fallback_used ? `<div class="compare-warning">ONNX unavailable · ${esc(r.engine_used || "PyTorch")} fallback · ${esc(r.device_used || "")}</div>` : `<div class="compare-warning">${esc(r.engine_used || "")} · ${esc(r.device_used || "")}</div>`;
  card.innerHTML=`
    <div class="compare-card-header">${lbl} <span class="latency-badge">${esc(latency)}</span></div>
    ${warning}
    <img src="${safeDataImagePng(r.depth_map)}" alt="Depth map — ${lbl}" loading="lazy"/>`;
  el.compareResults.appendChild(card);
}


// ══════════════════════════════════════════════════════════════
