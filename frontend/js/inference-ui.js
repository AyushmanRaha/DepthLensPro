"use strict";

// INFERENCE
// ══════════════════════════════════════════════════════════════
function selModel()  { return $('input[name="model"]:checked')?.value    || "MiDaS_small"; }
function selCmap()   { return $('input[name="colormap"]:checked')?.value || "inferno"; }
function selDevice() { return $('input[name="device"]:checked')?.value || "auto"; }
function selEngine() { return el.engineMode?.value || "auto"; }
function prettyEngineName(engine) { const e=String(engine||"").toLowerCase(); if (e==="onnxruntime"||e==="onnx") return "ONNX Runtime"; if (e==="pytorch") return "PyTorch"; if (e==="auto") return "Auto"; if (e==="cache") return "Cached"; return e ? e.replace(/_/g," ") : "—"; }
function formatEngineStatus(result) { const req=result?.engine_requested||"auto", used=result?.engine_used; const lat=Number.isFinite(Number(result?.latency_ms)) ? `${result.latency_ms} ms` : "—"; if (result?.fallback_used) return `${lat} · ONNX unavailable → ${prettyEngineName(used)} fallback`; if (used==="cache") { const selected=result?.engine_selection?.selected_engine; return `${lat} · Cached${selected&&selected!=="cache" ? ` · ${prettyEngineName(selected)}` : ""}`; } if (req==="auto" && used) return `${lat} · Auto → ${prettyEngineName(used)}`; return `${lat} · ${prettyEngineName(used||req)}`; }

function setProgress(pct,status,eta,currentFile,countStr) {
  el.progressFill.style.width = `${pct}%`;
  el.progressPct.textContent = `${Math.round(pct)}%`;
  if (el.progressBar) el.progressBar.setAttribute("aria-valuenow",pct);
  el.progressStatusText.textContent = status;
  el.progressEta.textContent = eta||"";
  if (currentFile!==undefined) el.progressCurrentFile.textContent = currentFile;
  if (countStr!==undefined) el.progressItemCount.textContent = countStr;
}

function timingKey(model,device) { return `${model}::${device||"cpu"}`; }
function getEstimate(mode,model,device) {
  return state.timing[mode]?.[timingKey(model,device)] || state.timing[mode]?.default || 1300;
}
function updateEstimate(mode,model,device,ms) {
  const key=timingKey(model,device);
  const prev=state.timing[mode][key];
  state.timing[mode][key] = prev ? prev*0.65+ms*0.35 : ms;
  const vals=Object.values(state.timing[mode]).filter(v=>typeof v==="number");
  if (vals.length) state.timing[mode].default = vals.reduce((a,b)=>a+b,0)/vals.length;
}

function fmtDuration(ms) {
  const s=Math.round(ms/1000);
  if (s<60) return `${s}s`;
  if (s<3600) return `${Math.floor(s/60)}m ${s%60}s`;
  return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m ${s%60}s`;
}

el.runBtn?.addEventListener("click",runBatch);
el.cancelBtn?.addEventListener("click",cancelBatch);

function cancelBatch() {
  if (state.abort) { state.abort.abort(); state.abort=null; }
  pendingRunningFiles().forEach(entry => { entry.status = "pending"; setFileSt(entry.id, "pending", "Cancelled"); });
  el.runBtn.disabled = false; el.clearBtn.disabled = false; el.cancelBtn.hidden = true;
  toast("Batch cancelled","warning");
}

function pendingRunningFiles() { return state.files.filter(f => f.status === "running"); }

async function runBatch() {
  if (!engineReady()) {
    const ok = await checkHealth();
    if (!ok) { toast(`Depth engine is unavailable at ${API || DEFAULT_API_BASE_URL}`, "error", 6000); return; }
  }
  const pending = state.files.filter(f=>f.status==="pending"||f.status==="error");
  if (!pending.length) return;
  state.abort = new AbortController();
  el.runBtn.disabled=true; el.clearBtn.disabled=true;
  el.cancelBtn.hidden=false; el.progressBlock.hidden=false;
  setProgress(0,"Starting batch","","",`0 / ${pending.length}`);

  const batchStart=Date.now();
  const model=selModel(), colormap=selCmap(), device=selDevice(), engine=selEngine();

  try {
  for (let i=0;i<pending.length;i++) {
    if (state.abort?.signal?.aborted) break;
    const entry=pending[i];
    setFileSt(entry.id,"running","Running");
    const estCurrent=getEstimate("workspace",model,device);
    const estRemainingStatic=pending.slice(i+1).reduce(acc=>acc+getEstimate("workspace",model,device),0);
    const itemStart=Date.now();
    const tick=setInterval(()=>{
      const elapsedCurrent=Date.now()-itemStart;
      const unitDone=i+Math.min(elapsedCurrent/estCurrent,0.98);
      const pct=Math.min((unitDone/pending.length)*100,99);
      const rem=Math.max(0,estCurrent-elapsedCurrent+estRemainingStatic);
      setProgress(pct,`Processing ${i+1} of ${pending.length}`,`ETA ${fmtDuration(rem)}`,esc(entry.file.name),`${i+1} / ${pending.length}`);
    },120);

    try {
      if (settings.warnOnDegradedEngine && backendOnline && !inferenceReady) toastOnce("Depth engine readiness is degraded; inference may fail", "warning", 6000);
      const result=await inferOne(entry.file,model,colormap,device,state.abort.signal,state.gtMode?"full":"fast",state.gtMode?"color,gray":"color",state.gtMode?state.gtFile:null,state.gtMode,getInteractiveMaxDim(),engine);
      clearInterval(tick);
      updateEstimate("workspace",model,device,result.latency_ms);
      entry.result=result; entry.status=result.fallback_used?"completed_with_warning":"done";
      setFileSt(entry.id,result.fallback_used?"warning":"done",`${result.fallback_used?"⚠":"✓"} ${formatEngineStatus(result)}`);
      if (result.fallback_used) {
        toastOnce(settings.allowFallbackEngine ? "Depth map generated with PyTorch fallback · ONNX unavailable" : "PyTorch fallback is disabled in Settings · review result", "warning", 4500);
        if (!settings.allowFallbackEngine) entry.status = "completed_with_warning";
      }
      state.session.total++; state.session.totalInferenceMs+=result.latency_ms;
      if (result.cached) state.session.cached++;
      state.session.latencies.push(result.latency_ms);
      updateMetrics(); pushLatency(result.latency_ms);
      loadCacheMetrics();
      state.results.push({...result,originalSrc:entry.thumb,filename:entry.file.name});
      appendGalleryItem(state.results.at(-1));
      syncReconstructControls();
      el.resultsCard.hidden=false;
    } catch(err) {
      clearInterval(tick);
      if (err.name==="AbortError") { entry.status = "pending"; setFileSt(entry.id,"pending","Cancelled"); break; }
      entry.status="error"; setFileSt(entry.id,"error","Error");
      state.session.errors++; updateMetrics();
      toast(`"${entry.file.name}": ${err.message}`,"error");
    }
  }

  const elapsed=Date.now()-batchStart;
  const done=pending.filter(e=>e.status==="done"||e.status==="completed_with_warning").length;
  setProgress(100,`Batch complete — ${done} image${done!==1?"s":""} in ${fmtDuration(elapsed)}`,"");
  if (done>0) toast(`Batch complete — ${done} succeeded`,"success");
  if (done > 0 && done === pending.length && settings.autoClearQueueAfterBatch) { state.files = []; if (el.fileQueue) el.fileQueue.innerHTML = ""; }
  } catch (err) {
    pending.filter(e=>e.status==="running").forEach(e=>{ e.status="error"; setFileSt(e.id,"error","Error"); });
    state.session.errors += pending.filter(e=>e.status==="error").length;
    updateMetrics();
    setProgress(100,"Batch failed",err.message || "Inference failed");
    toast(`Batch failed: ${err.message}`, "error", 6000);
  } finally {
    setTimeout(()=>{ el.progressBlock.hidden=true; },3000);
    state.abort=null;
    el.runBtn.disabled = false; el.clearBtn.disabled = false;
    el.cancelBtn.hidden=true;
    syncQueueControls();
  }
}

async function inferOne(file,model,colormap,device,signal,metrics="fast",outputs="color",gtFile=null,gtRequired=false,maxDim=null,engine="auto") {
  const fd=new FormData();
  fd.append("file",file); fd.append("model",model);
  fd.append("colormap",colormap); fd.append("device",device);
  fd.append("metrics", metrics);
  fd.append("outputs", outputs);
  fd.append("engine", engine || "auto");
  if (gtFile) fd.append("gt_file", gtFile);
  if (gtRequired) fd.append("gt_required", "true");
  if (maxDim) fd.append("max_dim", String(maxDim));
  if (settings.useInferenceCache === false) fd.append("use_cache", "false");
  if (!engineReady()) throw new Error(`Depth engine is unavailable at ${API || DEFAULT_API_BASE_URL}`);
  const res=await apiFetch("/estimate",{
    method:"POST", body:fd,
    signal: requestSignal(signal, 180_000),
  });
  return res.json();
}

// ══════════════════════════════════════════════════════════════
