"use strict";

// GALLERY
// ══════════════════════════════════════════════════════════════
function appendGalleryItem(r) {
  const m=r.metrics||{};
  const item=Object.assign(document.createElement("div"),{className:"gallery-item"});
  item.setAttribute("role","listitem"); item.setAttribute("tabindex","0");
  item.innerHTML = `
    <div class="gallery-img-wrap">
      <img src="${safeDataImagePng(r.depth_map)}" alt="Depth map — ${esc(r.filename)}" loading="lazy"/>
      <div class="gallery-overlay">◍</div>
    </div>
    <div class="gallery-meta">
      <div class="gallery-filename" title="${esc(r.filename)}">${esc(r.filename)}</div>
      <div class="gallery-tags">
        <span class="gallery-tag">${esc(r.model?.replace("MiDaS_","").replace("DPT_","DPT "))}</span>
        <span class="gallery-tag">${esc(r.colormap)}</span>
        <span class="gallery-tag">${esc(r.device_used||"")}</span>
        ${r.cached && settings.showCacheBadges?'<span class="gallery-tag">cached</span>':""}
      </div>
      <div class="gallery-stats-row">
        <span>Latency <strong>${esc(Number.isFinite(Number(r.latency_ms)) ? `${Number(r.latency_ms)}ms` : "—")}</strong></span>
        <span>Proxy <strong>${Number.isFinite(Number(m.ssim)) ? Number(m.ssim).toFixed(3) : "—"}</strong></span>
        <span>${esc(r.resolution?.width ?? "?")}×${esc(r.resolution?.height ?? "?")}</span>
      </div>
    </div>`;
  item.addEventListener("click",()=>openLightbox(r));
  item.addEventListener("keydown",e=>{ if (e.key==="Enter"||e.key===" ") openLightbox(r); });
  el.gallery.appendChild(item);
}

el.clearResultsBtn?.addEventListener("click",()=>{ state.results=[]; el.gallery.innerHTML=""; el.resultsCard.hidden=true; });
el.downloadAllBtn?.addEventListener("click",()=>{
  if (!state.results.length) return;
  state.results.forEach(r=>dlB64(`depth_${r.filename}`,r.depth_map));
  toast(`Downloading ${state.results.length} depth maps`);
});

// ══════════════════════════════════════════════════════════════
// LIGHTBOX METRICS ACCORDION
// ══════════════════════════════════════════════════════════════
const METRIC_GROUPS = [
  { id:"error", icon:"", label:"Core Error Metrics",
    note:"Computed from predicted depth distribution only (no ground truth required). Values reflect self-consistency of the depth map.",
    metrics:[
      {key:"mae",label:"Mean Absolute Deviation from Predicted Mean",unit:"",desc:"Prediction-only mean absolute deviation from the predicted depth mean. Proxy statistic, not GT MAE.",needsGT:false},
      {key:"rmse",label:"RMS Deviation from Predicted Mean",unit:"",desc:"Prediction-only RMS deviation from the predicted depth mean. Proxy statistic, not GT RMSE.",needsGT:false},
      {key:"log_rmse",label:"Log-Depth Deviation",unit:"",desc:"Prediction-only log-depth deviation around the predicted mean; not benchmark Log RMSE.",needsGT:false},
      {key:"abs_rel",label:"Absolute Relative Error (Abs Rel)",unit:"",desc:"Mean of |pred−GT|/GT. Standard MDE benchmark metric. Requires ground-truth depth.",needsGT:true},
      {key:"sq_rel",label:"Squared Relative Error (Sq Rel)",unit:"",desc:"Mean of (pred−GT)²/GT. Penalises large relative errors. Requires ground-truth depth.",needsGT:true},
      {key:"gt_mae",label:"True MAE vs GT",unit:"",desc:"Mean absolute error after median-scale alignment to valid GT pixels.",needsGT:true},
      {key:"gt_rmse",label:"True RMSE vs GT",unit:"",desc:"Root mean squared error after median-scale alignment to valid GT pixels.",needsGT:true},
      {key:"gt_log_rmse",label:"True Log RMSE vs GT",unit:"",desc:"Log-depth RMSE after median-scale alignment to valid GT pixels.",needsGT:true},
    ]},
  { id:"accuracy", icon:"", label:"Threshold Accuracy",
    note:"δ metrics require ground-truth depth maps and cannot be computed here.",
    metrics:[
      {key:"delta_1",label:"δ < 1.25¹",unit:"%",desc:"Fraction of pixels within 25% scale of ground truth. Requires GT.",needsGT:true},
      {key:"delta_2",label:"δ < 1.25²",unit:"%",desc:"Looser threshold (56%). Requires GT.",needsGT:true},
      {key:"delta_3",label:"δ < 1.25³",unit:"%",desc:"Loosest threshold (95%). Requires GT.",needsGT:true},
    ]},
  { id:"scaleinv", icon:"", label:"Scale-Invariant Metrics",
    metrics:[
      {key:"silog",label:"Log-Depth Dispersion Proxy",unit:"",desc:"Prediction-only log-depth dispersion proxy. True SILog requires GT.",needsGT:false},
      {key:"dynamic_range",label:"Dynamic Range",unit:" bits",desc:"Log₂ ratio of max/min non-zero depth. Larger = more depth variation captured.",needsGT:false},
      {key:"entropy",label:"Shannon Entropy",unit:" bits",desc:"Entropy of the depth histogram. Higher = more uniformly distributed depth values.",needsGT:false},
      {key:"coverage",label:"Depth Coverage",unit:"%",desc:"Fraction of histogram bins with ≥1% of peak count. Higher = depth values spread across the full range.",needsGT:false,pct:true},
    ]},
  { id:"structural", icon:"", label:"Structural & Geometric Metrics",
    metrics:[
      {key:"ssim",label:"RGB–Depth Structural Proxy",unit:"",desc:"Proxy comparing predicted depth structure to grayscale RGB input; not reference SSIM.",needsGT:false},
      {key:"gradient_mean",label:"Gradient Mean",unit:"",desc:"Mean Sobel gradient magnitude over the depth map. Higher = more depth edges/transitions.",needsGT:false},
      {key:"gradient_std",label:"Gradient Std Dev",unit:"",desc:"Variation in gradient strength. High std means some regions have sharp edges while others are smooth.",needsGT:false},
      {key:"gradient_error",label:"Depth Edge Proxy",unit:"",desc:"Prediction-only edge-detail proxy equal to mean depth gradient; not GT gradient error.",needsGT:false},
      {key:"edge_density",label:"Edge Density",unit:"%",desc:"Fraction of pixels with gradient > mean+std. Indicates how richly detailed the depth edges are.",needsGT:false,pct:true},
      {key:"surface_normal_error",label:"Surface Normal Error",unit:"",desc:"Requires ground-truth normals derived from GT depth. Not computable without GT.",needsGT:true},
    ]},
  { id:"perceptual", icon:"", label:"Perceptual & Consistency Metrics",
    metrics:[
      {key:"psnr",label:"Depth Variance PSNR Proxy",unit:" dB",desc:"Prediction-only PSNR-like variance proxy; true PSNR requires a reference depth map.",needsGT:false},
      {key:"lpips",label:"LPIPS (Perceptual Similarity)",unit:"",desc:"Learned perceptual metric. Requires a reference depth map. Not computable without GT.",needsGT:true},
    ]},
  { id:"ranking", icon:"", label:"Ranking / Relative Depth Metrics",
    metrics:[
      {key:"ordinal_error",label:"Ordinal Error",unit:"",desc:"Fraction of pixel pairs where relative ordering of pred depth disagrees with GT. Requires GT.",needsGT:true},
    ]},
];

function valColor(key,val) {
  if (val===null||val===undefined) return "na";
  if (key==="ssim")  return val>0.7?"good":val>0.4?"warn":"bad";
  if (key==="silog") return val<10?"good":val<25?"warn":"bad";
  if (key==="psnr")  return val>30?"good":val>15?"warn":"bad";
  return "";
}

function metricUnavailableReason(metrics, key, mode, needsGT) {
  const unavailable = metrics?.unavailable || {};
  if (unavailable[key] === "not_requested_fast_mode") return "Not requested in fast mode";
  if (unavailable[key] === "needs_gt_depth_upload" || needsGT) return "Needs GT depth upload";
  if (unavailable[key] === "not_implemented") return "Not implemented yet";
  if (unavailable[key]) return String(unavailable[key]).replace(/_/g," ");
  if (mode === "fast") return "Not requested in fast mode";
  return "—";
}

function renderMetricsAccordion(resultOrMetrics) {
  const metrics = resultOrMetrics?.metrics || resultOrMetrics || {};
  const mode = resultOrMetrics?.metrics_mode || "full";
  const hasGt = Boolean(resultOrMetrics?.gt_metadata?.provided);
  el.lightboxMetrics.innerHTML = "";
  METRIC_GROUPS.forEach((group,gi) => {
    const div=document.createElement("div");
    div.className="metric-group"; div.id=`mg-${group.id}`;
    const hdr=document.createElement("div");
    hdr.className="metric-group-header"; hdr.setAttribute("role","button");
    hdr.setAttribute("tabindex","0"); hdr.setAttribute("aria-expanded","false");
    hdr.innerHTML=`<span><span class="mg-icon">${esc(group.icon)}</span>${esc(group.label)}</span><span class="mg-toggle" aria-hidden="true"></span>`;
    hdr.addEventListener("click",()=>toggleAccordion(div));
    hdr.addEventListener("keydown",e=>{ if (e.key==="Enter"||e.key===" "){e.preventDefault();toggleAccordion(div);} });
    const body=document.createElement("div");
    body.className="metric-group-body"; body.setAttribute("role","region");
    const content=document.createElement("div"); content.className="metric-group-content";
    if (group.note) {
      const noteEl=document.createElement("p");
      noteEl.style.cssText="font-family:var(--ff-mono);font-size:.6rem;color:var(--text-dim);margin-bottom:.4rem;line-height:1.4;";
      noteEl.textContent=group.note; content.appendChild(noteEl);
    }
    const seen=new Set();
    group.metrics.forEach(m => {
      if (seen.has(m.key)) return; seen.add(m.key);
      const raw=metrics?.[m.key];
      const isNull=raw===null||raw===undefined;
      let valText,cls;
      if (m.needsGT && isNull) { valText=hasGt ? metricUnavailableReason(metrics,m.key,mode,true) : "Needs GT depth upload"; cls="na"; }
      else if (isNull)         { valText=metricUnavailableReason(metrics,m.key,mode,false); cls="na"; }
      else if (m.pct)          { valText=`${(raw*100).toFixed(1)}%`; cls=valColor(m.key,raw); }
      else                     { valText=`${raw}${m.unit||""}`; cls=valColor(m.key,raw); }
      const badge = m.needsGT ? '<span class="metric-badge gt">GT</span>' : (metrics?.proxy_metrics && Object.prototype.hasOwnProperty.call(metrics.proxy_metrics,m.key) ? '<span class="metric-badge">Proxy</span>' : '');
      const row=document.createElement("div"); row.className="metric-row";
      row.innerHTML=`
        <div class="metric-row-left">
          <span class="metric-row-name">${esc(m.label)}${badge}</span>
          <span class="metric-row-desc">${esc(m.desc)}</span>
        </div>
        <span class="metric-row-val ${esc(cls)}">${esc(valText)}</span>`;
      content.appendChild(row);
    });
    body.appendChild(content); div.appendChild(hdr); div.appendChild(body);
    el.lightboxMetrics.appendChild(div);
    if (gi===0) toggleAccordion(div);
  });
}

function toggleAccordion(div) {
  const isOpen=div.classList.contains("open");
  div.classList.toggle("open",!isOpen);
  div.querySelector(".metric-group-header").setAttribute("aria-expanded",String(!isOpen));
}

function updateBlendPreview() {
  const v=Number(el.lbSlider.value||50)/100;
  el.lbOrigImg.style.opacity=(1-v*0.85).toFixed(2);
  el.lbDepthImg.style.opacity=(0.2+v*0.8).toFixed(2);
  if (el.lbRangeValue) el.lbRangeValue.textContent=`${Math.round(v*100)}%`;
}

let lightboxCloseTimer=null, lightboxTransitionCleanup=null, bodyScrollY=0;
function lockBodyScroll() {
  bodyScrollY=window.scrollY||window.pageYOffset||0;
  document.body.classList.add("modal-open");
  document.body.style.top=`-${bodyScrollY}px`;
}
function unlockBodyScroll() {
  document.body.classList.remove("modal-open");
  document.body.style.top="";
  window.scrollTo(0,bodyScrollY);
}

function openLightbox(r) {
  if (lightboxTransitionCleanup) { lightboxTransitionCleanup(); lightboxTransitionCleanup=null; }
  state.lb.current=r;
  el.lbOrigImg.src=r.originalSrc||"";
  el.lbDepthImg.src=safeDataImagePng(r.depth_map);
  el.lbSlider.value=50; updateBlendPreview();
  el.lbTags.innerHTML=[
    r.model?.replace("MiDaS_","").replace("DPT_","DPT "),
    r.colormap, r.device_used, `${r.latency_ms} ms`,
    r.cached?"cached":null, `${r.resolution?.width}×${r.resolution?.height}`,
  ].filter(Boolean).map(t=>`<span class="lb-tag">${esc(t)}</span>`).join("");
  renderMetricsAccordion(r);
  if (lightboxCloseTimer) { clearTimeout(lightboxCloseTimer); lightboxCloseTimer=null; }
  el.lightboxBackdrop.hidden=false;
  el.lightboxBackdrop.classList.remove("is-closing","is-open");
  el.lightboxBackdrop.classList.add("is-mounted");
  requestAnimationFrame(()=>{ requestAnimationFrame(()=>{ el.lightboxBackdrop.classList.add("is-open"); }); });
  el.lightboxMetrics.scrollTop=0; lockBodyScroll(); el.lightboxClose.focus();
}

function closeLightbox() {
  if (el.lightboxBackdrop.hidden) return;
  if (el.lightboxBackdrop.classList.contains("is-closing")) return;
  if (lightboxTransitionCleanup) { lightboxTransitionCleanup(); lightboxTransitionCleanup=null; }
  el.lightboxBackdrop.classList.remove("is-open");
  el.lightboxBackdrop.classList.add("is-closing");
  let finalized = false;
  const finalize=()=>{
    if (finalized) return;
    finalized = true;
    if (lightboxCloseTimer) { clearTimeout(lightboxCloseTimer); lightboxCloseTimer=null; }
    if (lightboxTransitionCleanup) { lightboxTransitionCleanup(); lightboxTransitionCleanup=null; }
    el.lightboxBackdrop.hidden=true;
    el.lightboxBackdrop.classList.remove("is-mounted","is-closing");
    unlockBodyScroll(); state.lb.current=null;
  };
  const onDone=(ev)=>{
    if (ev.target!==el.lightboxBackdrop) return;
    finalize();
  };
  lightboxTransitionCleanup = () => el.lightboxBackdrop.removeEventListener("transitionend",onDone);
  el.lightboxBackdrop?.addEventListener("transitionend",onDone, { once: true });
  lightboxCloseTimer=setTimeout(finalize,420);
}

el.lightboxClose?.addEventListener("click",closeLightbox);
el.lightboxBackdrop?.addEventListener("click",e=>{ if (e.target===el.lightboxBackdrop) closeLightbox(); });
document.addEventListener("keydown",e=>{ if (e.key==="Escape") closeLightbox(); });
el.lbSlider?.addEventListener("input",()=>{ updateBlendPreview(); });
el.lbDlDepth?.addEventListener("click",()=>{ const r=state.lb.current; if (r) dlB64(`depth_${r.filename}`,r.depth_map); });
el.lbDlGray?.addEventListener("click",()=>{ const r=state.lb.current; if (r) dlB64(`gray_${r.filename}`,r.grayscale); });
