"use strict";

// ══════════════════════════════════════════════════════════════
// GLOBAL SETTINGS PANEL
// ══════════════════════════════════════════════════════════════
const SETTINGS_SECTIONS = [
  ["appearance", "Appearance"], ["motion", "Motion & Physics"], ["engine", "Engine & Diagnostics"], ["performance", "Performance & Cache"],
  ["privacy", "Privacy & Uploads"], ["notifications", "Notifications & Logs"], ["layout", "Layout & Navigation"], ["reset", "Reset"]
];
function controlRow(label, help, control) { return `<div class="settings-row"><div class="settings-copy"><strong>${esc(label)}</strong><span>${esc(help)}</span></div><div class="settings-control">${control}</div></div>`; }
function sw(key) { return `<label class="settings-switch"><input type="checkbox" data-setting="${esc(key)}" ${settings[key] ? "checked" : ""}><span aria-hidden="true"></span></label>`; }
function sel(key, opts) { return `<select data-setting="${esc(key)}">${opts.map(([v,l])=>`<option value="${esc(v)}" ${settings[key] == v ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>`; }
function settingsAdvancedReadout() {
  return JSON.stringify({ backendUrl: API || settings.backendUrl || DEFAULT_API_BASE_URL, backendOnline, inferenceReady, readiness: readinessDetails, engineStatus: state.engineStatus, cacheMetrics: state.cacheMetrics }, null, 2);
}
function renderSettingsPanel() {
  if (!el.settingsPanelBody) return;
  const theme = currentTheme || getSavedTheme();
  const html = {
    appearance: [
      controlRow("Theme mode", "Synchronizes with the header toggle.", `<select data-setting="__theme"><option value="dark" ${theme === "dark" ? "selected" : ""}>Dark</option><option value="light" ${theme === "light" ? "selected" : ""}>Light</option></select>`),
      controlRow("UI color palette", "DepthLens Cyan preserves the current default colors.", sel("palette", [["depthlens","DepthLens Cyan"],["violet","Violet Prism"],["emerald","Emerald Scan"],["amber","Amber Lab"],["rose","Rose Infrared"],["graphite","Mono Graphite"]])),
      controlRow("Accent intensity", "Adjust global glow and focus strength.", `<input type="range" min="70" max="130" value="${Math.round((settings.accentIntensity||1)*100)}" data-setting="accentIntensity"><span>${Math.round((settings.accentIntensity||1)*100)}%</span>`),
      controlRow("Glass intensity", "Changes panel opacity, blur, and glass depth.", sel("glassIntensity", [["subtle","Subtle"],["normal","Normal"],["strong","Strong"]])),
      controlRow("Background grid", "Show the animated workspace grid canvas.", sw("backgroundGrid")),
      controlRow("High contrast", "Strengthen borders, text, and focus clarity.", sw("highContrast")),
      controlRow("Compact UI", "Reduce card padding and vertical spacing globally.", sw("compactUi")),
      controlRow("Larger text", "Increase base text size modestly.", sw("largerText"))
    ].join(""),
    motion: [
      controlRow("UI animations", "Control global transition and animation intensity.", sel("motion", [["full","Full"],["reduced","Reduced"],["off","Off"]])),
      controlRow("3D panel physics", "Enable pointer tilt on settings and engine panels.", sw("panelPhysics")),
      controlRow("Animated borders", "Enable rotating/conic border effects where used.", sw("animatedBorders")),
      controlRow("Reduce motion override", "Force reduced motion even when the OS does not request it.", sw("reduceMotionOverride"))
    ].join(""),
    engine: [
      controlRow("Backend URL", "Empty means auto/default behavior.", `<input type="text" data-setting="backendUrl" value="${esc(settings.backendUrl)}" placeholder="${esc(DEFAULT_API_BASE_URL)}"><button class="settings-action" data-action="apply-backend" type="button">Apply</button>`),
      controlRow("Auto-check engine on launch", "Avoid aggressive boot diagnostics when disabled.", sw("autoCheckEngine")),
      controlRow("Diagnostics refresh interval", "Controls visible polling cadence.", sel("diagnosticsRefresh", [["slow","Slow"],["normal","Normal"],["fast","Fast"]])),
      controlRow("Refresh diagnostics when settings opens", "Run lightweight diagnostics when opening settings.", sw("refreshDiagnosticsOnOpen")),
      controlRow("Warn before degraded engine", "Show a warning before inference when readiness is degraded.", sw("warnOnDegradedEngine")),
      controlRow("Allow PyTorch fallback", "If disabled, fallback results are treated as warnings/failures in the UI.", sw("allowFallbackEngine")),
      controlRow("Auto-retry failed engine checks", "Allow background retries after failed health checks.", sw("autoRetryEngineChecks"))
    ].join(""),
    performance: [
      controlRow("Use inference cache", "Frontend policy flag for cache-aware requests and badges.", sw("useInferenceCache")),
      controlRow("Show cache badges", "Hide cached labels/messages without disabling cache.", sw("showCacheBadges")),
      controlRow("Maximum interactive image dimension", "Applies to Workspace, Compare, and Experiments estimates only.", sel("maxInteractiveDim", [["default","Default"],["768","768"],["1024","1024"],["1536","1536"],["2048","2048"]])),
      controlRow("Clear frontend session metrics", "Reset totals and chart history without clearing files or results.", `<button class="settings-action" data-action="clear-session" type="button">Clear</button>`),
      controlRow("Clear local UI preferences", "Reset only depthlens_settings_v1 after confirmation.", `<button class="settings-action" data-action="clear-settings" type="button">Clear</button>`),
      controlRow("Clear backend cache", "Attempts POST /cache/clear if available.", `<button class="settings-action" data-action="clear-backend-cache" type="button">Clear</button>`)
    ].join(""),
    privacy: ["warnLargeFiles","autoClearQueueAfterBatch","autoClearResultsOnClose","stripFilenamesFromExports","pauseWebcamWhenHidden","stopCameraOnTabSwitch"].map(k=>controlRow(({warnLargeFiles:"Warn before processing large files",autoClearQueueAfterBatch:"Auto-clear queue after successful batch",autoClearResultsOnClose:"Auto-clear results on app close",stripFilenamesFromExports:"Strip filenames from exported metadata",pauseWebcamWhenHidden:"Pause webcam when app is hidden",stopCameraOnTabSwitch:"Stop camera on tab switch"})[k], "Privacy and upload behavior policy.", sw(k))).join(""),
    notifications: [
      controlRow("Toast level", "Filter notification visibility.", sel("toastLevel", [["all","All"],["warnings","Warnings only"],["errors","Errors only"],["silent","Silent"]])),
      controlRow("Toast duration", "Default notification lifetime.", sel("toastDuration", [["short","Short"],["normal","Normal"],["long","Long"]])),
      controlRow("Deduplicate repeated warnings", "Respect toastOnce duplicate suppression.", sw("dedupeWarnings")),
      controlRow("Show endpoint timing logs", "Enable console endpoint timings.", sw("endpointTimingLogs")),
      controlRow("Verbose console logging", "Enable additional diagnostic console logs.", sw("verboseConsoleLogs")),
      controlRow("Copy diagnostics snapshot", "Copy settings, engine, cache, selections, and current tab JSON.", `<button class="settings-action" data-action="copy-diagnostics" type="button">Copy</button>`)
    ].join(""),
    layout: [
      controlRow("Remember last opened tab", "Restore the last valid workspace panel on startup.", sw("rememberLastTab")),
      controlRow("Skip welcome screen next time", "Bypass welcome and show the app shell on load.", sw("skipWelcome")),
      controlRow("Close floating panels on outside click", "Applies to settings and engine panels.", sw("closePanelsOnOutsideClick")),
      controlRow("Nav drag sensitivity", "Adjust horizontal tab drag threshold.", sel("navDragSensitivity", [["low","Low"],["normal","Normal"],["high","High"]]))
    ].join(""),
    reset: [
      controlRow("Reset appearance", "Restore palette, accent, glass, grid, contrast, density, and text size.", `<button class="settings-action" data-action="reset-appearance" type="button">Reset</button>`),
      controlRow("Reset all settings", "Restore DEFAULT_SETTINGS without deleting tab preferences.", `<button class="settings-action" data-action="reset-all" type="button">Reset</button>`),
      controlRow("Restore current DepthLens defaults", "Set dark + DepthLens Cyan unless saved theme is light.", `<button class="settings-action" data-action="restore-depthlens" type="button">Restore</button>`)
    ].join("")
  };
  el.settingsPanelBody.innerHTML = SETTINGS_SECTIONS.map(([id,title], i) => `<section class="settings-section ${i<2?'open':''}" data-section="${id}"><button class="settings-section-toggle" type="button" aria-expanded="${i<2?'true':'false'}" aria-controls="settings-section-${id}"><span>${esc(title)}</span><span class="settings-chevron" aria-hidden="true">›</span></button><div class="settings-section-body" id="settings-section-${id}"><div class="settings-section-inner">${html[id]}</div></div></section>`).join("");
  bindSettingsControls(); refreshSettingsSectionHeights();
}
function refreshSettingsSectionHeights() { $$(".settings-section", el.settingsPanelBody).forEach(sec=>{ const body=$(".settings-section-body",sec); if(body) body.style.maxHeight = sec.classList.contains("open") ? (prefersReducedMotion()?"none":`${body.scrollHeight}px`) : "0px"; }); }
function bindSettingsControls() {
  if (!el.settingsPanelBody || el.settingsPanelBody.dataset.bound === "true") return;
  el.settingsPanelBody.dataset.bound = "true";
  el.settingsPanelBody.addEventListener("click", async event => {
    const sectionBtn = event.target.closest?.(".settings-section-toggle");
    if (sectionBtn) { const sec=sectionBtn.closest(".settings-section"); const open=!sec.classList.contains("open"); sec.classList.toggle("open",open); sectionBtn.setAttribute("aria-expanded",String(open)); refreshSettingsSectionHeights(); return; }
    const action = event.target.closest?.("[data-action]")?.dataset.action;
    if (!action) return;
    if (action === "apply-backend") await applyBackendUrlSetting();
    if (action === "clear-session") clearFrontendSessionMetrics();
    if (action === "clear-settings") { if(confirm("Clear local settings and reapply defaults?")){ localStorage.removeItem(SETTINGS_KEY); settings=loadSettings(); applySettings({persist:true}); renderSettingsPanel(); } }
    if (action === "clear-backend-cache") clearBackendCache();
    if (action === "copy-diagnostics") copyDiagnosticsSnapshot();
    if (action === "reset-appearance") resetSettingsSection("appearance");
    if (action === "reset-all") resetSettingsSection("all");
    if (action === "restore-depthlens") { settings.palette="depthlens"; currentTheme=getSavedTheme()==="light"?"light":"dark"; applyTheme(currentTheme,true); saveTheme(currentTheme); applySettings({persist:true}); renderSettingsPanel(); }
  });
  el.settingsPanelBody.addEventListener("input", handleSettingsInput);
  el.settingsPanelBody.addEventListener("change", handleSettingsInput);
}
function handleSettingsInput(event) {
  const node = event.target.closest?.("[data-setting]"); if (!node) return;
  const key = node.dataset.setting;
  if (key === "__theme") { currentTheme = node.value === "light" ? "light" : "dark"; applyTheme(currentTheme,true); saveTheme(currentTheme); updateChartTheme(); updateSettingsControlState(); return; }
  let value = node.type === "checkbox" ? node.checked : node.value;
  if (key === "accentIntensity") value = Number(value) / 100;
  settings[key] = value;
  applySettings({ persist:true });
  if (["diagnosticsRefresh"].includes(key)) startPollingLoops();
  if (key === "backendUrl") return;
  updateSettingsControlState();
}
function setSettingsPanelOpen(open) {
  state.settingsPanelOpen = Boolean(open);
  if (!el.settingsPanel || !el.settingsMenuButton) return;
  el.settingsMenuButton.setAttribute("aria-expanded", String(open));
  if (open) { renderSettingsPanel(); el.settingsPanel.hidden=false; if(settings.refreshDiagnosticsOnOpen) Promise.allSettled([checkLive({quiet:true}), checkReadiness({quiet:true}), checkDiagnostics({quiet:true}), loadCacheMetrics()]).then(()=>renderSettingsPanel()); }
  else { el.settingsPanel.hidden=true; }
}
function updateSettingsControlState() { if (!el.settingsPanelBody) return; const themeSel = $('[data-setting="__theme"]', el.settingsPanelBody); if (themeSel) themeSel.value = currentTheme || getSavedTheme(); const rng=$('[data-setting="accentIntensity"]', el.settingsPanelBody); if(rng?.nextElementSibling) rng.nextElementSibling.textContent=`${Math.round((settings.accentIntensity||1)*100)}%`; }
function resetSettingsSection(section) { const keys = section === "appearance" ? ["palette","accentIntensity","glassIntensity","compactUi","highContrast","largerText","backgroundGrid"] : Object.keys(DEFAULT_SETTINGS); for (const k of keys) settings[k]=DEFAULT_SETTINGS[k]; applySettings({persist:true}); renderSettingsPanel(); }
async function applyBackendUrlSetting() { apiResolved=false; apiResolutionPromise=null; API=null; if (settings.backendUrl && !runningInElectron) try{localStorage.setItem("depthlens_api_url", settings.backendUrl)}catch{}; await resolveApiBaseUrl(); await checkLive({quiet:false}); await checkReadiness({quiet:true}); renderEngineStatusPanel(); renderSettingsPanel(); }
function clearFrontendSessionMetrics() { state.session={ total:0,cached:0,errors:0,latencies:[],totalInferenceMs:0 }; if (latencyChart) { latencyChart.values=[]; drawSparkline(latencyChart.canvas, [], { label: "Inference ms", emptyMessage: "No latency samples yet" }); } updateMetrics(); toast("Frontend session metrics cleared","success"); }
async function clearBackendCache() { try { await apiFetch("/cache/clear", { method:"POST", signal:timeoutSignal(8000) }); state.cacheMetrics=null; await loadCacheMetrics(); toast("Backend cache cleared","success"); } catch(err) { toast(`Backend cache clear unavailable · ${err.message}`,"warning",6000); } }
async function copyDiagnosticsSnapshot() { const snap={ settings, currentTheme, palette:settings.palette, api:API||settings.backendUrl||DEFAULT_API_BASE_URL, backendOnline, inferenceReady, engineStatus:state.engineStatus, selected:{model:selModel?.(), colormap:selCmap?.(), device:selDevice?.()}, cacheMetrics:state.cacheMetrics, currentTab:$(".nav-btn.active")?.dataset.panel||"main" }; try{ await navigator.clipboard.writeText(JSON.stringify(snap,null,2)); toast("Diagnostics snapshot copied","success"); }catch(err){ toast(`Diagnostics copy failed · ${err.message}`,"error"); } }
function initSettingsPanel() {
  if (!el.settingsMenuButton || el.settingsMenuButton.dataset.bound === "true") return;
  el.settingsMenuButton.dataset.bound = "true"; renderSettingsPanel();
  el.settingsMenuButton.addEventListener("click", e=>{ e.stopPropagation(); setSettingsPanelOpen(!state.settingsPanelOpen); });
  el.settingsPanelClose?.addEventListener("click",()=>setSettingsPanelOpen(false));
  document.addEventListener("click", e=>{ if(!state.settingsPanelOpen || !settings.closePanelsOnOutsideClick) return; if(el.settingsMenuHost?.contains(e.target)||el.settingsPanel?.contains(e.target)) return; setSettingsPanelOpen(false); });
  document.addEventListener("keydown", e=>{ if(e.key==="Escape" && state.settingsPanelOpen) setSettingsPanelOpen(false); });
  el.settingsMenuButton.addEventListener("mousemove", e=>{ if(settings.panelPhysics) applyPointerPhysics(el.settingsMenuButton,e,{strength:10}); });
  el.settingsMenuButton.addEventListener("mouseleave", ()=>resetPointerPhysics(el.settingsMenuButton));
  el.settingsPanel?.addEventListener("mousemove", e=>{ if(settings.panelPhysics) applyPointerPhysics(el.settingsPanel,e,{strength:2.2,panel:true}); });
  el.settingsPanel?.addEventListener("mouseleave", ()=>resetPointerPhysics(el.settingsPanel,{panel:true}));
  window.addEventListener("resize", refreshSettingsSectionHeights);
}

// ══════════════════════════════════════════════════════════════
// THEME TOGGLE LOGIC
// ══════════════════════════════════════════════════════════════
let currentTheme = getSavedTheme();

function toggleTheme() {
  currentTheme = currentTheme === "dark" ? "light" : "dark";
  applyTheme(currentTheme, true);
  saveTheme(currentTheme);
  updateChartTheme();
  updateSettingsControlState();
}

el.themeToggleBtn?.addEventListener("click", toggleTheme);

// ══════════════════════════════════════════════════════════════
// LANDING → WORKSPACE TRANSITION
// ══════════════════════════════════════════════════════════════
function enterWorkspace() {
  if (!el.welcomeScreen || !el.appShell) return;
  el.getStartedBtn?.setAttribute("disabled", "true");

  // Migrate theme toggle from landing corner → header
  migrateThemeToggle(() => {
    el.welcomeScreen.classList.add("is-exiting");
    setTimeout(() => {
      el.welcomeScreen.hidden = true;
      el.appShell.classList.add("ready");
    }, 650);
  });
}

function migrateThemeToggle(onComplete) {
  const landingWrap = el.themeToggleLanding;
  const headerWrap  = el.themeToggleHeader;
  const btn         = el.themeToggleBtn;
  if (!landingWrap || !headerWrap || !btn) { onComplete(); return; }

  const srcRect = btn.getBoundingClientRect();

  // Move button into header slot
  headerWrap.appendChild(btn);
  const dstRect = btn.getBoundingClientRect();

  const dx = srcRect.left - dstRect.left;
  const dy = srcRect.top  - dstRect.top;

  btn.style.transform  = `translate(${dx}px, ${dy}px) scale(1.1)`;
  btn.style.transition = "none";
  btn.style.opacity    = "1";

  // Hide the landing placeholder visually
  landingWrap.classList.add("is-migrating");

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      btn.style.transition = "transform 0.62s cubic-bezier(0.22,1,0.36,1), opacity 0.4s ease";
      btn.style.transform  = "translate(0,0) scale(1)";
      btn.classList.add("visible");
    });
  });

  setTimeout(() => {
    btn.style.transform  = "";
    btn.style.transition = "";
    onComplete();
  }, 480);
}

el.getStartedBtn?.addEventListener("click", enterWorkspace);

// ══════════════════════════════════════════════════════════════
// PERSISTENCE
// ══════════════════════════════════════════════════════════════
const PREFS_KEY = "depthlens_prefs_v3";

function savePrefs() {
  const prefs = {
    model:   $('input[name="model"]:checked')?.value   || "MiDaS_small",
    colormap:$('input[name="colormap"]:checked')?.value|| "inferno",
    device:  $('input[name="device"]:checked')?.value  || "auto",
  };
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch {}
}

function loadPrefs() {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (!raw) return;
    const p = JSON.parse(raw);
    const modelEl = $(`input[name="model"][value="${p.model}"]`);
    if (modelEl) modelEl.checked = true;
    const cmapEl = $(`input[name="colormap"][value="${p.colormap}"]`);
    if (cmapEl) cmapEl.checked = true;
    window._savedDevice = p.device;
  } catch {}
}

document.addEventListener("change", (e) => {
  if (["model","colormap","device"].includes(e.target.name)) {
    savePrefs();
    updateWebcamTelemetry();
  }
});
