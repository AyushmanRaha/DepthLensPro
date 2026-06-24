"use strict";

// ══════════════════════════════════════════════════════════════
// POLISHED UI MOTION + GUIDE ACCORDION
// ══════════════════════════════════════════════════════════════
function prefersReducedMotion() {
  return settings.reduceMotionOverride || settings.motion === "off" || settings.motion === "reduced" || window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches || false;
}

function bindPointerGlow(selector, { tilt = 0 } = {}) {
  $$(selector).forEach(node => {
    if (node.dataset.pointerGlowBound === "true") return;
    node.dataset.pointerGlowBound = "true";
    node.style.setProperty("--mx", "50%");
    node.style.setProperty("--my", "50%");
    node.style.setProperty("--tilt-x", "0deg");
    node.style.setProperty("--tilt-y", "0deg");

    node.addEventListener("pointermove", event => {
      const rect = node.getBoundingClientRect();
      const px = (event.clientX - rect.left) / Math.max(rect.width, 1);
      const py = (event.clientY - rect.top) / Math.max(rect.height, 1);
      node.style.setProperty("--mx", `${(px * 100).toFixed(1)}%`);
      node.style.setProperty("--my", `${(py * 100).toFixed(1)}%`);
      if (!prefersReducedMotion() && tilt) {
        node.style.setProperty("--tilt-y", `${((px - 0.5) * tilt).toFixed(2)}deg`);
        node.style.setProperty("--tilt-x", `${((0.5 - py) * tilt).toFixed(2)}deg`);
      }
    });

    node.addEventListener("pointerleave", () => {
      node.style.setProperty("--mx", "50%");
      node.style.setProperty("--my", "50%");
      node.style.setProperty("--tilt-x", "0deg");
      node.style.setProperty("--tilt-y", "0deg");
    });
  });
}

function initScrollableNav() {
  const shell = el.headerNavShell;
  if (!shell || shell.dataset.scrollNavBound === "true") return;
  shell.dataset.scrollNavBound = "true";

  let pointerActive = false;
  let dragMoved = false;
  let ignoreNextClick = false;
  let activePointerId = null;
  let startX = 0;
  let startY = 0;
  let startScrollLeft = 0;
  const dragThreshold = ({ low: 14, high: 4, normal: 8 })[settings.navDragSensitivity] || 8;

  shell.addEventListener("wheel", event => {
    if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
    if (shell.scrollWidth <= shell.clientWidth) return;

    event.preventDefault();
    shell.scrollLeft += event.deltaY;
  }, { passive: false });

  shell.addEventListener("pointerdown", event => {
    if (event.button !== undefined && event.button !== 0) return;

    pointerActive = true;
    dragMoved = false;
    ignoreNextClick = false;
    activePointerId = event.pointerId;
    startX = event.clientX;
    startY = event.clientY;
    startScrollLeft = shell.scrollLeft;
  });

  shell.addEventListener("pointermove", event => {
    if (!pointerActive) return;
    if (activePointerId !== null && event.pointerId !== activePointerId) return;

    const dx = event.clientX - startX;
    const dy = event.clientY - startY;

    const hasHorizontalIntent =
      Math.abs(dx) > dragThreshold &&
      Math.abs(dx) > Math.abs(dy) * 1.15;

    if (!dragMoved && hasHorizontalIntent) {
      dragMoved = true;
      ignoreNextClick = true;
      shell.classList.add("dragging");
      shell.setPointerCapture?.(event.pointerId);
    }

    if (dragMoved) {
      event.preventDefault();
      shell.scrollLeft = startScrollLeft - dx;
    }
  }, { passive: false });

  ["pointerup", "pointercancel", "pointerleave"].forEach(type => {
    shell.addEventListener(type, event => {
      if (!pointerActive) return;

      pointerActive = false;
      activePointerId = null;

      if (dragMoved) {
        ignoreNextClick = true;
        shell.releasePointerCapture?.(event.pointerId);
      }

      shell.classList.remove("dragging");

      window.setTimeout(() => {
        ignoreNextClick = false;
      }, 120);
    });
  });

  shell.addEventListener("click", event => {
    if (!ignoreNextClick) return;

    event.preventDefault();
    event.stopPropagation();
    ignoreNextClick = false;
  }, true);
}

function setGuideSectionOpen(section, open) {
  const toggle = $(".guide-section-toggle", section);
  const body = $(".guide-section-body", section);
  if (!toggle || !body) return;

  section.classList.toggle("open", open);
  toggle.setAttribute("aria-expanded", String(open));

  if (prefersReducedMotion()) {
    body.style.maxHeight = open ? "none" : "0px";
    return;
  }

  if (open) {
    body.style.maxHeight = `${body.scrollHeight}px`;
  } else {
    body.style.maxHeight = `${body.scrollHeight}px`;
    requestAnimationFrame(() => { body.style.maxHeight = "0px"; });
  }
}

function refreshOpenGuideHeights() {
  $$(".guide-section.open .guide-section-body").forEach(body => {
    body.style.maxHeight = prefersReducedMotion() ? "none" : `${body.scrollHeight}px`;
  });
}

function initGuideAccordion() {
  const accordion = $("#guideAccordion");
  if (!accordion || accordion.dataset.guideBound === "true") return;
  accordion.dataset.guideBound = "true";

  const toggles = $$(".guide-section-toggle", accordion);
  toggles.forEach((toggle, index) => {
    const section = toggle.closest(".guide-section");
    if (!section) return;
    setGuideSectionOpen(section, section.classList.contains("open") || toggle.getAttribute("aria-expanded") === "true");

    toggle.addEventListener("click", () => {
      setGuideSectionOpen(section, toggle.getAttribute("aria-expanded") !== "true");
    });

    toggle.addEventListener("keydown", event => {
      if (!["ArrowDown", "ArrowUp"].includes(event.key)) return;
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      const next = toggles[(index + direction + toggles.length) % toggles.length];
      next?.focus();
    });
  });

  window.addEventListener("resize", refreshOpenGuideHeights);
}

// ══════════════════════════════════════════════════════════════
// PANEL NAVIGATION
// ══════════════════════════════════════════════════════════════
function switchPanel(name) {
  const panelName = String(name || "").trim();
  if (!panelName) return false;
  const prevPanel = $(".nav-btn.active")?.dataset.panel;

  const targetId = `panel-${panelName}`;
  const targetPanel = document.getElementById(targetId);

  if (!targetPanel) {
    console.warn(`[DepthLens] Cannot switch panel: missing #${targetId}`);
    return false;
  }

  const navBtns = $$(".nav-btn[data-panel]");
  const panels = $$(".panel");

  navBtns.forEach(btn => {
    const isActive = btn.dataset.panel === panelName;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-current", isActive ? "page" : "false");
  });

  panels.forEach(panel => {
    const isActive = panel.id === targetId;
    panel.hidden = !isActive;
    panel.classList.toggle("active", isActive);
  });

  const activeBtn = navBtns.find(btn => btn.dataset.panel === panelName);
  activeBtn?.scrollIntoView({
    block: "nearest",
    inline: "center",
    behavior: prefersReducedMotion() ? "auto" : "smooth",
  });

  if (settings.rememberLastTab) { try { localStorage.setItem(LAST_TAB_KEY, panelName); } catch {} }
  if (settings.stopCameraOnTabSwitch && prevPanel === "webcam" && panelName !== "webcam") stopWebcam({ quiet: true });

  if (panelName === "performance" && state.performanceView === "observability") loadObservability({ quiet: true });

  if (panelName === "guide") {
    refreshOpenGuideHeights();
  }

  if (panelName === "reconstruct") {
    syncReconstructControls();
    resizePointCloudCanvas();
    drawPointCloudFrame();
    if (state.reconstruct.viewer.autoRotate) startPointCloudViewer();
  } else {
    stopPointCloudViewer();
  }

  const resizeCharts = window.DepthLensCharts?.scheduleChartResize
    || (typeof scheduleChartResize === "function" ? scheduleChartResize : null);
  if (resizeCharts) requestAnimationFrame(() => resizeCharts());

  return true;
}

function bindPanelNavigation() {
  const nav = document.querySelector(".header-nav");
  if (!nav || nav.dataset.panelNavBound === "true") return;

  nav.dataset.panelNavBound = "true";

  nav.addEventListener("click", event => {
    const btn = event.target.closest?.(".nav-btn[data-panel]");
    if (!btn || !nav.contains(btn)) return;

    event.preventDefault();
    event.stopPropagation();

    switchPanel(btn.dataset.panel);
  });

  nav.addEventListener("keydown", event => {
    if (event.key !== "Enter" && event.key !== " ") return;

    const btn = event.target.closest?.(".nav-btn[data-panel]");
    if (!btn || !nav.contains(btn)) return;

    event.preventDefault();
    switchPanel(btn.dataset.panel);
  });
}

bindPanelNavigation();

// ══════════════════════════════════════════════════════════════
// GROUND TRUTH MODE
// ══════════════════════════════════════════════════════════════
const GT_ALLOWED = /\.(png|tif|tiff|npy)$/i;
function syncGtUi() {
  if (!el.gtToggle) return;
  state.gtMode = Boolean(el.gtToggle.checked);
  if (el.gtUpload) el.gtUpload.hidden = !state.gtMode;
  if (el.fileInput) el.fileInput.multiple = !state.gtMode;
  if (!state.gtMode) {
    state.gtFile = null;
    if (el.gtFileInput) el.gtFileInput.value = "";
    if (el.gtFileName) el.gtFileName.textContent = "No GT file selected";
  }
  syncQueueControls();
}
function setGtFile(file) {
  if (!file) return;
  if (!GT_ALLOWED.test(file.name)) { toast("GT depth file must be PNG, TIFF, or NPY", "warning"); return; }
  if (file.size > 20*1024*1024) { toast("GT depth file exceeds 20 MB", "warning"); return; }
  state.gtFile = file;
  if (el.gtFileName) el.gtFileName.textContent = file.name;
  syncQueueControls();
}
el.gtToggle?.addEventListener("change", () => {
  syncGtUi();
  toast(state.gtMode ? "GT mode enabled — add one image and one GT depth file" : "GT mode disabled — batch upload restored", "info");
});
el.gtFileInput?.addEventListener("change", () => { setGtFile(el.gtFileInput.files[0]); });
el.clearGtBtn?.addEventListener("click", () => { state.gtFile=null; if (el.gtFileInput) el.gtFileInput.value=""; if (el.gtFileName) el.gtFileName.textContent="No GT file selected"; syncQueueControls(); });
