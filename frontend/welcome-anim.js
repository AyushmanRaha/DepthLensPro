/**
 * DepthLens Pro — Welcome Screen Animation v6.0
 * Depth Field Calibration: technical grid → scanner sweep → depth contours → crisp logo.
 * Canvas-based, theme-aware, reduced-motion friendly, and dependency-free.
 */
(function initWelcomeAnimation() {
  "use strict";

  const logoCanvas = document.getElementById("logoCanvas");
  const bgCanvas = document.getElementById("welcomeBgCanvas");
  if (!logoCanvas || !bgCanvas) return;

  const logoCtx = logoCanvas.getContext("2d");
  const bgCtx = bgCanvas.getContext("2d");
  const stage = logoCanvas.parentElement;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const DURATION = 2100;
  const PHASE_GRID_END = 400;
  const PHASE_SCAN_START = 400;
  const PHASE_SCAN_END = 1100;
  const PHASE_RESOLVE_START = 1100;
  const PHASE_RESOLVE_END = 1700;
  const PHASE_POLISH_START = 1700;
  const PHASE_POLISH_END = 2100;

  let dpr = 1;
  let bgW = 0;
  let bgH = 0;
  let logoW = 0;
  let logoH = 0;
  let logoMetrics = null;
  let pointCloud = [];
  let contourLines = [];
  let startTime = 0;
  let rafId = 0;
  let done = false;
  let hasReducedMotion = reduceMotion.matches;

  function isDark() {
    return document.documentElement.getAttribute("data-theme") !== "light";
  }

  function getPalette() {
    if (isDark()) {
      return {
        bgTop: "#06101c",
        bgBottom: "#091421",
        grid: "rgba(142, 190, 224, 0.075)",
        gridMajor: "rgba(128, 211, 255, 0.13)",
        point: "rgba(138, 218, 255, 0.34)",
        pointDim: "rgba(138, 218, 255, 0.12)",
        scan: "rgba(51, 214, 255, 0.88)",
        scanSoft: "rgba(51, 214, 255, 0.16)",
        contour: "rgba(108, 219, 255, 0.54)",
        contourDim: "rgba(121, 174, 215, 0.18)",
        textTop: "#ffffff",
        textMid: "#e8f6ff",
        textBottom: "#aacce8",
        proTop: "#6cf2ff",
        proMid: "#00c8ff",
        proBottom: "#167db8",
        shadow: "rgba(0, 18, 32, 0.42)",
        sweep: "rgba(255, 255, 255, 0.24)",
      };
    }

    return {
      bgTop: "#f5f8fc",
      bgBottom: "#eaf1f8",
      grid: "rgba(19, 74, 122, 0.075)",
      gridMajor: "rgba(0, 112, 204, 0.12)",
      point: "rgba(0, 112, 204, 0.30)",
      pointDim: "rgba(0, 80, 150, 0.10)",
      scan: "rgba(0, 102, 204, 0.82)",
      scanSoft: "rgba(0, 112, 204, 0.13)",
      contour: "rgba(0, 112, 204, 0.45)",
      contourDim: "rgba(19, 74, 122, 0.16)",
      textTop: "#06192c",
      textMid: "#0d2744",
      textBottom: "#1f466b",
      proTop: "#0094eb",
      proMid: "#0070cc",
      proBottom: "#004f9e",
      shadow: "rgba(0, 45, 95, 0.14)",
      sweep: "rgba(255, 255, 255, 0.55)",
    };
  }

  function easeOutCubic(t) {
    t = clamp01(t);
    return 1 - Math.pow(1 - t, 3);
  }

  function easeInOutCubic(t) {
    t = clamp01(t);
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  function seededRandom(seed) {
    let s = seed >>> 0;
    return function next() {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  function resizeCanvas(canvas, ctx, width, height) {
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(height * dpr));
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function resizeBg() {
    bgW = window.innerWidth || document.documentElement.clientWidth || 1;
    bgH = window.innerHeight || document.documentElement.clientHeight || 1;
    resizeCanvas(bgCanvas, bgCtx, bgW, bgH);
    buildPointCloud();
  }

  function resizeLogo() {
    const rect = stage.getBoundingClientRect();
    logoW = Math.max(1, rect.width || stage.clientWidth || 1);
    logoH = Math.max(1, rect.height || stage.clientHeight || 1);
    resizeCanvas(logoCanvas, logoCtx, logoW, logoH);
    buildLogoMetrics();
    buildContourLines();
  }

  function resizeAll() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    resizeBg();
    resizeLogo();
  }

  function getLogoFont(fontSize) {
  const displayFont = getComputedStyle(document.documentElement)
    .getPropertyValue("--ff-display")
    .trim();

  return `700 ${fontSize}px ${displayFont || "Rajdhani, Exo 2, sans-serif"}`;
}

function measureLogoParts(fontSize) {
  logoCtx.font = getLogoFont(fontSize);

  const mainText = "DepthLens";
  const proText = "Pro";

  const main = logoCtx.measureText(mainText);
  const pro = logoCtx.measureText(proText);

  /*
    Slight negative gap makes DepthLensPro read as one logo,
    not two separated words.
  */
  const proGap = -fontSize * 0.035;
  const totalWidth = main.width + proGap + pro.width;

  return {
    main,
    pro,
    proGap,
    totalWidth,
  };
}

function buildLogoMetrics() {
  const vw = window.innerWidth || document.documentElement.clientWidth || logoW;
  const isMobile = vw <= 560;

  const clampValue = (value, min, max) => Math.max(min, Math.min(max, value));

  let fontSize = isMobile
    ? clampValue(vw * 0.20, 84, 155)
    : clampValue(vw * 0.135, 140, 240);

  fontSize = Math.min(fontSize, logoH * 0.82);

  logoCtx.textBaseline = "middle";
  logoCtx.textAlign = "left";

  let measured = measureLogoParts(fontSize);
  const maxTextWidth = logoW * 0.92;

  if (measured.totalWidth > maxTextWidth) {
    fontSize = Math.max(
      isMobile ? 72 : 120,
      Math.floor(fontSize * (maxTextWidth / measured.totalWidth))
    );

    measured = measureLogoParts(fontSize);
  }

  const ascent = Math.max(
    measured.main.actualBoundingBoxAscent || 0,
    measured.pro.actualBoundingBoxAscent || 0,
    fontSize * 0.72
  );

  const descent = Math.max(
    measured.main.actualBoundingBoxDescent || 0,
    measured.pro.actualBoundingBoxDescent || 0,
    fontSize * 0.22
  );

  const x = (logoW - measured.totalWidth) / 2;
  const y = logoH / 2 + (ascent - descent) / 2;

  logoMetrics = {
    fontSize,
    x,
    y,
    mainWidth: measured.main.width,
    proWidth: measured.pro.width,
    proGap: measured.proGap,
    totalWidth: measured.totalWidth,
    textTop: y - ascent,
    textBottom: y + descent,
  };
}

  function buildPointCloud() {
    const rand = seededRandom(Math.floor(bgW * 17 + bgH * 31));
    const count = Math.max(80, Math.min(160, Math.round((bgW * bgH) / 13000)));
    pointCloud = Array.from({ length: count }, function makePoint(_, index) {
      const depth = Math.pow(rand(), 1.45);
      return {
        x: rand() * bgW,
        y: rand() * bgH,
        z: depth,
        r: lerp(0.55, 1.55, depth),
        phase: rand() * Math.PI * 2,
        drift: lerp(0.22, 0.85, rand()),
        alpha: lerp(0.16, 0.46, depth) * (index % 7 === 0 ? 1.25 : 1),
      };
    });
  }

  function buildContourLines() {
    if (!logoMetrics) return;

    const rand = seededRandom(Math.floor(logoW * 13 + logoH * 29));
    const lines = [];
    const count = 12;
    const left = logoMetrics.x - logoMetrics.fontSize * 0.22;
    const right = logoMetrics.x + logoMetrics.totalWidth + logoMetrics.fontSize * 0.18;
    const width = right - left;
    const centerY = logoH / 2;
    const band = logoMetrics.fontSize * 0.74;

    for (let i = 0; i < count; i += 1) {
      const yBase = centerY - band / 2 + (i / (count - 1)) * band;
      const amp = logoMetrics.fontSize * lerp(0.035, 0.090, rand());
      const freq = lerp(1.4, 2.8, rand());
      const phase = rand() * Math.PI * 2;
      const points = [];
      const segments = 64;

      for (let j = 0; j <= segments; j += 1) {
        const u = j / segments;
        const edgeFalloff = Math.sin(Math.PI * u);
        const x = left + width * u;
        const wave = Math.sin(u * Math.PI * 2 * freq + phase);
        const micro = Math.sin(u * Math.PI * 6 + phase * 0.7) * 0.35;
        const y = yBase + (wave + micro) * amp * edgeFalloff;
        points.push({ x, y });
      }

      lines.push({
        points,
        alpha: lerp(0.35, 0.82, rand()),
        offset: i / count,
        width: i % 4 === 0 ? 1.15 : 0.85,
      });
    }

    contourLines = lines;
  }

  function drawBackground(time, progress) {
    const palette = getPalette();
    const gridAlpha = easeOutCubic(Math.min(progress / (PHASE_GRID_END / DURATION), 1));
    const drift = hasReducedMotion ? 0 : time * 0.004;
    const bgGradient = bgCtx.createLinearGradient(0, 0, 0, bgH);
    bgGradient.addColorStop(0, palette.bgTop);
    bgGradient.addColorStop(1, palette.bgBottom);

    bgCtx.clearRect(0, 0, bgW, bgH);
    bgCtx.fillStyle = bgGradient;
    bgCtx.fillRect(0, 0, bgW, bgH);

    bgCtx.save();
    bgCtx.globalAlpha = 0.72 * gridAlpha;
    drawGrid(palette, drift);
    bgCtx.restore();

    drawPointCloud(time, lerp(0.36, 1, gridAlpha));
  }

  function drawGrid(palette, drift) {
    const fine = 36;
    const majorEvery = 4;
    const offsetX = (drift * 3) % fine;
    const offsetY = (drift * 2) % fine;

    for (let x = -fine + offsetX; x <= bgW + fine; x += fine) {
      const column = Math.round((x - offsetX) / fine);
      bgCtx.beginPath();
      bgCtx.moveTo(x, 0);
      bgCtx.lineTo(x, bgH);
      bgCtx.lineWidth = column % majorEvery === 0 ? 0.9 : 0.55;
      bgCtx.strokeStyle = column % majorEvery === 0 ? palette.gridMajor : palette.grid;
      bgCtx.stroke();
    }

    for (let y = -fine + offsetY; y <= bgH + fine; y += fine) {
      const row = Math.round((y - offsetY) / fine);
      bgCtx.beginPath();
      bgCtx.moveTo(0, y);
      bgCtx.lineTo(bgW, y);
      bgCtx.lineWidth = row % majorEvery === 0 ? 0.9 : 0.55;
      bgCtx.strokeStyle = row % majorEvery === 0 ? palette.gridMajor : palette.grid;
      bgCtx.stroke();
    }

    const cx = bgW / 2;
    const cy = bgH / 2;
    bgCtx.strokeStyle = palette.gridMajor;
    bgCtx.lineWidth = 1;
    bgCtx.globalAlpha *= 0.55;
    bgCtx.beginPath();
    bgCtx.moveTo(cx - 28, cy);
    bgCtx.lineTo(cx + 28, cy);
    bgCtx.moveTo(cx, cy - 28);
    bgCtx.lineTo(cx, cy + 28);
    bgCtx.stroke();
  }

  function drawPointCloud(time, alpha) {
    const palette = getPalette();
    bgCtx.save();
    pointCloud.forEach(function drawPoint(point) {
      const driftX = hasReducedMotion ? 0 : Math.cos(time * 0.00016 * point.drift + point.phase) * point.z * 7;
      const driftY = hasReducedMotion ? 0 : Math.sin(time * 0.00013 * point.drift + point.phase) * point.z * 5;
      const opacity = point.alpha * alpha;
      bgCtx.beginPath();
      bgCtx.arc(point.x + driftX, point.y + driftY, point.r, 0, Math.PI * 2);
      bgCtx.fillStyle = point.z > 0.54 ? palette.point : palette.pointDim;
      bgCtx.globalAlpha = opacity;
      bgCtx.fill();
    });
    bgCtx.restore();
  }

  function drawLogoFrame(elapsed, time) {
    const palette = getPalette();
    const scanT = clamp01((elapsed - PHASE_SCAN_START) / (PHASE_SCAN_END - PHASE_SCAN_START));
    const resolveT = clamp01((elapsed - PHASE_RESOLVE_START) / (PHASE_RESOLVE_END - PHASE_RESOLVE_START));
    const polishT = clamp01((elapsed - PHASE_POLISH_START) / (PHASE_POLISH_END - PHASE_POLISH_START));
    const logoAlpha = easeOutCubic(resolveT);
    const contourAlpha = elapsed < PHASE_SCAN_START ? 0 : 1 - easeInOutCubic(resolveT);
    const sweepProgress = easeInOutCubic(polishT);
    const scanPosition = scanT;

    logoCtx.clearRect(0, 0, logoW, logoH);

    if (elapsed >= PHASE_SCAN_START && elapsed <= PHASE_RESOLVE_END) {
      drawContours(contourAlpha, scanPosition, time);
    }

    if (elapsed >= PHASE_SCAN_START && elapsed <= PHASE_SCAN_END) {
      drawScanBeam(scanPosition);
    }

    if (logoAlpha > 0) {
      drawLogo(logoAlpha, sweepProgress, false);
    } else if (elapsed > PHASE_SCAN_START) {
      drawCalibrationTicks(palette, scanT * 0.55);
    }
  }

  function drawScanBeam(progress) {
    if (!logoMetrics) return;
    const palette = getPalette();
    const left = logoMetrics.x - logoMetrics.fontSize * 0.55;
    const right = logoMetrics.x + logoMetrics.totalWidth + logoMetrics.fontSize * 0.55;
    const x = lerp(left, right, easeInOutCubic(progress));
    const top = logoMetrics.textTop - logoMetrics.fontSize * 0.18;
    const bottom = logoMetrics.textBottom + logoMetrics.fontSize * 0.20;

    logoCtx.save();
    logoCtx.globalCompositeOperation = "lighter";

    const trail = logoCtx.createLinearGradient(x - 78, 0, x + 20, 0);
    trail.addColorStop(0, "rgba(0, 0, 0, 0)");
    trail.addColorStop(0.70, palette.scanSoft);
    trail.addColorStop(1, palette.scan);
    logoCtx.fillStyle = trail;
    logoCtx.fillRect(x - 78, top, 98, bottom - top);

    logoCtx.strokeStyle = palette.scan;
    logoCtx.lineWidth = 1.25;
    logoCtx.beginPath();
    logoCtx.moveTo(x, top);
    logoCtx.lineTo(x + logoMetrics.fontSize * 0.10, bottom);
    logoCtx.stroke();

    logoCtx.globalAlpha = 0.28;
    logoCtx.lineWidth = 5;
    logoCtx.strokeStyle = palette.scanSoft;
    logoCtx.beginPath();
    logoCtx.moveTo(x - 2, top);
    logoCtx.lineTo(x + logoMetrics.fontSize * 0.10 - 2, bottom);
    logoCtx.stroke();
    logoCtx.restore();
  }

  function drawContours(alpha, scanPosition, time) {
    if (!logoMetrics) return;
    const palette = getPalette();
    const revealX = lerp(
      logoMetrics.x - logoMetrics.fontSize * 0.5,
      logoMetrics.x + logoMetrics.totalWidth + logoMetrics.fontSize * 0.5,
      easeInOutCubic(scanPosition)
    );

    logoCtx.save();
    logoCtx.lineCap = "round";
    logoCtx.lineJoin = "round";
    logoCtx.beginPath();
    logoCtx.rect(
      logoMetrics.x - logoMetrics.fontSize * 0.60,
      logoMetrics.textTop - logoMetrics.fontSize * 0.24,
      Math.max(0, revealX - (logoMetrics.x - logoMetrics.fontSize * 0.60)),
      logoMetrics.fontSize * 1.15
    );
    logoCtx.clip();

    contourLines.forEach(function drawLine(line, index) {
      const localAlpha = alpha * line.alpha * (0.78 + 0.22 * Math.sin(time * 0.0012 + index));
      logoCtx.globalAlpha = localAlpha;
      logoCtx.strokeStyle = index % 3 === 0 ? palette.contour : palette.contourDim;
      logoCtx.lineWidth = line.width;
      logoCtx.beginPath();
      line.points.forEach(function drawPoint(point, pointIndex) {
        if (pointIndex === 0) logoCtx.moveTo(point.x, point.y);
        else logoCtx.lineTo(point.x, point.y);
      });
      logoCtx.stroke();
    });

    drawCalibrationTicks(palette, alpha * 0.55);
    logoCtx.restore();
  }

  function drawCalibrationTicks(palette, alpha) {
    if (!logoMetrics || alpha <= 0) return;
    const pad = logoMetrics.fontSize * 0.32;
    const left = logoMetrics.x - pad;
    const right = logoMetrics.x + logoMetrics.totalWidth + pad;
    const top = logoMetrics.textTop - pad * 0.55;
    const bottom = logoMetrics.textBottom + pad * 0.55;
    const tick = Math.max(10, logoMetrics.fontSize * 0.11);

    logoCtx.save();
    logoCtx.globalAlpha = alpha;
    logoCtx.strokeStyle = palette.gridMajor;
    logoCtx.lineWidth = 1;
    [[left, top, 1, 1], [right, top, -1, 1], [left, bottom, 1, -1], [right, bottom, -1, -1]].forEach(
      function corner(c) {
        const x = c[0];
        const y = c[1];
        logoCtx.beginPath();
        logoCtx.moveTo(x, y + tick * c[3]);
        logoCtx.lineTo(x, y);
        logoCtx.lineTo(x + tick * c[2], y);
        logoCtx.stroke();
      }
    );
    logoCtx.restore();
  }

  function drawLogo(alpha, sweepProgress, finalState) {
  if (!logoMetrics) return;

  const palette = getPalette();
  const x = logoMetrics.x;
  const y = logoMetrics.y;
  const fontSize = logoMetrics.fontSize;

  const mainText = "DepthLens";
  const proText = "Pro";

  const mainX = x;
  const proX = x + logoMetrics.mainWidth + logoMetrics.proGap;

  const revealWidth = finalState
    ? logoMetrics.totalWidth
    : logoMetrics.totalWidth * easeOutCubic(alpha);

  logoCtx.save();

  logoCtx.font = getLogoFont(fontSize);
  logoCtx.textBaseline = "middle";
  logoCtx.textAlign = "left";
  logoCtx.globalAlpha = clamp01(alpha);
  logoCtx.shadowColor = palette.shadow;
  logoCtx.shadowBlur = isDark() ? 10 : 5;
  logoCtx.shadowOffsetY = isDark() ? 2 : 1;

  logoCtx.beginPath();
  logoCtx.rect(x - 4, 0, revealWidth + 8, logoH);
  logoCtx.clip();

  const mainGradient = logoCtx.createLinearGradient(
    0,
    y - fontSize * 0.48,
    0,
    y + fontSize * 0.42
  );

  mainGradient.addColorStop(0, palette.textTop);
  mainGradient.addColorStop(0.52, palette.textMid);
  mainGradient.addColorStop(1, palette.textBottom);

  logoCtx.fillStyle = mainGradient;
  logoCtx.fillText(mainText, mainX, y);

  const proGradient = logoCtx.createLinearGradient(
    0,
    y - fontSize * 0.48,
    0,
    y + fontSize * 0.42
  );

  proGradient.addColorStop(0, palette.proTop);
  proGradient.addColorStop(0.52, palette.proMid);
  proGradient.addColorStop(1, palette.proBottom);

  logoCtx.fillStyle = proGradient;
  logoCtx.fillText(proText, proX, y);

  if (sweepProgress > 0 && sweepProgress < 1) {
    logoCtx.shadowBlur = 0;
    logoCtx.globalCompositeOperation = "source-atop";

    const sweepX = lerp(
      x - fontSize * 0.9,
      x + logoMetrics.totalWidth + fontSize * 0.9,
      sweepProgress
    );

    const sweep = logoCtx.createLinearGradient(sweepX - 45, 0, sweepX + 45, 0);
    sweep.addColorStop(0, "rgba(255, 255, 255, 0)");
    sweep.addColorStop(0.5, palette.sweep);
    sweep.addColorStop(1, "rgba(255, 255, 255, 0)");

    logoCtx.fillStyle = sweep;
    logoCtx.fillRect(
      x - 10,
      logoMetrics.textTop - 10,
      logoMetrics.totalWidth + 20,
      fontSize
    );
  }

  logoCtx.restore();

  logoCanvas.style.filter = isDark()
    ? "drop-shadow(0 10px 26px rgba(0, 0, 0, 0.28))"
    : "drop-shadow(0 8px 18px rgba(0, 54, 110, 0.10))";
}

  function drawFinalLogo() {
    logoCtx.clearRect(0, 0, logoW, logoH);
    drawLogo(1, 1, true);
  }

  function revealControls() {
    const cta = document.querySelector(".welcome-cta-wrap");
    const theme = document.querySelector(".theme-toggle-landing");
    if (cta) cta.classList.add("revealed");
    if (theme) theme.classList.add("revealed");
  }

  function render(timestamp) {
    if (done || hasReducedMotion) return;
    if (!startTime) startTime = timestamp;

    const elapsed = timestamp - startTime;
    const progress = clamp01(elapsed / DURATION);

    drawBackground(timestamp, progress);
    drawLogoFrame(elapsed, timestamp);

    if (elapsed >= DURATION) {
      done = true;
      drawBackground(timestamp, 1);
      drawFinalLogo();
      setTimeout(revealControls, 80);
      return;
    }

    rafId = requestAnimationFrame(render);
  }

  function drawReducedMotionState() {
    done = true;
    drawBackground(0, 1);
    drawFinalLogo();
    setTimeout(revealControls, 40);
  }

  function handleThemeChange() {
    drawBackground(performance.now(), done || hasReducedMotion ? 1 : clamp01(((performance.now() - startTime) || 0) / DURATION));
    if (done || hasReducedMotion) drawFinalLogo();
  }

  function handleResize() {
    resizeAll();
    if (done || hasReducedMotion) {
      drawBackground(0, 1);
      drawFinalLogo();
    }
  }

  function handleMotionPreferenceChange(event) {
    hasReducedMotion = event.matches;
    if (hasReducedMotion) {
      if (rafId) cancelAnimationFrame(rafId);
      drawReducedMotionState();
    } else if (!done) {
      startTime = 0;
      rafId = requestAnimationFrame(render);
    }
  }

  resizeAll();

  document.addEventListener("depthlens-theme-changed", handleThemeChange);
  window.addEventListener("resize", handleResize);

  if (document.fonts && typeof document.fonts.ready?.then === "function") {
  document.fonts.ready.then(function redrawAfterFontsLoad() {
    resizeAll();

    if (done || hasReducedMotion) {
      drawBackground(0, 1);
      drawFinalLogo();
    }
  });

  if (typeof document.fonts.addEventListener === "function") {
    document.fonts.addEventListener("loadingdone", handleResize);
  }
}

  if (typeof reduceMotion.addEventListener === "function") {
    reduceMotion.addEventListener("change", handleMotionPreferenceChange);
  } else if (typeof reduceMotion.addListener === "function") {
    reduceMotion.addListener(handleMotionPreferenceChange);
  }

  if (hasReducedMotion) {
    drawReducedMotionState();
  } else {
    rafId = requestAnimationFrame(render);
  }
})();
