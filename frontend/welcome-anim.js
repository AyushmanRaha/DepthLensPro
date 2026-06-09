/**
 * DepthLens Pro — Welcome Screen Animation v5.0
 * Liquid metal / glass drops → merge → morph into logo
 * Canvas-based, 60fps, theme-aware, self-contained.
 */
(function initWelcomeAnimation() {
  "use strict";

  const canvas = document.getElementById("logoCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function isDark() {
    return document.documentElement.getAttribute("data-theme") !== "light";
  }

  /* ── Canvas resize ─────────────────────────────────── */
  const stage = canvas.parentElement;
  function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width  = stage.clientWidth  * dpr;
    canvas.height = stage.clientHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resizeCanvas();

  /* ── Easing ─────────────────────────────────────────── */
  function easeInOut(t)  { return t < 0.5 ? 2*t*t : -1+(4-2*t)*t; }
  function easeOutElastic(t) {
    if (t===0||t===1) return t;
    return Math.pow(2,-10*t)*Math.sin((t*10-0.75)*(2*Math.PI/3))+1;
  }
  function easeOutCubic(t) { return 1-Math.pow(1-t,3); }
  function easeInCubic(t)  { return t*t*t; }
  function lerp(a,b,t)     { return a+(b-a)*t; }

  /* ── Dimensions helpers ─────────────────────────────── */
  const W = () => stage.clientWidth;
  const H = () => stage.clientHeight;

  /* ── Drop data ──────────────────────────────────────── */
  function makeDrops() {
    const w=W(), h=H(), cx=w/2, cy=h/2;
    const count = 14 + Math.floor(Math.random()*6);
    return Array.from({length:count}, (_,i) => {
      const angle = (i/count)*Math.PI*2;
      const radius = 0.32*w + Math.random()*0.14*w;
      return {
        sx: cx + Math.cos(angle)*radius,
        sy: cy + Math.sin(angle)*radius + (Math.random()-0.5)*h*0.45,
        ex: cx + (Math.random()-0.5)*18,
        ey: cy + (Math.random()-0.5)*12,
        r: 4 + Math.random()*9,
        phase: Math.random()*Math.PI*2,
        speed: 0.55 + Math.random()*0.9,
        hue: 185 + Math.random()*70,
      };
    });
  }
  const drops = makeDrops();

  /* ── Logo pixel sampler ─────────────────────────────── */
  let logoPixels = null;
  function buildLogoPixels() {
    const w=W(), h=H();
    const tc = document.createElement("canvas");
    tc.width=w; tc.height=h;
    const t2 = tc.getContext("2d");
    const fontSize = Math.min(h*0.72, w*0.125);
    t2.font = `700 ${fontSize}px Rajdhani, sans-serif`;
    t2.textBaseline = "middle";
    t2.textAlign    = "left";
    const mMain = t2.measureText("DepthLens");
    const mPro  = t2.measureText("Pro");
    const totalW = mMain.width + mPro.width;
    const sx = (w-totalW)/2, cy = h/2;
    t2.fillStyle = "#dff0ff";
    t2.fillText("DepthLens", sx, cy);
    t2.fillStyle = "#00c8ff";
    t2.fillText("Pro", sx+mMain.width, cy);
    const data = t2.getImageData(0,0,w,h).data;
    const pts  = [];
    const step = 3;
    for (let y=0; y<h; y+=step) {
      for (let x=0; x<w; x+=step) {
        const idx=(y*w+x)*4;
        if (data[idx+3]>80) pts.push({x,y,r:data[idx],g:data[idx+1],b:data[idx+2]});
      }
    }
    logoPixels = pts;
  }

  /* ── Particle pool ──────────────────────────────────── */
  let particles = [];
  function buildParticles() {
    if (!logoPixels||!logoPixels.length) return;
    const cx=W()/2, cy=H()/2;
    const count = Math.min(logoPixels.length, 1400);
    const step  = Math.floor(logoPixels.length/count);
    particles = Array.from({length:count}, (_,i) => {
      const tgt = logoPixels[i*step];
      return {
        sx: cx+(Math.random()-0.5)*44,
        sy: cy+(Math.random()-0.5)*20,
        tx: tgt.x, ty: tgt.y,
        r: tgt.r, g: tgt.g, b: tgt.b,
        delay: Math.random()*0.38,
        size: 1.1+Math.random()*0.9,
      };
    });
  }

  /* ── Metallic blob draw ─────────────────────────────── */
  function drawBlob(ctx, x, y, r, t, hue, alpha) {
    const wobble = Math.sin(t*3.5+hue)*r*0.22;
    const rw = r+wobble;
    const dark = isDark();
    const g = ctx.createRadialGradient(x-rw*0.3,y-rw*0.35,rw*0.05, x,y,rw);
    if (dark) {
      g.addColorStop(0,   `hsla(${hue},100%,90%,${alpha})`);
      g.addColorStop(0.35,`hsla(${hue}, 90%,62%,${alpha*0.9})`);
      g.addColorStop(0.7, `hsla(${hue}, 80%,38%,${alpha*0.72})`);
      g.addColorStop(1,   `hsla(${hue}, 70%,16%,${alpha*0.4})`);
    } else {
      g.addColorStop(0,   `hsla(${hue}, 90%,82%,${alpha})`);
      g.addColorStop(0.35,`hsla(${hue}, 80%,52%,${alpha*0.9})`);
      g.addColorStop(0.7, `hsla(${hue}, 70%,30%,${alpha*0.72})`);
      g.addColorStop(1,   `hsla(${hue}, 60%,12%,${alpha*0.4})`);
    }
    ctx.save();
    ctx.beginPath();
    const pts=6;
    for (let i=0; i<=pts; i++) {
      const a=(i/pts)*Math.PI*2;
      const wr=rw*(1+0.18*Math.sin(t*2+i*1.1+hue));
      const px=x+Math.cos(a)*wr, py=y+Math.sin(a)*wr;
      if (i===0) { ctx.moveTo(px,py); continue; }
      const prev=((i-1)/pts)*Math.PI*2;
      const cr=rw*(1+0.18*Math.sin(t*2+(i-0.5)*1.1+hue));
      ctx.quadraticCurveTo(
        x+Math.cos((prev+a)/2)*cr*1.15,
        y+Math.sin((prev+a)/2)*cr*1.15,
        px, py
      );
    }
    ctx.closePath();
    ctx.fillStyle = g;
    ctx.fill();
    /* specular */
    ctx.beginPath();
    ctx.arc(x-rw*0.28, y-rw*0.32, rw*0.22, 0, Math.PI*2);
    ctx.fillStyle = `hsla(${hue},80%,96%,${alpha*0.52})`;
    ctx.fill();
    ctx.restore();
  }

  /* ── Welcome background canvas (vector lines) ───────── */
  const bgCanvas = document.getElementById("welcomeBgCanvas");
  let   bgCtx    = null;
  if (bgCanvas) {
    bgCtx = bgCanvas.getContext("2d");
    resizeBg(); drawBg();
  }

  function resizeBg() {
    if (!bgCanvas) return;
    const dpr = Math.min(window.devicePixelRatio||1,2);
    bgCanvas.width  = window.innerWidth  * dpr;
    bgCanvas.height = window.innerHeight * dpr;
    bgCtx.setTransform(dpr,0,0,dpr,0,0);
  }

  function drawBg() {
    if (!bgCanvas||!bgCtx) return;
    const w=window.innerWidth, h=window.innerHeight;
    bgCtx.clearRect(0,0,w,h);
    const dark = isDark();

    /* ── theme-dependent colors — clearly visible both modes ── */
    const cGrid   = dark ? "rgba(0,200,255,0.13)"  : "rgba(0,90,180,0.14)";
    const cGridFn = dark ? "rgba(0,200,255,0.055)" : "rgba(0,90,180,0.065)";
    const cDiag   = dark ? "rgba(123,92,248,0.09)" : "rgba(80,50,200,0.09)";
    const cDot    = dark ? "rgba(0,200,255,0.30)"  : "rgba(0,90,180,0.32)";
    const cConn   = dark ? "rgba(0,200,255,0.10)"  : "rgba(0,90,180,0.12)";
    const cNode   = dark ? "rgba(0,200,255,0.12)"  : "rgba(0,90,180,0.10)";

    const gs = 56; /* grid spacing */

    /* fine grid */
    bgCtx.lineWidth = 0.5;
    bgCtx.strokeStyle = cGridFn;
    for (let x=0;x<w;x+=gs/2){ bgCtx.beginPath();bgCtx.moveTo(x,0);bgCtx.lineTo(x,h);bgCtx.stroke(); }
    for (let y=0;y<h;y+=gs/2){ bgCtx.beginPath();bgCtx.moveTo(0,y);bgCtx.lineTo(w,y);bgCtx.stroke(); }

    /* bold grid */
    bgCtx.lineWidth = 0.9;
    bgCtx.strokeStyle = cGrid;
    for (let x=0;x<w;x+=gs){ bgCtx.beginPath();bgCtx.moveTo(x,0);bgCtx.lineTo(x,h);bgCtx.stroke(); }
    for (let y=0;y<h;y+=gs){ bgCtx.beginPath();bgCtx.moveTo(0,y);bgCtx.lineTo(w,y);bgCtx.stroke(); }

    /* diagonal accents */
    bgCtx.lineWidth = 0.75;
    bgCtx.strokeStyle = cDiag;
    const dsp = 200;
    for (let d=-h; d<w+h; d+=dsp){
      bgCtx.beginPath(); bgCtx.moveTo(d,0);   bgCtx.lineTo(d+h,h);   bgCtx.stroke();
    }
    for (let d=-h; d<w+h; d+=dsp*1.5){
      bgCtx.beginPath(); bgCtx.moveTo(d+h,0); bgCtx.lineTo(d,h);     bgCtx.stroke();
    }

    /* intersection dots */
    bgCtx.fillStyle = cDot;
    for (let x=0;x<w;x+=gs){
      for (let y=0;y<h;y+=gs){
        if ((Math.floor(x/gs)+Math.floor(y/gs))%3===0){
          bgCtx.beginPath(); bgCtx.arc(x,y,2,0,Math.PI*2); bgCtx.fill();
        }
      }
    }

    /* circuit connector lines (two clusters) */
    bgCtx.strokeStyle = cConn;
    bgCtx.lineWidth = 1.2;
    const segs = [
      [0,0,3,0],[0,0,0,2],[3,0,3,2],[0,2,3,2],
      [6,1,9,1],[6,1,6,4],[9,1,9,3],[6,4,9,4],
      [12,0,15,0],[12,0,12,3],[15,0,15,2],[12,3,15,3],
      [2,5,5,5],[2,5,2,8],[5,5,5,8],[2,8,5,8],
      [8,4,11,4],[8,4,8,7],[11,4,11,6],
    ];
    const drawSegs = (ox,oy) => {
      segs.forEach(([x1,y1,x2,y2])=>{
        bgCtx.beginPath();
        bgCtx.moveTo(ox+x1*gs, oy+y1*gs);
        bgCtx.lineTo(ox+x2*gs, oy+y2*gs);
        bgCtx.stroke();
      });
    };
    drawSegs(gs*2, gs*2);
    drawSegs(w-gs*13, h-gs*10);

    /* node squares at connector junctions */
    bgCtx.fillStyle = cNode;
    const nodeJcts = [[0,0],[3,0],[0,2],[3,2],[6,1],[9,1],[6,4],[9,3]];
    const drawNodes = (ox,oy) => {
      nodeJcts.forEach(([x,y])=>{
        const nx=ox+x*gs, ny=oy+y*gs;
        bgCtx.beginPath();
        bgCtx.rect(nx-4, ny-4, 8, 8);
        bgCtx.fill();
      });
    };
    drawNodes(gs*2, gs*2);
    drawNodes(w-gs*13, h-gs*10);
  }

  /* ── Final static logo draw ─────────────────────────── */
  function drawFinalLogo() {
    const w=W(), h=H();
    ctx.clearRect(0,0,w,h);
    const fontSize = Math.min(h*0.72, w*0.125);
    ctx.font = `700 ${fontSize}px Rajdhani, sans-serif`;
    ctx.textBaseline="middle"; ctx.textAlign="left";
    const mMain = ctx.measureText("DepthLens");
    const mPro  = ctx.measureText("Pro");
    const sx = (w-(mMain.width+mPro.width))/2, cy=h/2;
    const dark = isDark();

    const gm = ctx.createLinearGradient(sx, cy-fontSize/2, sx, cy+fontSize/2);
    if (dark){
      gm.addColorStop(0,"#ffffff"); gm.addColorStop(0.5,"#e8f6ff"); gm.addColorStop(1,"#a0c8e8");
    } else {
      gm.addColorStop(0,"#0a1e35"); gm.addColorStop(0.5,"#0d2744"); gm.addColorStop(1,"#1a3a5c");
    }
    ctx.fillStyle = gm;
    ctx.fillText("DepthLens", sx, cy);

    const gp = ctx.createLinearGradient(sx+mMain.width, cy-fontSize/2, sx+mMain.width, cy+fontSize/2);
    if (dark){
      gp.addColorStop(0,"#5deeff"); gp.addColorStop(0.5,"#00c8ff"); gp.addColorStop(1,"#007aaa");
    } else {
      gp.addColorStop(0,"#0088dd"); gp.addColorStop(0.5,"#0070cc"); gp.addColorStop(1,"#005599");
    }
    ctx.fillStyle = gp;
    ctx.fillText("Pro", sx+mMain.width, cy);

    canvas.style.filter = dark
      ? "drop-shadow(0 0 14px rgba(0,200,255,0.38)) drop-shadow(0 0 36px rgba(0,200,255,0.16))"
      : "drop-shadow(0 0 10px rgba(0,112,204,0.28)) drop-shadow(0 0 24px rgba(0,112,204,0.12))";
    canvas.style.transition = "filter 0.8s ease";
  }

  /* ── Reveal CTA + theme toggle ───────────────────────── */
  function revealControls() {
    const cta   = document.querySelector(".welcome-cta-wrap");
    const theme = document.querySelector(".theme-toggle-landing");
    if (cta)   cta.classList.add("revealed");
    if (theme) theme.classList.add("revealed");
  }

  /* ── Main render loop ───────────────────────────────── */
  const T = { drop:1000, merge:800, morph:1200, total:3000 };
  let startTime=null, rafId=null, done=false;

  function render(ts) {
    if (done) return;
    if (!startTime) {
      startTime=ts;
      buildLogoPixels();
      buildParticles();
    }
    const el=ts-startTime, w=W(), h=H(), t=el/1000;
    ctx.clearRect(0,0,w,h);

    /* ── Phase 1 + 2: drops fall then merge (0 → 1800ms) ── */
    if (el < T.drop+T.merge) {
      const dropT  = Math.min(el/T.drop, 1);
      const mergeT = el>T.drop ? Math.min((el-T.drop)/T.merge,1) : 0;
      const mEase  = easeInOut(mergeT);

      drops.forEach(d=>{
        const fe = easeOutElastic(Math.min(dropT*d.speed,1));
        const px = lerp(lerp(d.sx,d.ex,fe), d.ex, mEase);
        const py = lerp(lerp(d.sy,d.ey,fe), d.ey, mEase);
        const rScale = mergeT<0.5 ? 1 : 1-easeInCubic(mergeT*2-1)*0.72;
        const r = d.r * fe * rScale;
        if (r>0.5) drawBlob(ctx, px,py, r, t+d.phase, d.hue, 0.82*fe);
      });

      if (mergeT>0.28){
        const ma = easeOutCubic((mergeT-0.28)/0.72);
        drawBlob(ctx, w/2,h/2, 30*easeOutCubic(mergeT), t, 190, ma*0.88);
      }
    }

    /* ── Phase 3: morph into text (1800ms → 3000ms) ─────── */
    else {
      const morphT  = Math.min((el-(T.drop+T.merge))/T.morph, 1);
      const morphE  = easeOutCubic(morphT);
      const solidT  = morphT>0.68 ? Math.min((morphT-0.68)/0.32,1) : 0;

      /* particles stream to text positions */
      particles.forEach(p=>{
        const pt = Math.max(0, Math.min((morphT-p.delay)/(1-p.delay),1));
        const pe = easeOutCubic(pt);
        const px = lerp(p.sx,p.tx,pe);
        const py = lerp(p.sy,p.ty,pe);
        const fr = lerp(0,   p.r, solidT);
        const fg = lerp(200, p.g, solidT);
        const fb = lerp(255, p.b, solidT);
        const alpha = pt>0.05 ? Math.min(pt*3,0.9) : 0;
        const shimmer = 0.72+0.28*Math.sin(t*7+p.tx*0.018);
        ctx.beginPath();
        ctx.arc(px, py, p.size*(0.55+pe*0.45), 0, Math.PI*2);
        ctx.fillStyle = `rgba(${Math.round(fr*shimmer)},${Math.round(fg*shimmer)},${Math.round(fb)},${alpha})`;
        ctx.fill();
      });

      /* center blob fades as text forms */
      if (morphT<0.42){
        drawBlob(ctx, w/2,h/2, 24*(1-morphT/0.42), t, 190, (1-morphT/0.42)*0.65);
      }

      /* clean text fades in over particles */
      if (morphT>0.72) {
        const ta = easeOutCubic((morphT-0.72)/0.28);
        ctx.save();
        ctx.globalAlpha = ta;
        const fontSize = Math.min(h*0.72, w*0.125);
        ctx.font=`700 ${fontSize}px Rajdhani, sans-serif`;
        ctx.textBaseline="middle"; ctx.textAlign="left";
        const mMain=ctx.measureText("DepthLens");
        const mPro =ctx.measureText("Pro");
        const sx=(w-(mMain.width+mPro.width))/2, cy=h/2;
        const dark=isDark();

        const gm=ctx.createLinearGradient(sx,cy-fontSize/2,sx,cy+fontSize/2);
        if (dark){
          gm.addColorStop(0,"#ffffff"); gm.addColorStop(0.5,"#e8f6ff"); gm.addColorStop(1,"#a0c8e8");
        } else {
          gm.addColorStop(0,"#0a1e35"); gm.addColorStop(0.5,"#0d2744"); gm.addColorStop(1,"#1a3a5c");
        }
        ctx.fillStyle=gm; ctx.fillText("DepthLens",sx,cy);

        const gp=ctx.createLinearGradient(sx+mMain.width,cy-fontSize/2,sx+mMain.width,cy+fontSize/2);
        if (dark){
          gp.addColorStop(0,"#5deeff"); gp.addColorStop(0.5,"#00c8ff"); gp.addColorStop(1,"#007aaa");
        } else {
          gp.addColorStop(0,"#0088dd"); gp.addColorStop(0.5,"#0070cc"); gp.addColorStop(1,"#005599");
        }
        ctx.fillStyle=gp; ctx.fillText("Pro",sx+mMain.width,cy);

        /* light-sweep shimmer */
        const sw=((t*0.55)%2)*((w+(mMain.width+mPro.width))+120)-60;
        const sg=ctx.createLinearGradient(sw-70,0,sw+70,0);
        sg.addColorStop(0,"rgba(255,255,255,0)");
        sg.addColorStop(0.5,`rgba(255,255,255,${0.20*ta})`);
        sg.addColorStop(1,"rgba(255,255,255,0)");
        ctx.fillStyle=sg;
        ctx.fillText("DepthLens",sx,cy);
        ctx.fillText("Pro",sx+mMain.width,cy);
        ctx.restore();
      }

      if (morphT>=1 && !done) {
        done=true;
        cancelAnimationFrame(rafId);
        drawFinalLogo();
        setTimeout(revealControls, 100);
        return;
      }
    }

    rafId=requestAnimationFrame(render);
  }

  /* ── Theme-change listener ───────────────────────────── */
  document.addEventListener("depthlens-theme-changed", ()=>{
    resizeBg(); drawBg();
    if (done) drawFinalLogo();
  });

  /* ── Resize ─────────────────────────────────────────── */
  window.addEventListener("resize",()=>{
    resizeCanvas();
    resizeBg(); drawBg();
    if (done){ buildLogoPixels(); buildParticles(); drawFinalLogo(); }
  });

  /* ── Start ───────────────────────────────────────────── */
  requestAnimationFrame(render);

})();
