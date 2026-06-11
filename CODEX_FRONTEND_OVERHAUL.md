# DepthLens Pro — Complete Visual & Frontend Overhaul

## Prime Directive

**Do NOT change any existing features, backend code, API contracts, Python files, Electron
main/preload/security files, or test files.** Touch only:

- `frontend/style.css` (full replacement)
- `frontend/script.js` (animation/physics/resize helpers only — no feature logic changes)
- `frontend/welcome-anim.js` (typography/palette updates only)
- `frontend/index.html` (remove emoji/icon text nodes only; keep all IDs, classes, aria attributes,
  and structural HTML 100% intact)

If any file outside this list needs a character changed to satisfy a design rule, make a note in
`FRONTEND_OVERHAUL_NOTES.md` instead and leave the file untouched.

---

## 1. Typography System

Replace every font reference. Use these Google Fonts imports (place at top of `<head>` in
`index.html`, replacing the existing `<link>` tags for fonts):

```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link
  href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@600;700;800&display=swap"
  rel="stylesheet"
/>
```

### Font Role Mapping

| Role              | Old Font      | New Font             | CSS Variable          |
|-------------------|---------------|----------------------|-----------------------|
| Display / Headings| Rajdhani      | Syne                 | `--ff-display`        |
| Body / UI labels  | Exo 2         | Inter                | `--ff-body`           |
| Monospace / data  | JetBrains Mono| JetBrains Mono       | `--ff-mono`           |

### Font Size Scale (replace all hardcoded rem values)

```css
--text-xs:   0.6875rem;   /* 11px  — status sub, meta labels        */
--text-sm:   0.75rem;     /* 12px  — captions, mono data             */
--text-base: 0.875rem;    /* 14px  — body, UI descriptions           */
--text-md:   1rem;        /* 16px  — card titles, nav labels         */
--text-lg:   1.125rem;    /* 18px  — section headings                */
--text-xl:   1.375rem;    /* 22px  — logo, metric values             */
--text-2xl:  1.75rem;     /* 28px  — hero / welcome                  */

--lh-tight:  1.25;
--lh-normal: 1.5;
--lh-loose:  1.7;

--ls-tighter: -0.02em;
--ls-tight:  -0.01em;
--ls-normal:  0em;
--ls-wide:    0.04em;
--ls-wider:   0.08em;
--ls-widest:  0.14em;
```

Card titles: `font-family: var(--ff-display); font-size: var(--text-sm); font-weight: 700;
letter-spacing: var(--ls-widest); text-transform: uppercase;`

Nav buttons: `font-family: var(--ff-display); font-size: var(--text-sm); font-weight: 600;
letter-spacing: var(--ls-wider);`

Metric values: `font-family: var(--ff-mono); font-size: var(--text-xl); font-weight: 500;`

Body paragraphs: `font-family: var(--ff-body); font-size: var(--text-base);
line-height: var(--lh-loose);`

---

## 2. Colour Palettes

### 2a. Dark Theme (default, `[data-theme="dark"]` or `:root`)

```css
:root,
[data-theme="dark"] {
  /* Backgrounds — true deep navy, not black */
  --bg-base:        #080e1a;
  --bg-surface:     #0d1525;
  --bg-card:        #111d30;
  --bg-card-hover:  #162338;
  --bg-elevated:    #1a2b40;

  /* Borders */
  --border:         rgba(99, 179, 237, 0.08);
  --border-hi:      rgba(99, 179, 237, 0.20);
  --border-focus:   rgba(99, 179, 237, 0.50);
  --border-active:  rgba(99, 179, 237, 0.80);

  /* Accent — a cooler, more refined cyan-blue */
  --accent:         #63b3ed;
  --accent-bright:  #90cdf4;
  --accent-dim:     #2b6cb0;
  --accent-glow:    rgba(99, 179, 237, 0.12);
  --accent-glow-md: rgba(99, 179, 237, 0.20);

  /* Secondary accent — muted violet */
  --accent2:        #9f7aea;
  --accent2-glow:   rgba(159, 122, 234, 0.12);

  /* Danger accent — warm coral */
  --accent3:        #fc8181;
  --accent3-glow:   rgba(252, 129, 129, 0.10);

  /* Text hierarchy */
  --text:           #e2e8f0;
  --text-muted:     #a0b3c8;
  --text-dim:       #607080;
  --text-disabled:  #3a4a5a;

  /* Semantic */
  --success:        #68d391;
  --success-glow:   rgba(104, 211, 145, 0.12);
  --warning:        #f6c90e;
  --warning-glow:   rgba(246, 201, 14, 0.12);
  --error:          #fc8181;
  --error-glow:     rgba(252, 129, 129, 0.10);
  --danger:         #fc8181;

  /* Scrollbar */
  --scrollbar-thumb: rgba(99, 179, 237, 0.18);
  --scrollbar-thumb-hover: rgba(99, 179, 237, 0.36);
}
```

### 2b. Light Theme (`[data-theme="light"]`)

```css
[data-theme="light"] {
  /* Backgrounds — warm whites with slight blue-grey cast */
  --bg-base:        #f7f9fc;
  --bg-surface:     #ffffff;
  --bg-card:        #ffffff;
  --bg-card-hover:  #f0f4f9;
  --bg-elevated:    #e8eef6;

  /* Borders */
  --border:         rgba(45, 90, 140, 0.10);
  --border-hi:      rgba(45, 90, 140, 0.22);
  --border-focus:   rgba(37, 99, 172, 0.50);
  --border-active:  rgba(37, 99, 172, 0.80);

  /* Accent — deeper, accessible blue */
  --accent:         #2563eb;
  --accent-bright:  #1d4ed8;
  --accent-dim:     #93c5fd;
  --accent-glow:    rgba(37, 99, 235, 0.08);
  --accent-glow-md: rgba(37, 99, 235, 0.15);

  /* Secondary accent */
  --accent2:        #7c3aed;
  --accent2-glow:   rgba(124, 58, 237, 0.08);

  /* Danger accent */
  --accent3:        #dc2626;
  --accent3-glow:   rgba(220, 38, 38, 0.08);

  /* Text */
  --text:           #1a202c;
  --text-muted:     #4a5568;
  --text-dim:       #718096;
  --text-disabled:  #a0aec0;

  /* Semantic */
  --success:        #276749;
  --success-glow:   rgba(39, 103, 73, 0.08);
  --warning:        #92400e;
  --warning-glow:   rgba(146, 64, 14, 0.08);
  --error:          #c53030;
  --error-glow:     rgba(197, 48, 48, 0.08);
  --danger:         #c53030;

  /* Scrollbar */
  --scrollbar-thumb: rgba(45, 90, 140, 0.18);
  --scrollbar-thumb-hover: rgba(45, 90, 140, 0.36);
}
```

---

## 3. Spacing & Layout Tokens

```css
:root {
  /* Spacing scale — 4px base unit */
  --space-1:  0.25rem;   /* 4px  */
  --space-2:  0.5rem;    /* 8px  */
  --space-3:  0.75rem;   /* 12px */
  --space-4:  1rem;      /* 16px */
  --space-5:  1.25rem;   /* 20px */
  --space-6:  1.5rem;    /* 24px */
  --space-8:  2rem;      /* 32px */
  --space-10: 2.5rem;    /* 40px */
  --space-12: 3rem;      /* 48px */

  /* Border radii */
  --r-sm:     4px;
  --r:        8px;
  --r-md:     10px;
  --r-lg:     14px;
  --r-xl:     20px;
  --r-full:   9999px;

  /* Shadows — layered, physically accurate */
  --shadow-sm:   0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  --shadow:      0 4px 12px rgba(0,0,0,0.12), 0 2px 6px rgba(0,0,0,0.08);
  --shadow-md:   0 8px 24px rgba(0,0,0,0.16), 0 4px 10px rgba(0,0,0,0.10);
  --shadow-lg:   0 16px 48px rgba(0,0,0,0.22), 0 8px 18px rgba(0,0,0,0.14);
  --shadow-xl:   0 28px 72px rgba(0,0,0,0.30), 0 12px 28px rgba(0,0,0,0.18);
  --card-shadow: 0 1px 4px rgba(0,0,0,0.10), 0 2px 10px rgba(0,0,0,0.07),
                 inset 0 1px 0 rgba(255,255,255,0.035);

  /* Transitions */
  --t-fast:   0.12s cubic-bezier(0.4, 0, 0.2, 1);
  --t:        0.18s cubic-bezier(0.4, 0, 0.2, 1);
  --t-slow:   0.28s cubic-bezier(0.4, 0, 0.2, 1);
  --t-spring: 0.32s cubic-bezier(0.34, 1.56, 0.64, 1);
  --t-enter:  0.22s cubic-bezier(0.0, 0.0, 0.2, 1);
  --t-exit:   0.16s cubic-bezier(0.4, 0.0, 1, 1);

  /* Z-index scale */
  --z-base:    1;
  --z-card:    2;
  --z-sticky:  10;
  --z-header:  100;
  --z-overlay: 200;
  --z-modal:   300;
  --z-toast:   400;
}
```

**Homogeneous spacing rules (enforce in every component):**

- Card padding: `var(--space-5)` (20px) on all sides, consistently.
- Gap between stacked cards in a column: `var(--space-4)` (16px).
- Gap in grid layouts (workspace, metrics, gallery): `var(--space-4)`.
- Card title margin-bottom: `var(--space-4)`.
- Form control groups (label + input): `var(--space-2)` gap.
- Button row gaps: `var(--space-3)`.
- Section padding top/bottom within panels: `var(--space-5)`.
- All icons/decorative elements removed from text — spacing must not rely on character width.

---

## 4. Icon / Emoji Removal

In `index.html`, remove **all** decorative text-based icons except the spinning SVG logo next to
"DepthLensPro" in the header. Specifically:

- Remove every `<span class="card-icon">` and its text child (the `▣`, `◉`, etc.). The `<h2>`
  card titles must stand alone without any icon prefix.
- Remove `<span class="btn-icon">` elements from all buttons. Button text only.
- Remove every status dot emoji, arrow, or symbol from any `<span>`, `<p>`, or text node that is
  purely decorative.
- Keep: the SVG in `.logo-icon` (the spinning rings SVG beside "DepthLensPro").
- Keep: functional SVG icons embedded in buttons if they convey meaning (e.g. the drop-zone upload
  SVG illustration — keep it, it's illustrative not decorative).
- Keep: the theme toggle SVG icons (sun/moon in `.theme-icon`).
- Keep: all `aria-label` and `aria-hidden` attributes untouched.
- In CSS, remove all `content:` pseudo-element rules that output decorative symbols.

After removal, update the `.card-title` rule to not rely on `.card-icon` spacing:

```css
.card-title {
  font-family: var(--ff-display);
  font-size: var(--text-sm);
  font-weight: 700;
  letter-spacing: var(--ls-widest);
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: var(--space-4);
  display: flex;
  align-items: center;
  gap: 0;   /* no icon gap needed */
}
```

---

## 5. Component-by-Component Design Rules

### 5a. Header (`.site-header`)

```css
.site-header {
  height: 56px;
  background: color-mix(in srgb, var(--bg-base) 85%, transparent);
  backdrop-filter: blur(20px) saturate(1.4);
  -webkit-backdrop-filter: blur(20px) saturate(1.4);
  border-bottom: 1px solid var(--border);
  box-shadow: 0 1px 0 var(--border);
  transition: background var(--t-slow), border-color var(--t-slow);
}
.header-inner {
  max-width: 1440px;
  margin: 0 auto;
  padding: 0 var(--space-6);
  height: 100%;
  display: flex;
  align-items: center;
  gap: var(--space-4);
}
.logo-main {
  font-family: var(--ff-display);
  font-size: var(--text-lg);
  font-weight: 800;
  letter-spacing: var(--ls-tight);
}
.logo-main em {
  font-style: normal;
  color: var(--accent);
}
.logo-sub {
  font-family: var(--ff-mono);
  font-size: var(--text-xs);
  color: var(--text-dim);
  letter-spacing: var(--ls-wider);
  text-transform: uppercase;
}
.nav-btn {
  font-family: var(--ff-display);
  font-size: var(--text-sm);
  font-weight: 600;
  letter-spacing: var(--ls-wider);
  padding: var(--space-2) var(--space-4);
  border-radius: var(--r);
  border: 1px solid transparent;
  color: var(--text-dim);
  background: transparent;
  transition: color var(--t), background var(--t), border-color var(--t);
  position: relative;
}
.nav-btn::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 50%; right: 50%;
  height: 2px;
  background: var(--accent);
  border-radius: var(--r-full);
  transition: left var(--t-spring), right var(--t-spring);
}
.nav-btn.active::after { left: var(--space-4); right: var(--space-4); }
.nav-btn:hover { color: var(--text); background: var(--accent-glow); }
.nav-btn.active {
  color: var(--accent);
  background: var(--accent-glow);
}
```

### 5b. Cards

```css
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: var(--space-5);
  box-shadow: var(--card-shadow);
  transition: border-color var(--t), box-shadow var(--t), background var(--t-slow);
  /* Physics: subtle lift on hover */
  transform: translateY(0);
  transition: border-color var(--t), box-shadow var(--t), transform var(--t-spring),
              background var(--t-slow);
}
.card:hover {
  border-color: var(--border-hi);
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}
```

### 5c. Buttons

All buttons: uniform height, no icon spans, consistent padding.

```css
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  font-family: var(--ff-display);
  font-size: var(--text-sm);
  font-weight: 600;
  letter-spacing: var(--ls-wide);
  height: 36px;
  padding: 0 var(--space-5);
  border-radius: var(--r);
  border: 1px solid transparent;
  white-space: nowrap;
  cursor: pointer;
  transition: background var(--t), color var(--t), border-color var(--t),
              box-shadow var(--t), transform var(--t-spring);
  transform: translateY(0) scale(1);
  position: relative;
  overflow: hidden;
}
.btn:not(:disabled):active {
  transform: translateY(1px) scale(0.98);
  transition-duration: 0.08s;
}
.btn:disabled {
  opacity: 0.36;
  cursor: not-allowed;
  pointer-events: none;
}
/* Primary */
.btn-primary {
  background: var(--accent);
  color: var(--bg-base);
  border-color: var(--accent);
  font-weight: 700;
}
.btn-primary:not(:disabled):hover {
  background: var(--accent-bright);
  box-shadow: 0 0 0 3px var(--accent-glow-md);
  transform: translateY(-1px) scale(1.01);
}
/* Outline */
.btn-outline {
  background: transparent;
  color: var(--accent);
  border-color: var(--border-hi);
}
.btn-outline:not(:disabled):hover {
  background: var(--accent-glow);
  border-color: var(--accent);
}
/* Ghost */
.btn-ghost {
  background: transparent;
  color: var(--text-muted);
  border-color: var(--border);
}
.btn-ghost:not(:disabled):hover {
  color: var(--text);
  background: var(--bg-elevated);
  border-color: var(--border-hi);
}
/* Danger */
.btn-danger {
  background: var(--accent3-glow);
  color: var(--error);
  border-color: rgba(252, 129, 129, 0.22);
}
.btn-danger:not(:disabled):hover {
  background: rgba(252, 129, 129, 0.18);
  box-shadow: 0 0 0 3px var(--accent3-glow);
}
/* Size modifiers */
.btn-sm { height: 30px; padding: 0 var(--space-4); font-size: var(--text-xs); }
.w-full { width: 100%; }
```

**Remove `.btn-icon` span from all button HTML in `index.html`.** Buttons are text-only.

### 5d. Form Controls (select inputs, text inputs)

```css
.select-input,
input[type="text"].select-input,
.experiment-name {
  height: 34px;
  padding: 0 var(--space-3);
  background: var(--bg-surface);
  border: 1px solid var(--border-hi);
  border-radius: var(--r);
  color: var(--text);
  font-family: var(--ff-mono);
  font-size: var(--text-sm);
  outline: none;
  transition: border-color var(--t), box-shadow var(--t);
  appearance: none;
  -webkit-appearance: none;
}
.select-input:focus,
.experiment-name:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-glow);
}
/* Custom dropdown arrow for selects */
select.select-input {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23607080'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;
  padding-right: 28px;
  cursor: pointer;
}
```

### 5e. Device Selector

Replace the `.device-opt-icon` glyph spans with a thin left-border accent instead of a character:

```css
.device-opt-inner {
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--r);
  background: var(--bg-surface);
  display: flex;
  align-items: center;
  gap: var(--space-3);
  transition: border-color var(--t), background var(--t), box-shadow var(--t),
              transform var(--t-spring);
}
.device-opt:hover .device-opt-inner {
  border-color: var(--border-hi);
  background: var(--bg-elevated);
  transform: translateX(2px);
}
.device-opt input:checked ~ .device-opt-inner {
  border-color: var(--accent);
  background: var(--accent-glow);
  box-shadow: 0 0 0 2px var(--accent-glow-md);
}
.device-opt-icon { display: none; } /* Remove glyph icons entirely */
```

Remove the `<span class="device-opt-icon">` text nodes from HTML.

### 5f. Model Cards

```css
.model-card-inner {
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--r);
  background: var(--bg-surface);
  transition: border-color var(--t), background var(--t), transform var(--t-spring),
              box-shadow var(--t);
}
.model-card:hover .model-card-inner {
  border-color: var(--border-hi);
  background: var(--bg-elevated);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}
.model-card input:checked ~ .model-card-inner {
  border-color: var(--accent);
  background: var(--accent-glow);
  box-shadow: 0 0 0 2px var(--accent-glow-md), var(--shadow-sm);
}
```

### 5g. Drop Zone

```css
.drop-zone {
  border: 1.5px dashed var(--border-hi);
  border-radius: var(--r-lg);
  background: var(--bg-surface);
  transition: border-color var(--t), background var(--t), box-shadow var(--t),
              transform var(--t-spring);
}
.drop-zone:hover,
.drop-zone.drag-over {
  border-color: var(--accent);
  background: var(--accent-glow);
  transform: scale(1.005);
}
.drop-zone.drag-over {
  box-shadow: 0 0 0 3px var(--accent-glow-md), var(--shadow-md);
}
```

### 5h. Gallery

```css
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: var(--space-4);
}
.gallery-item {
  border-radius: var(--r-md);
  overflow: hidden;
  border: 1px solid var(--border);
  background: var(--bg-card);
  transition: border-color var(--t), transform var(--t-spring), box-shadow var(--t);
  transform: translateY(0);
  cursor: pointer;
}
.gallery-item:hover {
  border-color: var(--border-hi);
  transform: translateY(-3px);
  box-shadow: var(--shadow-md);
}
```

### 5i. Metrics Grid

```css
.metrics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-3);
  margin-bottom: var(--space-4);
}
.metric-cell {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: var(--space-3) var(--space-4);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  transition: border-color var(--t), background var(--t);
}
.metric-cell:hover { border-color: var(--border-hi); }
.metric-value {
  font-family: var(--ff-mono);
  font-size: var(--text-xl);
  font-weight: 500;
  color: var(--accent);
  line-height: 1;
}
.metric-label {
  font-family: var(--ff-display);
  font-size: var(--text-xs);
  font-weight: 600;
  letter-spacing: var(--ls-widest);
  text-transform: uppercase;
  color: var(--text-muted);
}
.metric-desc {
  font-family: var(--ff-mono);
  font-size: var(--text-xs);
  color: var(--text-dim);
  line-height: var(--lh-normal);
}
```

### 5j. Progress Bar

```css
.progress-bar-fill {
  background: linear-gradient(90deg, var(--accent-dim), var(--accent), var(--accent-bright));
  background-size: 200% 100%;
  animation: progressShimmer 1.8s ease infinite;
  box-shadow: none; /* Remove old glow */
}
@keyframes progressShimmer {
  0%   { background-position: 200% center; }
  100% { background-position: -200% center; }
}
```

Remove the `box-shadow: 0 0 8px var(--accent)` glow from `.progress-bar-fill`.

### 5k. File Queue Items

```css
.file-item {
  padding: var(--space-2) var(--space-3);
  gap: var(--space-3);
  border: 1px solid var(--border);
  border-radius: var(--r);
  background: var(--bg-surface);
  transition: border-color var(--t), background var(--t);
  animation: fileItemEnter 0.18s var(--t-enter) both;
}
@keyframes fileItemEnter {
  from { opacity: 0; transform: translateX(-8px); }
  to   { opacity: 1; transform: translateX(0); }
}
.file-status.running { animation: pulseStatus 1.1s ease infinite; }
@keyframes pulseStatus {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.45; }
}
```

### 5l. Toasts

```css
.toast {
  padding: var(--space-3) var(--space-4);
  gap: var(--space-3);
  background: var(--bg-elevated);
  border: 1px solid var(--border-hi);
  border-radius: var(--r);
  font-family: var(--ff-mono);
  font-size: var(--text-sm);
  box-shadow: var(--shadow-lg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  max-width: 340px;
  animation: toastEnter 0.22s var(--t-enter) both;
}
@keyframes toastEnter {
  from { opacity: 0; transform: translateY(8px) scale(0.96); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes toastExit {
  from { opacity: 1; transform: translateX(0); }
  to   { opacity: 0; transform: translateX(16px); }
}
.toast-dot { display: none; } /* Remove dot, use left border instead */
.toast { border-left: 3px solid var(--accent); }
.toast.success { border-left-color: var(--success); }
.toast.error   { border-left-color: var(--error);   }
.toast.warning { border-left-color: var(--warning);  }
```

### 5m. Lightbox

```css
.lightbox {
  background: var(--bg-card);
  border: 1px solid var(--border-hi);
  border-radius: var(--r-xl);
  box-shadow: var(--shadow-xl);
  /* Remove the ::before gradient line — clean top border instead */
}
.lightbox::before { display: none; }
.lightbox-backdrop {
  background: rgba(4, 8, 16, 0.80);
  backdrop-filter: blur(20px) saturate(1.2);
  -webkit-backdrop-filter: blur(20px) saturate(1.2);
}
.lightbox {
  transform: translateY(24px) scale(0.97);
  opacity: 0;
  transition: transform 0.40s cubic-bezier(0.22, 1, 0.36, 1),
              opacity 0.30s ease;
}
.lightbox-backdrop.is-open .lightbox {
  transform: translateY(0) scale(1);
  opacity: 1;
}
```

Remove all `rgba(0,200,255,...)` references inside the lightbox. Use CSS variables only.

### 5n. Status Indicator

Remove glyph from `.status-dot`, keep it as a pure CSS dot:

```css
.status-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--text-dim);
  flex-shrink: 0;
  transition: background var(--t), box-shadow var(--t);
}
.status-dot.online {
  background: var(--success);
  box-shadow: 0 0 0 3px var(--success-glow);
  animation: statusPulse 2.8s ease infinite;
}
@keyframes statusPulse {
  0%, 100% { box-shadow: 0 0 0 3px var(--success-glow); }
  50%       { box-shadow: 0 0 0 5px transparent; }
}
.status-dot.offline     { background: var(--error); box-shadow: none; }
.status-dot.connecting  { background: var(--warning); animation: statusPulse 0.9s ease infinite; }
```

---

## 6. Background Canvas (Workspace)

Replace the current particle/grid background in the IIFE at the bottom of `script.js` with a
cleaner, higher-performance version. Find the `(function bgCanvas()` block and replace it:

```javascript
(function bgCanvas() {
  const cv = document.getElementById('bgCanvas');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let W = 0, H = 0, pts = [], raf = 0;
  const N = 48;

  function isDark() {
    return document.documentElement.getAttribute('data-theme') !== 'light';
  }

  function mkP() {
    return {
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.22, vy: (Math.random() - 0.5) * 0.22,
      r: Math.random() * 1.1 + 0.4,
    };
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth; H = window.innerHeight;
    cv.width = Math.floor(W * dpr); cv.height = Math.floor(H * dpr);
    cv.style.width = `${W}px`; cv.style.height = `${H}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const dark = isDark();
    const gridColor = dark ? 'rgba(99,179,237,0.04)' : 'rgba(37,99,235,0.04)';
    const nodeColor = dark ? 'rgba(99,179,237,0.22)' : 'rgba(37,99,235,0.18)';
    const edgeColor = dark ? 'rgba(99,179,237,0.07)' : 'rgba(37,99,235,0.06)';

    // Grid
    ctx.strokeStyle = gridColor; ctx.lineWidth = 0.75;
    for (let x = 0; x < W; x += 72) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += 72) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Nodes + edges
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      p.x = (p.x + p.vx + W) % W; p.y = (p.y + p.vy + H) % H;
      for (let j = i + 1; j < pts.length; j++) {
        const q = pts[j], d = Math.hypot(p.x - q.x, p.y - q.y);
        if (d < 120) {
          ctx.strokeStyle = edgeColor;
          ctx.globalAlpha = (1 - d / 120) * 0.7;
          ctx.lineWidth = 0.5; ctx.beginPath();
          ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = nodeColor; ctx.fill();
    }
    if (!reduce) raf = requestAnimationFrame(draw);
  }

  function init() {
    resize();
    pts = Array.from({ length: N }, mkP);
    raf = requestAnimationFrame(draw);
  }

  window.addEventListener('resize', () => {
    cancelAnimationFrame(raf);
    resize();
    if (!reduce) raf = requestAnimationFrame(draw);
  });
  document.addEventListener('depthlens-theme-changed', () => {
    // Redraw once on theme change even in reduced motion
    if (reduce) { ctx.clearRect(0, 0, W, H); draw(); cancelAnimationFrame(raf); }
  });
  init();
})();
```

---

## 7. Dynamic Resize System

Add this ResizeObserver block at the **end of `init()`** in `script.js`, before the closing brace:

```javascript
// Responsive layout: collapse sidebar to top on narrow viewports
const workspaceGrid = document.querySelector('.workspace-grid');
if (workspaceGrid && typeof ResizeObserver !== 'undefined') {
  const ro = new ResizeObserver(entries => {
    for (const entry of entries) {
      const w = entry.contentRect.width;
      workspaceGrid.classList.toggle('layout-stacked', w < 900);
    }
  });
  ro.observe(workspaceGrid);
}

// Smooth panel height transitions
document.querySelectorAll('.panel').forEach(panel => {
  panel.style.setProperty('--panel-height', panel.scrollHeight + 'px');
});
```

In CSS, add:

```css
.workspace-grid { transition: grid-template-columns var(--t-slow); }
.workspace-grid.layout-stacked { grid-template-columns: 1fr !important; }

@media (max-width: 960px) {
  .workspace-grid { grid-template-columns: 1fr; }
  .metrics-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 600px) {
  .metrics-grid { grid-template-columns: 1fr; }
  .colormap-grid { grid-template-columns: repeat(4, 1fr); }
  .gallery { grid-template-columns: 1fr; }
  .header-inner { padding: 0 var(--space-4); gap: var(--space-3); }
}

/* Fluid font scaling */
@media (max-width: 480px) {
  :root {
    --text-xs:   0.625rem;
    --text-sm:   0.6875rem;
    --text-base: 0.8125rem;
    --text-lg:   1rem;
    --text-xl:   1.25rem;
  }
}
```

---

## 8. Physics / Interaction Animations

Add these CSS rules (they require no JS changes):

```css
/* Ripple on button press */
.btn::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: currentColor;
  opacity: 0;
  transform: scale(0);
  transition: transform 0.4s ease, opacity 0.4s ease;
  pointer-events: none;
}
.btn:active::after {
  transform: scale(2.5);
  opacity: 0.06;
  transition-duration: 0s;
}

/* Panel fade-slide entrance */
.panel.active,
.panel:not([hidden]) {
  animation: panelEnter 0.24s var(--t-enter) both;
}
@keyframes panelEnter {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Gallery item staggered entrance */
.gallery-item:nth-child(1)  { animation-delay: 0ms; }
.gallery-item:nth-child(2)  { animation-delay: 30ms; }
.gallery-item:nth-child(3)  { animation-delay: 60ms; }
.gallery-item:nth-child(4)  { animation-delay: 90ms; }
.gallery-item:nth-child(n+5){ animation-delay: 120ms; }

/* Card hover lift — already in card rules, ensure no conflict */
/* File item entrance — already defined in 5k */

/* Skeleton loading shimmer for metric cells while backend initialises */
.metric-cell.is-loading .metric-value {
  background: linear-gradient(90deg, var(--border) 25%, var(--border-hi) 50%, var(--border) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s ease infinite;
  border-radius: var(--r-sm);
  color: transparent;
  user-select: none;
}
@keyframes shimmer {
  0%   { background-position: 200% center; }
  100% { background-position: -200% center; }
}

/* Accordion open/close spring */
.metric-group-body {
  transition: max-height 0.30s cubic-bezier(0.4, 0, 0.2, 1),
              opacity 0.22s ease;
  opacity: 0;
}
.metric-group.open .metric-group-body {
  opacity: 1;
}

/* Scroll-into-view for result cards */
.gallery-item {
  animation: galleryItemIn 0.22s var(--t-enter) both;
}
@keyframes galleryItemIn {
  from { opacity: 0; transform: translateY(10px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
```

---

## 9. Compare Panel Refinements

```css
.compare-card {
  border-radius: var(--r-md);
  border: 1px solid var(--border);
  background: var(--bg-card);
  overflow: hidden;
  transition: border-color var(--t), transform var(--t-spring), box-shadow var(--t);
}
.compare-card:hover {
  border-color: var(--border-hi);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}
.compare-card-header {
  padding: var(--space-3) var(--space-4);
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border);
  font-family: var(--ff-display);
  font-size: var(--text-base);
  font-weight: 600;
  letter-spacing: var(--ls-tight);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.latency-badge {
  font-family: var(--ff-mono);
  font-size: var(--text-xs);
  color: var(--success);
  background: var(--success-glow);
  padding: 2px 7px;
  border-radius: var(--r-sm);
  border: 1px solid rgba(104, 211, 145, 0.18);
}
```

---

## 10. Performance Panel

```css
.benchmark-grid .metric-value {
  font-size: var(--text-lg);
}
#benchmarkChart {
  border-radius: var(--r);
  overflow: hidden;
}
```

---

## 11. Welcome Screen Palette Update

In `welcome-anim.js`, update the `getPalette()` function to use the new colours:

**Dark palette** (`isDark() === true`):

```javascript
{
  bgTop: '#080e1a', bgBottom: '#0d1525',
  grid: 'rgba(99, 179, 237, 0.06)',
  gridMajor: 'rgba(99, 179, 237, 0.12)',
  point: 'rgba(99, 179, 237, 0.28)', pointDim: 'rgba(99, 179, 237, 0.10)',
  scan: 'rgba(99, 179, 237, 0.90)', scanSoft: 'rgba(99, 179, 237, 0.14)',
  contour: 'rgba(99, 179, 237, 0.48)', contourDim: 'rgba(99, 179, 237, 0.16)',
  textTop: '#f0f6ff', textMid: '#e2e8f0', textBottom: '#a0b3c8',
  proTop: '#90cdf4', proMid: '#63b3ed', proBottom: '#2b6cb0',
  shadow: 'rgba(0, 14, 28, 0.45)',
  sweep: 'rgba(255, 255, 255, 0.22)',
}
```

**Light palette** (`isDark() === false`):

```javascript
{
  bgTop: '#f7f9fc', bgBottom: '#eef3fa',
  grid: 'rgba(37, 99, 235, 0.06)',
  gridMajor: 'rgba(37, 99, 235, 0.11)',
  point: 'rgba(37, 99, 235, 0.24)', pointDim: 'rgba(37, 99, 235, 0.09)',
  scan: 'rgba(37, 99, 235, 0.82)', scanSoft: 'rgba(37, 99, 235, 0.11)',
  contour: 'rgba(37, 99, 235, 0.40)', contourDim: 'rgba(37, 99, 235, 0.14)',
  textTop: '#1a202c', textMid: '#2d3748', textBottom: '#4a5568',
  proTop: '#2563eb', proMid: '#1d4ed8', proBottom: '#1e40af',
  shadow: 'rgba(0, 30, 80, 0.12)',
  sweep: 'rgba(255, 255, 255, 0.60)',
}
```

Also update font references in `welcome-anim.js` from `Rajdhani` to `Syne`:

```javascript
logoCtx.font = `800 ${fontSize}px Syne, Inter, sans-serif`;
```

---

## 12. Light Theme Specific Overrides

```css
[data-theme="light"] {
  --card-shadow: 0 1px 3px rgba(0,50,120,0.06), 0 2px 8px rgba(0,50,120,0.05),
                 inset 0 1px 0 rgba(255,255,255,0.8);
}
[data-theme="light"] .site-header {
  background: color-mix(in srgb, var(--bg-surface) 88%, transparent);
  border-bottom-color: var(--border-hi);
  box-shadow: 0 1px 0 var(--border);
}
[data-theme="light"] .lightbox-backdrop {
  background: rgba(200, 215, 235, 0.75);
}
[data-theme="light"] .lightbox {
  background: var(--bg-surface);
  border-color: var(--border-hi);
}
[data-theme="light"] .bar-fill {
  background: linear-gradient(90deg, var(--accent-dim), var(--accent));
}
[data-theme="light"] .progress-bar-fill {
  background: linear-gradient(90deg, var(--accent), var(--accent-bright));
  background-size: 200% 100%;
  animation: progressShimmer 1.8s ease infinite;
}
[data-theme="light"] #bgCanvas { opacity: 0.5; }
```

---

## 13. Scrollbar

```css
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--scrollbar-thumb);
  border-radius: var(--r-full);
  transition: background var(--t);
}
::-webkit-scrollbar-thumb:hover { background: var(--scrollbar-thumb-hover); }
```

---

## 14. Focus States

```css
:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
  border-radius: var(--r-sm);
}
button:focus-visible, a:focus-visible { border-radius: var(--r); }
```

---

## 15. Specific Elements to Clean Up in `index.html`

Do the following find-and-remove passes in `index.html`:

1. Remove all `<span class="card-icon">…</span>` nodes (the glyph characters `▣`, `◉`, etc.).
2. Remove all `<span class="btn-icon">…</span>` nodes inside `.btn` elements.
3. Remove the `<span class="drop-icon">` SVG block? **NO — keep the upload SVG illustration.**
4. Remove `<span class="device-opt-icon">…</span>` text nodes (the `◎`, `▦`, `⬢` etc.).
5. Remove `<span class="model-badge">SMALL</span>` glyph prefix decorators? **NO — keep model
   badges, they carry semantic meaning.** Just remove the star/bullet prefix characters within
   them if any exist.
6. Remove `<span class="arch-label">` preceding bullet glyphs? **NO — keep arch-labels.**
7. Remove any stray `◍`, `◎`, `▶`, `▣`, `◉`, `≋`, `◔`, `◬`, `⇲`, `≋` text nodes that appear
   outside of semantic elements.
8. In the `<code-block>` section in the about panel, these characters are intentional content —
   leave them alone.
9. Remove `<span class="toast-dot"></span>` from toast HTML in `script.js` (the `toast()` function
   — remove the `toast-dot` span from the template literal).

---

## 16. GT Controls and Experiment Panel

Apply consistent spacing tokens:

```css
.gt-controls {
  padding: var(--space-4);
  gap: var(--space-4);
  border-radius: var(--r);
  border: 1px solid var(--border);
  background: var(--bg-surface);
}
.gt-toggle { gap: var(--space-3); }
.gt-toggle-copy strong {
  font-family: var(--ff-display);
  font-size: var(--text-base);
  font-weight: 600;
  letter-spacing: var(--ls-tight);
}
.gt-toggle-copy span, .gt-help {
  font-family: var(--ff-mono);
  font-size: var(--text-xs);
  color: var(--text-dim);
}
.experiment-toolbar { gap: var(--space-3); margin: var(--space-4) 0; }
.experiment-card { border-radius: var(--r-md); }
.experiment-card-head {
  padding: var(--space-3) var(--space-4);
  font-family: var(--ff-display);
  font-size: var(--text-base);
  font-weight: 600;
}
```

---

## 17. About Panel Refinements

```css
.about-content h3 {
  font-family: var(--ff-display);
  font-size: var(--text-xs);
  font-weight: 700;
  letter-spacing: var(--ls-widest);
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: var(--space-2);
}
.about-content p {
  font-family: var(--ff-body);
  font-size: var(--text-base);
  color: var(--text-muted);
  line-height: var(--lh-loose);
}
.arch-cell p {
  font-family: var(--ff-body);
  font-size: var(--text-sm);
  color: var(--text-muted);
  line-height: var(--lh-normal);
}
.arch-label {
  font-family: var(--ff-display);
  font-size: var(--text-xs);
  font-weight: 700;
  letter-spacing: var(--ls-widest);
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: var(--space-2);
}
.about-table th {
  font-family: var(--ff-display);
  font-size: var(--text-xs);
  font-weight: 700;
  letter-spacing: var(--ls-widest);
  text-transform: uppercase;
  color: var(--text-dim);
  padding: var(--space-2) var(--space-3);
}
.about-table td {
  font-family: var(--ff-mono);
  font-size: var(--text-sm);
  padding: var(--space-2) var(--space-3);
}
.code-block {
  font-family: var(--ff-mono);
  font-size: var(--text-sm);
  background: var(--bg-base);
  border: 1px solid var(--border-hi);
  border-radius: var(--r);
  padding: var(--space-4) var(--space-5);
  line-height: var(--lh-loose);
}
.code-comment { color: var(--text-dim); }
```

---

## 18. Validation Checklist

After implementing all changes, verify:

- [ ] `python -m pytest backend/` still passes (no backend files changed).
- [ ] `black --check .` passes (no Python files changed).
- [ ] `ruff check .` passes.
- [ ] `mypy backend/` passes.
- [ ] `cd electron-app && npm run test` passes (security policy tests).
- [ ] `cd electron-app && npm run verify:resources` passes.
- [ ] Open the app: welcome screen animation displays with updated palette and Syne font.
- [ ] Dark and light theme toggle works, all colours update correctly with no flash.
- [ ] All five nav panels switch without layout breaking.
- [ ] Inference runs end to end from drop zone through gallery.
- [ ] Lightbox opens and closes with physics animation.
- [ ] Benchmark panel runs and chart renders.
- [ ] Experiment panel runs and exports JSON/CSV.
- [ ] Resizing window to 480px does not break any layout.
- [ ] No emoji or glyph icons remain in the UI except the spinning logo SVG.
- [ ] Every card, button, and form control uses tokens — no magic hex or rem values hardcoded.
- [ ] All ARIA labels, IDs, and data attributes from original HTML are intact.
- [ ] `electron-app/test-security-policy.js` still passes: `node electron-app/test-security-policy.js`.

---

## 19. Files to Create or Modify (Summary)

| File                        | Action          | Scope                                      |
|-----------------------------|-----------------|--------------------------------------------|
| `frontend/style.css`        | Full replacement| All CSS — design tokens, components, themes|
| `frontend/index.html`       | Minimal edits   | Remove icon/emoji text nodes only          |
| `frontend/script.js`        | Targeted edits  | bgCanvas IIFE, toast template, ResizeObserver block in `init()` |
| `frontend/welcome-anim.js`  | Targeted edits  | `getPalette()` values, font string in `drawLogo()` |
| `FRONTEND_OVERHAUL_NOTES.md`| Create new      | Log any edge cases found during implementation |

**Do NOT modify:** `backend/`, `electron-app/`, `mypy.ini`, `pyproject.toml`, `docker-compose.yml`,
`Dockerfile`, `AGENTS.md`, `CONTRIBUTING.md`, `README.md`, `.github/`, `.codex/`.
