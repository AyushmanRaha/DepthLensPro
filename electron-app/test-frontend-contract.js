const assert = require('assert');
const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..');
const frontendRoot = path.join(repoRoot, 'frontend');
const indexPath = path.join(frontendRoot, 'index.html');
const indexHtml = fs.readFileSync(indexPath, 'utf8');

function localScriptSources() {
  return [...indexHtml.matchAll(/<script\b[^>]*\bsrc=["']([^"']+)["'][^>]*>/gi)]
    .map((match) => match[1])
    .filter((src) => !/^https?:\/\//i.test(src));
}

function htmlIds() {
  return [...indexHtml.matchAll(/\bid=["']([^"']+)["']/gi)].map((match) => match[1]);
}

function scriptRequiredIds() {
  const ids = new Set();
  for (const src of localScriptSources()) {
    const scriptPath = path.join(frontendRoot, src);
    const script = fs.readFileSync(scriptPath, 'utf8');
    for (const match of script.matchAll(/\$\(["']#([A-Za-z0-9_-]+)["']\)/g)) {
      ids.add(match[1]);
    }
  }
  return [...ids].sort();
}

function assertReferencedScriptsExist() {
  const sources = localScriptSources();
  assert(sources.length > 0, 'index.html must reference local frontend scripts');
  for (const src of sources) {
    const scriptPath = path.join(frontendRoot, src);
    assert(fs.existsSync(scriptPath), `index.html references missing script: ${src}`);
    assert(fs.statSync(scriptPath).isFile(), `index.html script is not a file: ${src}`);
  }
}

function assertNoDuplicateIds() {
  const seen = new Set();
  const duplicates = new Set();
  for (const id of htmlIds()) {
    if (seen.has(id)) duplicates.add(id);
    seen.add(id);
  }
  assert.deepStrictEqual([...duplicates].sort(), [], `duplicate DOM IDs found: ${[...duplicates].sort().join(', ')}`);
}

function assertRequiredDomIdsExist() {
  const available = new Set(htmlIds());
  const missing = scriptRequiredIds().filter((id) => !available.has(id));
  assert.deepStrictEqual(missing, [], `script-required DOM IDs missing from index.html: ${missing.join(', ')}`);
}

function assertExpectedFrontendModules() {
  const sources = localScriptSources();
  for (const required of [
    'js/settings.js',
    'js/api-client.js',
    'js/state.js',
    'js/dom.js',
    'js/uploads.js',
    'js/inference-ui.js',
    'js/compare.js',
    'js/webcam.js',
    'js/performance.js',
    'js/observability.js',
    'js/experiments.js',
    'js/reconstruction.js',
    'js/charts.js',
    'js/notifications.js',
    'js/compat.js',
  ]) {
    assert(sources.includes(required), `index.html must load ${required}`);
  }
}

function scriptIndex(src) {
  return localScriptSources().indexOf(src);
}

function readFrontendScript(src) {
  return fs.readFileSync(path.join(frontendRoot, src), 'utf8');
}

function assertScriptOrderingRespectsRendererDependencies() {
  const stateIndex = scriptIndex('js/state.js');
  const chartsIndex = scriptIndex('js/charts.js');
  assert(stateIndex >= 0 && chartsIndex >= 0, 'state.js and charts.js must both be loaded');
  assert(stateIndex < chartsIndex, 'state.js must load before charts.js because charts uses prefersReducedMotion');
}

function assertChartsReducedMotionReferenceIsSafe() {
  const charts = readFrontendScript('js/charts.js');
  if (!charts.includes('prefersReducedMotion')) return;
  assert(
    /function\s+safePrefersReducedMotion\s*\(/.test(charts) && /typeof\s+prefersReducedMotion\s*===\s*["']function["']/.test(charts),
    'charts.js must guard prefersReducedMotion behind a safe fallback'
  );
}

function assertChartGlobalsPrecedeEagerDecorativeStartup() {
  const charts = readFrontendScript('js/charts.js');
  const globals = [
    'let latencyChart',
    'let compareChart',
    'let benchmarkChart',
    'const COMPARE_METRICS',
  ];
  const eager = charts.indexOf('(function startBackgroundCanvas()');
  assert(eager > 0, 'charts.js must wrap eager background canvas startup');
  for (const marker of globals) {
    const pos = charts.indexOf(marker);
    assert(pos >= 0, `charts.js must initialize ${marker}`);
    assert(pos < eager, `${marker} must be initialized before eager decorative startup`);
  }
}

function assertBackgroundCanvasStartupIsGuarded() {
  const charts = readFrontendScript('js/charts.js');
  assert(/try\s*{\s*\n\s*bgCanvas\(\)/.test(charts), 'background canvas startup must be wrapped in try/catch');
  assert(/const\s+cv\s*=\s*el\?\.bgCanvas/.test(charts), 'background canvas must tolerate missing el/bgCanvas');
  assert(/if\s*\(\s*!cv\?\.getContext\s*\)\s*return/.test(charts), 'background canvas must tolerate missing getContext');
  assert(/if\s*\(\s*!ctx\s*\)\s*return/.test(charts), 'background canvas must tolerate missing 2D context');
}

function assertStartupCriticalPathIsFailureIsolated() {
  const compat = readFrontendScript('js/compat.js');
  const initPos = compat.indexOf('async function init()');
  assert(initPos >= 0, 'compat.js must define init()');
  const initBody = compat.slice(initPos);
  assert(initBody.includes('runOptionalInitializer("engine status panel", initEngineStatusPanel)'), 'engine status panel binding must be attempted independently');
  assert(initBody.indexOf('runOptionalInitializer("engine status panel", initEngineStatusPanel)') < initBody.indexOf('await resolveApiBaseUrl()'), 'engine status panel binding must happen before backend URL resolution');
  assert(initBody.includes('runOptionalInitializer("latency chart", initLatencyChart)'), 'chart setup must be optional');
  assert(initBody.includes('runOptionalInitializer("compare controls", initCompareControls)'), 'compare controls setup must be optional');
  assert(initBody.includes('await checkLive()') || initBody.includes('await checkLive({ quiet: true })'), 'startup must perform /live checks');
  assert(initBody.includes('await checkReadiness'), 'startup must perform /ready checks when live succeeds');
  assert(initBody.includes('await checkDiagnostics'), 'startup must perform diagnostics/device discovery when live succeeds');
  const apiClient = readFrontendScript('js/api-client.js');
  assert(apiClient.includes('statusMap = { offline: "offline", starting: "connecting", live: "online", diagnostics_pending: "online", ready: "online", degraded: "online"'), 'frontend must use explicit engine status states');
  assert(apiClient.includes('/ready?depth=quick'), 'startup readiness must use quick diagnostics');
  assert(apiClient.includes('/health?depth=quick'), 'startup health must use quick diagnostics');
  assert(initBody.indexOf('runOptionalInitializer("latency chart", initLatencyChart)') < initBody.indexOf('await resolveApiBaseUrl()'), 'optional chart setup must not wrap backend startup');
}

assertReferencedScriptsExist();
assertNoDuplicateIds();
assertRequiredDomIdsExist();
assertExpectedFrontendModules();
assertScriptOrderingRespectsRendererDependencies();
assertChartsReducedMotionReferenceIsSafe();
assertChartGlobalsPrecedeEagerDecorativeStartup();
assertBackgroundCanvasStartupIsGuarded();
assertStartupCriticalPathIsFailureIsolated();

console.log('frontend contract tests passed');

function assertNoRemoteRuntimeScripts() {
  const html = fs.readFileSync(path.join(__dirname, "..", "frontend", "index.html"), "utf8");
  assert(!html.includes("https://cdn.jsdelivr.net"), "frontend must not depend on jsdelivr runtime scripts");
  assert(html.includes("vendor/chart.umd.min.js"), "Chart.js must load from local vendor path");
}

assertNoRemoteRuntimeScripts();
