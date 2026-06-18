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

assertReferencedScriptsExist();
assertNoDuplicateIds();
assertRequiredDomIdsExist();
assertExpectedFrontendModules();

console.log('frontend contract tests passed');
