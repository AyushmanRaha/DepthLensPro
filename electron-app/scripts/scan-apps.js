#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const repoApp = path.resolve(__dirname, '..');
const roots = process.env.DEPTHLENS_APP_SCAN_ROOTS ? process.env.DEPTHLENS_APP_SCAN_ROOTS.split(path.delimiter) : [path.join(repoApp, 'dist'), '/Applications', path.join(os.homedir(), 'Applications'), path.join(os.homedir(), 'Downloads')];
const found = [];
function walk(dir, depth=0) {
  if (depth > 5 || !fs.existsSync(dir)) return;
  let entries=[]; try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name === 'DepthLens Pro.app') { found.push(p); continue; }
      if (e.name.endsWith('.app')) continue;
      walk(p, depth+1);
    }
  }
}
for (const r of roots) walk(r);
console.log('DepthLens Pro app bundle scan:');
if (!found.length) console.log('  No DepthLens Pro.app bundles found.');
for (const p of found) {
  const flags=[];
  if (p.includes(`${path.sep}dist${path.sep}mac${path.sep}DepthLens Pro.app`)) flags.push('STALE_UNSUPPORTED_DIST_MAC');
  if (/x64|universal|dist[\\/]mac[\\/]/i.test(p)) flags.push('UNSUPPORTED_X64_OR_UNIVERSAL_RISK');
  console.log(`  ${p}${flags.length ? '  ['+flags.join(', ')+']' : ''}`);
}
if (found.length > 1) {
  console.warn(`Duplicate Spotlight risk: ${found.length} app bundles named DepthLens Pro.app were found.`);
}
