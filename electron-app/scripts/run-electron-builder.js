#!/usr/bin/env node
const { spawnSync } = require('child_process');
const { preflight } = require('./package-size-preflight');

function valueAfter(flag, fallback) {
  const index = process.argv.indexOf(flag);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}
function inferPlatform(argv) {
  if (argv.includes('--mac')) return 'darwin';
  if (argv.includes('--win')) return 'win32';
  if (argv.includes('--linux')) return 'linux';
  return process.platform;
}
function inferArch(argv) {
  for (const arch of ['arm64', 'x64']) if (argv.includes(`--${arch}`)) return arch;
  return process.arch;
}

const builderArgs = process.argv.slice(2);
const platform = inferPlatform(builderArgs);
const arch = inferArch(builderArgs);
const result = preflight({ platform, arch, repoRoot: valueAfter('--repo-root', require('path').resolve(__dirname, '..', '..')) });
require('./package-size-preflight').fmt;
console.log(`[DepthLens] Packaging preflight payload=${result.payloadHuman}${platform === 'darwin' ? ` dmg=${result.dmgSize}` : ''}`);
if (!result.ok) {
  console.error('Packaging preflight failed; electron-builder was not started.');
  if (platform === 'darwin' && result.dmgBytes <= result.payloadBytes) console.error('The DMG image is too small for the packaged app; increase computed dmg.size or reduce bundled resources.');
  process.exit(1);
}
if (platform === 'darwin') builderArgs.push(`-c.dmg.size=${result.dmgSize}`);
const child = spawnSync(process.platform === 'win32' ? 'electron-builder.cmd' : 'electron-builder', builderArgs, { stdio: 'inherit', shell: false });
if (child.status !== 0) {
  console.error('[DepthLens] electron-builder failed.');
  if (platform === 'darwin') console.error('If the log mentions /Volumes/... and "No space left on device", the temporary DMG image is too small for the packaged app; increase computed dmg.size or reduce bundled resources. This is different from the Mac startup disk being full.');
}
process.exit(child.status || 1);
