#!/usr/bin/env node
const { UNSUPPORTED_MESSAGES, targetKey } = require('../src/platform-targets');
function arg(name, fallback) { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i+1] : fallback; }
const platform = arg('platform', process.platform);
const arch = arg('arch', process.arch);
const key = targetKey(platform, arch);
console.error(UNSUPPORTED_MESSAGES[key] || `Unsupported DepthLens Pro target: ${platform}/${arch}.`);
process.exit(1);
