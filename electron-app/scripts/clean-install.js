#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const targets = ['/Applications/DepthLens Pro.app', path.join(os.homedir(), 'Applications', 'DepthLens Pro.app')];
console.log('Known installed DepthLens Pro app bundles to remove:');
for (const target of targets) {
  if (fs.existsSync(target)) {
    console.log(`  Removing ${target}`);
    fs.rmSync(target, { recursive: true, force: true });
  } else {
    console.log(`  Not present: ${target}`);
  }
}
console.log('clean:install complete. User data, source repo, Python environments, and unrelated apps were not touched.');
