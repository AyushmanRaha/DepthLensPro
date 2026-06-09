#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
for (const target of [path.join(root, 'dist'), path.join(root, 'dist', 'mac'), path.join(root, 'dist', 'mac-arm64')]) {
  if (fs.existsSync(target)) {
    console.log(`Removing packaged build output: ${target}`);
    fs.rmSync(target, { recursive: true, force: true });
  } else {
    console.log(`Not present: ${target}`);
  }
}
console.log('clean:dist complete; source files were not touched.');
