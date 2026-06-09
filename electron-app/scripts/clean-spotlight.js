#!/usr/bin/env node
console.log(`Duplicate Spotlight icons usually mean multiple app bundles named "DepthLens Pro.app" exist.`);
console.log(`1. Run: npm run scan:apps`);
console.log(`2. Remove duplicates with: npm run clean:install and npm run clean:dist`);
console.log(`3. Rebuild/open only: electron-app/dist/mac-arm64/DepthLens Pro.app`);
console.log(`Spotlight can take time to update after deletion. Avoid aggressive system-wide reindexing unless you understand the macOS impact.`);
