#!/usr/bin/env node
const { spawnSync } = require('child_process');
function run(cmd, args) { console.log(`$ ${cmd} ${args.join(' ')}`); const r=spawnSync(cmd,args,{stdio:'inherit',shell:false}); if(r.status) process.exit(r.status); }
if (process.platform === 'darwin') spawnSync('osascript', ['-e', 'tell application "DepthLens Pro" to quit'], {stdio:'inherit'});
run('npm', ['run', 'kill:backend']);
run('npm', ['run', 'clean:dist']);
if (process.env.DEPTHLENS_CLEAN_INSTALL === '1') run('npm', ['run', 'clean:install']);
run('npm', ['run', 'build:mac']);
console.log('Open this Apple Silicon app path: electron-app/dist/mac-arm64/DepthLens Pro.app');
