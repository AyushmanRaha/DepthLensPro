#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const { execFileSync } = require('child_process');
const PORT = Number(process.env.DEPTHLENS_BACKEND_PORT || 8765);
const HOST = '127.0.0.1';
const APP_NAME = 'DepthLens Pro';
function userDataCandidates() {
  const home = os.homedir();
  if (process.platform === 'darwin') return [path.join(home, 'Library', 'Application Support', APP_NAME)];
  if (process.platform === 'win32') return [path.join(process.env.APPDATA || path.join(home, 'AppData', 'Roaming'), APP_NAME)];
  return [path.join(process.env.XDG_CONFIG_HOME || path.join(home, '.config'), APP_NAME)];
}
function readFileSafe(p){ try{return fs.readFileSync(p,'utf8').trim()}catch{return ''} }
function pidFiles(){ return userDataCandidates().map(d=>({pidPath:path.join(d,'backend.pid'), metaPath:path.join(d,'backend.json')})); }
function execCap(cmd,args,timeout=3000){ try{return execFileSync(cmd,args,{encoding:'utf8',timeout,windowsHide:true}).trim()}catch(e){return (e.stdout||'').toString().trim()} }
function listeningPid(){ if(process.platform==='win32'){ const out=execCap('cmd.exe',['/c',`netstat -ano -p tcp | findstr :${PORT}`]); const line=out.split(/\r?\n/).find(l=>l.includes('LISTENING')); return line?Number(line.trim().split(/\s+/).pop()):null;} const out=execCap('lsof',['-nP',`-iTCP:${PORT}`,'-sTCP:LISTEN']); const line=out.split(/\r?\n/).find(l=>/\bLISTEN\b/.test(l)&&!l.startsWith('COMMAND')); return line?Number(line.trim().split(/\s+/)[1]):null; }
function cmdline(pid){ if(!pid)return ''; return process.platform==='win32'?execCap('powershell.exe',['-NoProfile','-Command',`(Get-CimInstance Win32_Process -Filter "ProcessId=${pid}").CommandLine`]):execCap('ps',['-p',String(pid),'-o','command=']); }
function safe(pid, cmd, stored){ const low=(cmd||'').toLowerCase(); if(stored && pid===stored) return true; return low.includes('depthlens pro')||low.includes('depthlenspro')||(low.includes('uvicorn')&&low.includes('backend.app:app')&&(low.includes('depthlens')||low.includes(`${path.sep}backend`)) ); }
function kill(pid, force=false){ if(process.platform==='win32') execCap('taskkill.exe',['/PID',String(pid),'/T',...(force?['/F']:[])]); else { try{process.kill(pid,force?'SIGKILL':'SIGTERM')}catch(e){console.warn(e.message)} } }
function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
function request(endpoint, timeout){ return new Promise((resolve)=>{ let connected=false, body=''; const req=http.get({host:HOST,port:PORT,path:endpoint,timeout},res=>{connected=true;res.setEncoding('utf8');res.on('data',c=>body+=c);res.on('end',()=>resolve({endpoint,status:res.statusCode,body,empty:!body,connected}));}); req.on('socket',s=>s.on('connect',()=>connected=true)); req.on('timeout',()=>{req.destroy(); resolve({endpoint,error: connected?'TCP connected but no HTTP response before timeout':'timeout',connected});}); req.on('error',e=>resolve({endpoint,error:e.code==='ECONNREFUSED'?'connection refused':e.message,connected})); }); }
async function killBackend(){ const stored = pidFiles().map(f=>Number(readFileSafe(f.pidPath))).find(Boolean)||null; const pid=listeningPid()||stored; console.log(`Detected PID: ${pid||'none'}`); if(pid) { const c=cmdline(pid); console.log(`Command line: ${c||'unknown'}`); if(!safe(pid,c,stored)){ console.log('Refusing to kill: process is not a safe DepthLens backend match.'); return 2;} kill(pid,false); await sleep(3000); if(listeningPid()===pid){ console.log(`Still listening; force killing safe DepthLens backend PID ${pid}`); kill(pid,true); await sleep(1000); } } const final=listeningPid(); console.log(`Final port ${PORT} state: ${final?`LISTENING by PID ${final}`:'free'}`); return final?1:0; }
async function smoke(){ let code=0; for(const [ep,to] of [['/live',3000],['/devices',5000],['/health',8000]]){ const r=await request(ep,to); console.log(`${ep}: ${JSON.stringify(r)}`); if(r.error||!r.status||r.status>=400||r.empty){ code=1; if(r.connected&&r.error) console.error(`${ep} failure: TCP connects but receives 0 bytes / no HTTP response.`); } } const pid=listeningPid(); console.log(`lsof/netstat PID: ${pid||'none'}`); for(const f of pidFiles()){ console.log(`${f.pidPath}: ${readFileSafe(f.pidPath)||'(missing)'}`); console.log(`${f.metaPath}: ${readFileSafe(f.metaPath)||'(missing)'}`); } return code; }
(async()=>{ const mode=process.argv[2]; const code = mode==='kill'? await killBackend() : mode==='smoke'? await smoke() : 64; process.exit(code); })();
