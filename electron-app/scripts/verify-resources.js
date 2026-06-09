#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const root = path.resolve(process.argv[2] || path.join(__dirname, "..", ".."));
const checks = [];
function exists(rel, predicate = fs.existsSync) {
  const full = path.join(root, rel);
  const ok = predicate(full);
  checks.push({ rel, full, ok });
  return ok;
}

exists("backend");
exists(path.join("backend", "app.py"));
exists("frontend");
exists(path.join("frontend", "index.html"));

const posixPython = [path.join("venv", "bin", "python3"), path.join("venv", "bin", "python")];
const winPython = [path.join("venv", "Scripts", "python.exe")];
const platformCandidates = process.platform === "win32" ? winPython : posixPython;
const otherCandidates = process.platform === "win32" ? posixPython : winPython;
const pythonOk = [...platformCandidates, ...otherCandidates].some((rel) => fs.existsSync(path.join(root, rel)));
checks.push({ rel: `${platformCandidates.join(" or ")} (platform Python)`, full: root, ok: pythonOk });

for (const check of checks) {
  console.log(`${check.ok ? "✓" : "✗"} ${check.rel}${check.ok ? "" : ` missing under ${check.full}`}`);
}

if (checks.some((check) => !check.ok)) {
  console.error(`\nDepthLens resources are incomplete under: ${root}`);
  console.error("Create the repo-root venv before packaging, or pass a packaged resources directory to this script.");
  process.exit(1);
}

console.log(`\nDepthLens resources verified under: ${root}`);
