#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
let root = path.resolve(path.join(__dirname, "..", ".."));
let mode = "basic";
let onnxMode = "optional"; // off | optional | required | require-all
for (let i = 0; i < args.length; i += 1) {
  const arg = args[i];
  if (arg === "--mode") mode = args[++i] || mode;
  else if (arg === "--onnx") onnxMode = args[++i] || onnxMode;
  else if (!arg.startsWith("--")) root = path.resolve(arg);
}

const checks = [];
const infos = [];
function check(rel, required = true, predicate = fs.existsSync) {
  const full = path.join(root, rel);
  const ok = predicate(full);
  checks.push({ rel, full, ok, required });
  return ok;
}
function fileSize(rel) {
  const full = path.join(root, rel);
  try { return fs.statSync(full).size; } catch { return null; }
}

check("backend");
check(path.join("backend", "app.py"));
check("frontend");
check(path.join("frontend", "index.html"));

const posixPython = [path.join("venv", "bin", "python3"), path.join("venv", "bin", "python")];
const winPython = [path.join("venv", "Scripts", "python.exe")];
const platformCandidates = process.platform === "win32" ? winPython : posixPython;
const otherCandidates = process.platform === "win32" ? posixPython : winPython;
const pythonOk = [...platformCandidates, ...otherCandidates].some((rel) => fs.existsSync(path.join(root, rel)));
checks.push({ rel: `${platformCandidates.join(" or ")} (platform Python)`, full: root, ok: pythonOk, required: true });

const expectsOnnxDir = onnxMode !== "off" || mode === "native";
check("models", mode === "native" || expectsOnnxDir);
check(path.join("models", "onnx"), onnxMode === "required" || onnxMode === "require-all");

const onnxFiles = [
  { rel: path.join("models", "onnx", "midas_small.onnx"), required: onnxMode === "required" || onnxMode === "require-all" },
  { rel: path.join("models", "onnx", "dpt_hybrid.onnx"), required: onnxMode === "require-all" },
  { rel: path.join("models", "onnx", "dpt_large.onnx"), required: onnxMode === "require-all" },
];
for (const item of onnxFiles) {
  const full = path.join(root, item.rel);
  const exists = fs.existsSync(full);
  const size = exists ? fs.statSync(full).size : 0;
  infos.push({ rel: item.rel, exists, size });
  if (item.required) checks.push({ rel: item.rel, full, ok: exists && size > 0, required: true });
}

for (const check of checks) {
  const severity = check.required ? (check.ok ? "✓" : "✗") : (check.ok ? "✓" : "!");
  console.log(`${severity} ${check.rel}${check.ok ? "" : ` missing under ${check.full}`}${check.required ? "" : " (optional)"}`);
}
console.log("\nONNX resource summary:");
for (const info of infos) console.log(`${info.exists ? "✓" : "!"} ${info.rel} ${info.exists ? `${info.size} bytes` : "missing (optional unless --onnx requires it)"}`);

const failed = checks.filter((check) => check.required && !check.ok);
if (failed.length) {
  console.error(`\nDepthLens resources are incomplete under: ${root}`);
  console.error("Run the root setup script before packaging. Use --onnx required for MiDaS Small or --onnx require-all for every ONNX model.");
  process.exit(1);
}
console.log(`\nDepthLens resources verified under: ${root} (mode=${mode}, onnx=${onnxMode})`);
