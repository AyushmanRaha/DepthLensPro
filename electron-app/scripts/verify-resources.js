#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const DEFAULT_ROOT = path.resolve(path.join(__dirname, "..", ".."));
const VALID_ROOT_KINDS = new Set(["repo", "packaged"]);
const VALID_MODES = new Set(["basic", "native"]);
const VALID_ONNX_MODES = new Set(["off", "optional", "required", "require-all"]);

function parseArgs(argv = process.argv.slice(2)) {
  const options = {
    root: DEFAULT_ROOT,
    rootKind: "repo",
    mode: "basic",
    onnxMode: "optional",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--root-kind") options.rootKind = argv[++i] || options.rootKind;
    else if (arg === "--mode") options.mode = argv[++i] || options.mode;
    else if (arg === "--onnx") options.onnxMode = argv[++i] || options.onnxMode;
    else if (arg === "--root") options.root = path.resolve(argv[++i] || options.root);
    else if (arg === "--help" || arg === "-h") options.help = true;
    else if (!arg.startsWith("--")) options.root = path.resolve(arg);
    else throw new Error(`Unknown argument: ${arg}`);
  }

  if (!VALID_ROOT_KINDS.has(options.rootKind)) throw new Error(`Invalid --root-kind ${options.rootKind}; expected repo or packaged.`);
  if (!VALID_MODES.has(options.mode)) throw new Error(`Invalid --mode ${options.mode}; expected basic or native.`);
  if (!VALID_ONNX_MODES.has(options.onnxMode)) throw new Error(`Invalid --onnx ${options.onnxMode}; expected off, optional, required, or require-all.`);
  return options;
}

function usage() {
  return [
    "Usage: node scripts/verify-resources.js [--root-kind repo|packaged] [--mode basic|native] [--onnx off|optional|required|require-all] [resource-root]",
    "",
    "Examples:",
    "  node scripts/verify-resources.js --root-kind repo --mode native --onnx optional ..",
    "  node scripts/verify-resources.js --root-kind packaged --mode native --onnx optional 'dist/mac-arm64/DepthLens Pro.app/Contents/Resources'",
  ].join("\n");
}

function pathExists(targetPath) {
  try { return fs.existsSync(targetPath); } catch (_) { return false; }
}

function isFile(targetPath) {
  try { return fs.statSync(targetPath).isFile(); } catch (_) { return false; }
}

function isDirectory(targetPath) {
  try { return fs.statSync(targetPath).isDirectory(); } catch (_) { return false; }
}

function pythonCandidatesForPlatform(platform = process.platform) {
  const posixPython = [path.join("venv", "bin", "python3"), path.join("venv", "bin", "python")];
  const winPython = [path.join("venv", "Scripts", "python.exe"), path.join("venv", "Scripts", "python3.exe")];
  return platform === "win32" ? winPython : posixPython;
}

function verifyResourceRoot(options = {}) {
  const root = path.resolve(options.root || DEFAULT_ROOT);
  const rootKind = options.rootKind || "repo";
  const mode = options.mode || "basic";
  const onnxMode = options.onnxMode || "optional";
  const platform = options.platform || process.platform;
  const checks = [];
  const infos = [];

  function addCheck(rel, required = true, predicate = pathExists, label = rel) {
    const full = path.join(root, rel);
    const ok = predicate(full);
    checks.push({ rel, label, full, ok, required });
    return ok;
  }

  addCheck("backend", true, isDirectory);
  addCheck(path.join("backend", "app.py"), true, isFile);
  addCheck("frontend", true, isDirectory);
  addCheck(path.join("frontend", "index.html"), true, isFile);

  const platformPython = pythonCandidatesForPlatform(platform);
  const fallbackPython = platform === "win32" ? pythonCandidatesForPlatform("darwin") : pythonCandidatesForPlatform("win32");
  const pythonOk = [...platformPython, ...fallbackPython].some((rel) => isFile(path.join(root, rel)) || pathExists(path.join(root, rel)));
  checks.push({
    rel: platformPython.join(" or "),
    label: `${platformPython.join(" or ")} (platform Python)`,
    full: root,
    ok: pythonOk,
    required: true,
  });

  const requiresModelDirs = mode === "native" || onnxMode !== "off";
  addCheck("models", requiresModelDirs, isDirectory);
  addCheck(path.join("models", "onnx"), requiresModelDirs, isDirectory);

  const onnxFiles = [
    { rel: path.join("models", "onnx", "midas_small.onnx"), required: onnxMode === "required" || onnxMode === "require-all" },
    { rel: path.join("models", "onnx", "dpt_hybrid.onnx"), required: onnxMode === "require-all" },
    { rel: path.join("models", "onnx", "dpt_large.onnx"), required: onnxMode === "require-all" },
  ];
  for (const item of onnxFiles) {
    const full = path.join(root, item.rel);
    const exists = isFile(full);
    const size = exists ? fs.statSync(full).size : 0;
    infos.push({ rel: item.rel, exists, size, required: item.required });
    if (item.required) checks.push({ rel: item.rel, label: item.rel, full, ok: exists && size > 0, required: true });
  }

  const failed = checks.filter((check) => check.required && !check.ok);
  return { root, rootKind, mode, onnxMode, platform, checks, infos, failed, ok: failed.length === 0 };
}

function remediation(result) {
  if (result.rootKind === "packaged") {
    return [
      "Packaged app resources are incomplete. Rebuild with the supported root native build script:",
      "  macOS ARM:   scripts/build-native-macos.sh",
      "  Windows ARM: .\\scripts\\build-native-windows.ps1",
      "  Linux ARM:   scripts/build-native-linux.sh",
      "If this path is an installed app, replace the stale installed copy with the newly built artifact before launching it.",
    ].join("\n");
  }
  return [
    "Repo-root resources are incomplete. Run setup before packaging:",
    "  macOS/Linux: npm run setup",
    "  Windows:     npm run setup:win",
    "Then rebuild with the supported native build script. ONNX binaries remain optional unless --onnx required or --onnx require-all is used.",
  ].join("\n");
}

function formatResult(result) {
  const lines = [];
  lines.push(`DepthLens resource verification (${result.rootKind}, mode=${result.mode}, onnx=${result.onnxMode})`);
  lines.push(`Root: ${result.root}`);
  lines.push("");
  for (const check of result.checks) {
    const severity = check.required ? (check.ok ? "✓" : "✗") : (check.ok ? "✓" : "!");
    const optional = check.required ? "" : " (optional)";
    const missing = check.ok ? "" : ` missing under ${check.full}`;
    lines.push(`${severity} ${check.label}${missing}${optional}`);
  }
  lines.push("\nONNX resource summary:");
  for (const info of result.infos) {
    const required = info.required ? "required" : "optional";
    lines.push(`${info.exists ? "✓" : info.required ? "✗" : "!"} ${info.rel} ${info.exists ? `${info.size} bytes` : `missing (${required})`}`);
  }
  if (!result.ok) {
    lines.push("");
    lines.push(`DepthLens ${result.rootKind} resources are incomplete under: ${result.root}`);
    lines.push(remediation(result));
  } else {
    lines.push("");
    lines.push(`DepthLens resources verified under: ${result.root} (root-kind=${result.rootKind}, mode=${result.mode}, onnx=${result.onnxMode})`);
  }
  return lines.join("\n");
}

function main() {
  let options;
  try {
    options = parseArgs();
  } catch (err) {
    console.error(err.message);
    console.error(usage());
    process.exit(2);
  }
  if (options.help) {
    console.log(usage());
    return;
  }
  const result = verifyResourceRoot(options);
  const output = formatResult(result);
  if (result.ok) console.log(output);
  else {
    console.error(output);
    process.exit(1);
  }
}

if (require.main === module) main();

module.exports = {
  DEFAULT_ROOT,
  parseArgs,
  pythonCandidatesForPlatform,
  verifyResourceRoot,
  formatResult,
  remediation,
};
