#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const DEFAULT_ROOT = path.resolve(path.join(__dirname, "..", ".."));
const VALID_ROOT_KINDS = new Set(["repo", "packaged"]);
const VALID_MODES = new Set(["basic", "native"]);
const VALID_ONNX_MODES = new Set(["off", "optional", "required", "require-all"]);
const VALID_CACHE_MODES = new Set(["off", "optional", "required"]);
const ONNX_MODELS = ["midas_small", "dpt_hybrid", "dpt_large"];

function parseOnnxModels(value = "midas_small") {
  if (!value) return ["midas_small"];
  const items = String(value).split(",").map((item) => item.trim()).filter(Boolean);
  if (items.length === 1 && items[0] === "all") return [...ONNX_MODELS];
  const invalid = items.filter((item) => !ONNX_MODELS.includes(item));
  if (invalid.length) throw new Error(`Invalid --models ${invalid.join(",")}; expected ${ONNX_MODELS.join(",")} or all.`);
  return items;
}

function parseArgs(argv = process.argv.slice(2)) {
  const options = {
    root: DEFAULT_ROOT,
    rootKind: "repo",
    mode: "basic",
    onnxMode: "optional",
    onnxModels: ["midas_small"],
    torchCache: "required",
    detectorCache: "optional",
    json: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--root-kind") options.rootKind = argv[++i] || options.rootKind;
    else if (arg === "--mode") options.mode = argv[++i] || options.mode;
    else if (arg === "--onnx") options.onnxMode = argv[++i] || options.onnxMode;
    else if (arg === "--models" || arg === "--onnx-models") options.onnxModels = parseOnnxModels(argv[++i] || "midas_small");
    else if (arg === "--torch-cache") options.torchCache = argv[++i] || options.torchCache;
    else if (arg === "--detector-cache") options.detectorCache = argv[++i] || options.detectorCache;
    else if (arg === "--root") options.root = path.resolve(argv[++i] || options.root);
    else if (arg === "--json") options.json = true;
    else if (arg === "--help" || arg === "-h") options.help = true;
    else if (!arg.startsWith("--")) options.root = path.resolve(arg);
    else throw new Error(`Unknown argument: ${arg}`);
  }

  if (!VALID_ROOT_KINDS.has(options.rootKind)) throw new Error(`Invalid --root-kind ${options.rootKind}; expected repo or packaged.`);
  if (!VALID_MODES.has(options.mode)) throw new Error(`Invalid --mode ${options.mode}; expected basic or native.`);
  if (!VALID_ONNX_MODES.has(options.onnxMode)) throw new Error(`Invalid --onnx ${options.onnxMode}; expected off, optional, required, or require-all.`);
  if (!VALID_CACHE_MODES.has(options.torchCache)) throw new Error(`Invalid --torch-cache ${options.torchCache}; expected off, optional, or required.`);
  if (!VALID_CACHE_MODES.has(options.detectorCache)) throw new Error(`Invalid --detector-cache ${options.detectorCache}; expected off, optional, or required.`);
  return options;
}

function usage() {
  return [
    "Usage: node scripts/verify-resources.js [--json] [--root-kind repo|packaged] [--mode basic|native] [--torch-cache off|optional|required] [--detector-cache off|optional|required] [--onnx off|optional|required|require-all] [--models midas_small|dpt_hybrid|dpt_large|all|comma-list] [resource-root]",
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


function torchCacheStatus(root) {
  const cacheRoot = path.join(root, "models", "torch-cache");
  const hub = path.join(cacheRoot, "hub");
  const checkpoints = path.join(hub, "checkpoints");
  const repoEntries = isDirectory(hub) ? fs.readdirSync(hub).filter((name) => name.toLowerCase().includes("midas")) : [];
  const validRepos = repoEntries.filter((name) => isFile(path.join(hub, name, "hubconf.py")) && (isDirectory(path.join(hub, name, "midas")) || isDirectory(path.join(hub, name, "MiDaS"))));
  const checkpointFiles = isDirectory(checkpoints) ? fs.readdirSync(checkpoints).filter((name) => /\.(pt|pth)$/i.test(name) && fs.statSync(path.join(checkpoints, name)).size > 0) : [];
  const hints = {
    midas_small: ["midas_v21_small", "midas_small"],
    dpt_hybrid: ["dpt_hybrid", "dpt_hybrid_384"],
    dpt_large: ["dpt_large", "dpt_large_384"],
  };
  const modelReady = Object.fromEntries(Object.entries(hints).map(([model, tokens]) => [model, checkpointFiles.some((name) => tokens.some((token) => name.toLowerCase().includes(token))) ]));
  const allModelsReady = Object.values(modelReady).every(Boolean);
  return { cacheRoot, repoCached: validRepos.length > 0, validRepos, checkpoints, checkpointFiles, modelReady, ok: isDirectory(cacheRoot) && validRepos.length > 0 && allModelsReady };
}

function detectorCacheStatus(root) {
  const checkpoints = path.join(root, "models", "torch-cache", "hub", "checkpoints");
  const files = isDirectory(checkpoints) ? fs.readdirSync(checkpoints).filter((name) => /\.(pt|pth)$/i.test(name) && name.toLowerCase().includes("fasterrcnn") && fs.statSync(path.join(checkpoints, name)).size > 0) : [];
  return { checkpoints, files, ok: files.length > 0 };
}

function pythonCandidatesForPlatform(platform = process.platform) {
  const posixPython = [path.join("venv", "bin", "python3"), path.join("venv", "bin", "python")];
  const winPython = [path.join("venv", "Scripts", "python.exe"), path.join("venv", "Scripts", "python3.exe")];
  return platform === "win32" ? winPython : posixPython;
}

function pythonCandidateChecks(root, platform = process.platform) {
  const platformPython = pythonCandidatesForPlatform(platform);
  const fallbackPython = platform === "win32" ? pythonCandidatesForPlatform("darwin") : pythonCandidatesForPlatform("win32");
  return [...platformPython, ...fallbackPython].map((rel) => {
    const full = path.join(root, rel);
    return { rel, full, exists: isFile(full) || pathExists(full), platformCandidate: platformPython.includes(rel) };
  });
}

function verifyResourceRoot(options = {}) {
  const root = path.resolve(options.root || DEFAULT_ROOT);
  const rootKind = options.rootKind || "repo";
  const mode = options.mode || "basic";
  const onnxMode = options.onnxMode || "optional";
  const onnxModels = options.onnxModels || ["midas_small"];
  const torchCacheMode = options.torchCache || (mode === "native" ? "required" : "optional");
  const detectorCacheMode = options.detectorCache || "optional";
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

  const pythonChecks = pythonCandidateChecks(root, platform);
  const platformPython = pythonChecks.filter((item) => item.platformCandidate).map((item) => item.rel);
  const pythonOk = pythonChecks.some((item) => item.exists);
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

  const torchStatus = torchCacheStatus(root);
  infos.push({ rel: path.join("models", "torch-cache"), exists: isDirectory(torchStatus.cacheRoot), required: torchCacheMode === "required", torchStatus });
  if (torchCacheMode !== "off") {
    checks.push({ rel: path.join("models", "torch-cache"), label: "models/torch-cache MiDaS Torch Hub cache", full: torchStatus.cacheRoot, ok: torchStatus.ok, required: torchCacheMode === "required" });
  }
  const detectorStatus = detectorCacheStatus(root);
  if (detectorCacheMode !== "off") {
    checks.push({ rel: path.join("models", "torch-cache", "hub", "checkpoints"), label: "detector checkpoint cache", full: detectorStatus.checkpoints, ok: detectorStatus.ok, required: detectorCacheMode === "required" });
  }

  const requiredModels = onnxMode === "off" || onnxMode === "optional" ? [] : (onnxMode === "require-all" ? ONNX_MODELS : onnxModels);
  const onnxFiles = ONNX_MODELS.map((model) => ({
    rel: path.join("models", "onnx", `${model}.onnx`),
    required: requiredModels.includes(model),
  }));
  for (const item of onnxFiles) {
    const full = path.join(root, item.rel);
    const exists = isFile(full);
    const size = exists ? fs.statSync(full).size : 0;
    infos.push({ rel: item.rel, exists, size, required: item.required });
    if (item.required) checks.push({ rel: item.rel, label: item.rel, full, ok: exists && size > 0, required: true });
  }

  const failed = checks.filter((check) => check.required && !check.ok);
  return {
    schemaVersion: 1,
    root,
    rootKind,
    mode,
    onnxMode,
    onnxModels,
    platform,
    modelReadiness: {
      torchCache: torchStatus,
      onnx: infos.filter((item) => !item.torchStatus && item.rel && item.rel.endsWith(".onnx")),
    },
    detectorReadiness: detectorStatus,
    pythonCandidateChecks: pythonChecks,
    checks,
    infos,
    failed,
    remediation: remediation({ rootKind }),
    ok: failed.length === 0,
  };
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
    "  macOS/Linux: npm run setup:<platform>",
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
  lines.push("\nModel cache summary:");
  const torchInfo = result.infos.find((info) => info.torchStatus);
  if (torchInfo) {
    const st = torchInfo.torchStatus;
    lines.push(`${st.ok ? "✓" : "✗"} models/torch-cache repo=${st.repoCached ? "yes" : "no"} checkpoints=${st.checkpointFiles.length} models=${Object.entries(st.modelReady).map(([k,v]) => `${k}:${v ? "ok" : "missing"}`).join(",")}`);
  }
  lines.push("\nONNX resource summary:");
  for (const info of result.infos.filter((item) => !item.torchStatus)) {
    const required = info.required ? "required" : "optional";
    lines.push(`${info.exists ? "✓" : info.required ? "✗" : "!"} ${info.rel} ${info.exists ? `${info.size} bytes` : `missing (${required})`}`);
  }
  if (!result.ok) {
    lines.push("");
    lines.push(`DepthLens ${result.rootKind} resources are incomplete under: ${result.root}`);
    lines.push(remediation(result));
  } else {
    lines.push("");
    if (result.onnxMode === "off") lines.push("ONNX was intentionally skipped for this verification run.");
    lines.push(`DepthLens resources verified under: ${result.root} (root-kind=${result.rootKind}, mode=${result.mode}, onnx=${result.onnxMode}, models=${(result.onnxModels || []).join(",")})`);
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
  if (options.json) {
    const json = JSON.stringify(result, null, 2);
    if (result.ok) console.log(json);
    else {
      console.error(json);
      process.exit(1);
    }
    return;
  }
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
  pythonCandidateChecks,
  verifyResourceRoot,
  formatResult,
  remediation,
  parseOnnxModels,
  ONNX_MODELS,
  torchCacheStatus,
  detectorCacheStatus,
};
