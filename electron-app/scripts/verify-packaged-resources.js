#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const { verifyResourceRoot, formatResult, parseOnnxModels } = require("./verify-resources");

const APP_ROOT = path.resolve(__dirname, "..");

function parseArgs(argv = process.argv.slice(2)) {
  const options = {
    platform: process.platform,
    arch: process.arch,
    dist: path.join(APP_ROOT, "dist"),
    mode: "native",
    onnxMode: "optional",
    onnxModels: ["midas_small"],
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--platform") options.platform = argv[++i] || options.platform;
    else if (arg === "--arch") options.arch = argv[++i] || options.arch;
    else if (arg === "--dist") options.dist = path.resolve(argv[++i] || options.dist);
    else if (arg === "--mode") options.mode = argv[++i] || options.mode;
    else if (arg === "--onnx") options.onnxMode = argv[++i] || options.onnxMode;
    else if (arg === "--models" || arg === "--onnx-models") options.onnxModels = parseOnnxModels(argv[++i] || "midas_small");
    else if (arg === "--help" || arg === "-h") options.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return options;
}

function usage() {
  return [
    "Usage: node scripts/verify-packaged-resources.js [--platform darwin|win32|linux] [--arch arm64|x64] [--mode native] [--onnx optional|required|require-all|off] [--models midas_small|dpt_hybrid|dpt_large|all|comma-list]",
    "",
    "Discovers electron-builder packaged resource roots under electron-app/dist and verifies backend, frontend, venv, models, and models/onnx.",
  ].join("\n");
}

function isDirectory(targetPath) {
  try { return fs.statSync(targetPath).isDirectory(); } catch (_) { return false; }
}

function walkDirectories(root, visitor, depth = 0, maxDepth = 7) {
  if (depth > maxDepth || !isDirectory(root)) return;
  let entries = [];
  try { entries = fs.readdirSync(root, { withFileTypes: true }); } catch (_) { return; }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const full = path.join(root, entry.name);
    visitor(full, entry.name, depth + 1);
    if (entry.name.endsWith(".app")) continue;
    walkDirectories(full, visitor, depth + 1, maxDepth);
  }
}

function dedupe(paths) {
  return [...new Set(paths.map((item) => path.resolve(item)))];
}

function discoverMacResources(dist, arch) {
  const expected = path.join(dist, `mac-${arch}`, "DepthLens Pro.app", "Contents", "Resources");
  const roots = [];
  if (isDirectory(expected)) roots.push(expected);
  walkDirectories(dist, (full, name) => {
    if (name === "Resources" && full.endsWith(path.join("DepthLens Pro.app", "Contents", "Resources"))) roots.push(full);
  });
  return dedupe(roots);
}

function discoverUnpackedResources(dist, platform, arch) {
  const platformNeedle = platform === "win32" ? "win" : "linux";
  const roots = [];
  walkDirectories(dist, (full, name) => {
    if (name !== "resources") return;
    const parent = path.basename(path.dirname(full)).toLowerCase();
    const normalized = full.toLowerCase();
    const looksUnpacked = parent.includes("unpacked") || normalized.includes("unpacked");
    const matchesPlatform = normalized.includes(platformNeedle);
    const matchesArch = normalized.includes(arch.toLowerCase()) || arch === "arm64" && normalized.includes("aarch64");
    if (looksUnpacked && matchesPlatform && matchesArch) roots.push(full);
  });
  return dedupe(roots);
}

function discoverResourceRoots(options) {
  if (options.platform === "darwin") return discoverMacResources(options.dist, options.arch);
  if (options.platform === "win32" || options.platform === "linux") return discoverUnpackedResources(options.dist, options.platform, options.arch);
  return [];
}

function discoveryFailureMessage(options) {
  if (options.platform === "darwin") {
    return `No macOS packaged resource root found. Expected: ${path.join(options.dist, `mac-${options.arch}`, "DepthLens Pro.app", "Contents", "Resources")}`;
  }
  if (options.platform === "win32") {
    return `No Windows packaged resources directory found under ${options.dist}. Expected an electron-builder *win*${options.arch}*unpacked*/resources-style output. Re-run the matching Windows native build and inspect electron-app/dist.`;
  }
  if (options.platform === "linux") {
    return `No Linux packaged resources directory found under ${options.dist}. Expected an electron-builder *linux*${options.arch}*unpacked*/resources-style output. If only an AppImage remains, extract it with --appimage-extract on Linux and verify the extracted resources, or adjust the build to retain linux-${options.arch}-unpacked.`;
  }
  return `Unsupported platform for packaged resource discovery: ${options.platform}`;
}

function verifyPackagedResources(options) {
  const roots = discoverResourceRoots(options);
  const results = roots.map((root) => verifyResourceRoot({
    root,
    rootKind: "packaged",
    mode: options.mode,
    onnxMode: options.onnxMode,
    platform: options.platform,
    onnxModels: options.onnxModels,
  }));
  return { roots, results, ok: results.length > 0 && results.every((result) => result.ok) };
}

function main() {
  let options;
  try { options = parseArgs(); } catch (err) {
    console.error(err.message);
    console.error(usage());
    process.exit(2);
  }
  if (options.help) {
    console.log(usage());
    return;
  }

  const verification = verifyPackagedResources(options);
  if (!verification.roots.length) {
    console.error(discoveryFailureMessage(options));
    process.exit(1);
  }

  for (const result of verification.results) {
    const output = formatResult(result);
    if (result.ok) console.log(output);
    else console.error(output);
  }
  if (!verification.ok) process.exit(1);
}

if (require.main === module) main();

module.exports = {
  parseArgs,
  discoverResourceRoots,
  verifyPackagedResources,
  discoveryFailureMessage,
};
