const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");
const { verifyResourceRoot } = require("./scripts/verify-resources");
const { discoverResourceRoots } = require("./scripts/verify-packaged-resources");

function tempRoot() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "depthlens-resources-"));
}

function touch(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, "x");
}

function mkdir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function makeTree(root, { platform = process.platform, models = true, onnxDir = true, onnxFile = false } = {}) {
  mkdir(path.join(root, "backend"));
  touch(path.join(root, "backend", "app.py"));
  mkdir(path.join(root, "frontend"));
  touch(path.join(root, "frontend", "index.html"));
  if (platform === "win32") touch(path.join(root, "venv", "Scripts", "python.exe"));
  else touch(path.join(root, "venv", "bin", "python3"));
  if (models) mkdir(path.join(root, "models"));
  if (models && onnxDir) mkdir(path.join(root, "models", "onnx"));
  if (models && onnxDir && onnxFile) touch(path.join(root, "models", "onnx", "midas_small.onnx"));
}

function verify(root, options = {}) {
  return verifyResourceRoot({ root, rootKind: "repo", mode: "native", onnxMode: "optional", ...options });
}

{
  const root = tempRoot();
  makeTree(root);
  const result = verify(root);
  assert.strictEqual(result.ok, true, "valid minimal repo root should pass without ONNX binaries");
}

{
  const root = tempRoot();
  makeTree(root, { models: false });
  const result = verify(root);
  assert.strictEqual(result.ok, false, "missing models should fail native verification");
  assert(result.failed.some((item) => item.rel === "models"));
}

{
  const root = tempRoot();
  makeTree(root, { onnxDir: false });
  const result = verify(root);
  assert.strictEqual(result.ok, false, "missing models/onnx should fail native verification");
  assert(result.failed.some((item) => item.rel.endsWith(path.join("models", "onnx"))));
}

{
  const root = tempRoot();
  makeTree(root, { models: false });
  const result = verify(root, { rootKind: "packaged" });
  assert.strictEqual(result.ok, false, "packaged root missing models should fail");
}

{
  const root = tempRoot();
  makeTree(root, { onnxDir: false });
  const result = verify(root, { rootKind: "packaged" });
  assert.strictEqual(result.ok, false, "packaged root missing models/onnx should fail");
}

{
  const root = tempRoot();
  makeTree(root);
  const result = verify(root, { onnxMode: "optional" });
  assert.strictEqual(result.ok, true, "--onnx optional should allow missing .onnx binaries");
}

{
  const root = tempRoot();
  makeTree(root);
  const result = verify(root, { onnxMode: "required" });
  assert.strictEqual(result.ok, false, "--onnx required should fail without midas_small.onnx");
  assert(result.failed.some((item) => item.rel.endsWith(path.join("models", "onnx", "midas_small.onnx"))));
}

{
  const root = tempRoot();
  makeTree(root, { platform: "win32" });
  const result = verify(root, { platform: "win32" });
  assert.strictEqual(result.ok, true, "fake Windows venv/Scripts/python.exe should satisfy win32 platform Python check");
}

{
  const root = tempRoot();
  makeTree(root, { platform: "linux" });
  const result = verify(root, { platform: "linux" });
  assert.strictEqual(result.ok, true, "fake POSIX venv/bin/python3 should satisfy Linux platform Python check");
}

{
  const dist = tempRoot();
  const macResources = path.join(dist, "mac-arm64", "DepthLens Pro.app", "Contents", "Resources");
  makeTree(macResources, { platform: "darwin" });
  assert.deepStrictEqual(discoverResourceRoots({ dist, platform: "darwin", arch: "arm64" }), [macResources]);
}

{
  const dist = tempRoot();
  const winResources = path.join(dist, "win-arm64-unpacked", "resources");
  makeTree(winResources, { platform: "win32" });
  assert.deepStrictEqual(discoverResourceRoots({ dist, platform: "win32", arch: "arm64" }), [winResources]);
}

{
  const dist = tempRoot();
  const linuxResources = path.join(dist, "linux-arm64-unpacked", "resources");
  makeTree(linuxResources, { platform: "linux" });
  assert.deepStrictEqual(discoverResourceRoots({ dist, platform: "linux", arch: "arm64" }), [linuxResources]);
}

{
  const root = tempRoot();
  makeTree(root, { onnxDir: false });
  const cli = spawnSync(process.execPath, [path.join(__dirname, "scripts", "verify-resources.js"), "--root-kind", "packaged", "--mode", "native", "--onnx", "optional", root], { encoding: "utf8" });
  assert.notStrictEqual(cli.status, 0, "CLI should fail for packaged root missing models/onnx");
  assert(cli.stderr.includes("Packaged app resources are incomplete"), cli.stderr);
}

console.log("Resource verification tests passed.");

{
  const root = tempRoot();
  makeTree(root, { onnxFile: true });
  const result = verify(root, { onnxMode: "required", onnxModels: ["midas_small"] });
  assert.strictEqual(result.ok, true, "required MiDaS Small passes when midas_small.onnx is present");
}

{
  const root = tempRoot();
  makeTree(root, { onnxFile: true });
  const result = verify(root, { onnxMode: "required", onnxModels: ["dpt_hybrid"] });
  assert.strictEqual(result.ok, false, "required selected DPT Hybrid fails when dpt_hybrid.onnx is missing");
}

{
  const mac = spawnSync("bash", ["-n", path.join(__dirname, "..", "scripts", "build-native-macos.sh")], { encoding: "utf8" });
  assert.strictEqual(mac.status, 0, mac.stderr || mac.stdout);
  const linux = spawnSync("bash", ["-n", path.join(__dirname, "..", "scripts", "build-native-linux.sh")], { encoding: "utf8" });
  assert.strictEqual(linux.status, 0, linux.stderr || linux.stdout);
  const winScript = fs.readFileSync(path.join(__dirname, "..", "scripts", "build-native-windows.ps1"), "utf8");
  assert(winScript.includes("@SetupArgs"), "Windows build script must pass through all setup arguments");
  assert(winScript.includes("--onnx-models"), "Windows build script must parse --onnx-models examples");
}
