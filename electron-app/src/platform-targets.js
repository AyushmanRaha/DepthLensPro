const SUPPORTED_TARGETS = new Set([
  "darwin:arm64",
  "win32:arm64",
  "win32:x64",
  "linux:arm64",
  "linux:x64",
]);
const UNSUPPORTED_MESSAGES = {
  "darwin:x64": "macOS Intel x64 is not supported. DepthLens Pro supports macOS Apple Silicon arm64 only.",
  "darwin:universal": "macOS universal builds are not supported. Build the macOS arm64 target instead.",
};
function targetKey(platform = process.platform, arch = process.arch) { return `${platform}:${arch}`; }
function isSupportedTarget(platform = process.platform, arch = process.arch) { return SUPPORTED_TARGETS.has(targetKey(platform, arch)); }
function assertSupportedTarget(platform = process.platform, arch = process.arch) {
  const key = targetKey(platform, arch);
  if (!SUPPORTED_TARGETS.has(key)) throw new Error(UNSUPPORTED_MESSAGES[key] || `Unsupported DepthLens Pro target: ${platform}/${arch}. Supported targets: ${[...SUPPORTED_TARGETS].join(", ")}.`);
  return { platform, arch, key };
}
function electronBuilderArch(arch = process.arch) { return arch === "arm64" ? "arm64" : arch === "x64" ? "x64" : arch; }
module.exports = { SUPPORTED_TARGETS, UNSUPPORTED_MESSAGES, targetKey, isSupportedTarget, assertSupportedTarget, electronBuilderArch };
