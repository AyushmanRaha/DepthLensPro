"use strict";

const SUPPORTED_TARGETS = new Set([
  "darwin:arm64",
  "win32:arm64",
  "win32:x64",
  "linux:arm64",
  "linux:x64",
]);

function normalizeArch(value = process.arch) {
  const raw = String(value || "").toLowerCase();
  if (["x64", "x86_64", "amd64"].includes(raw)) return "x64";
  if (["arm64", "aarch64"].includes(raw)) return "arm64";
  return raw;
}

function normalizePlatform(value = process.platform) {
  const raw = String(value || "").toLowerCase();
  if (["darwin", "macos"].includes(raw)) return "darwin";
  if (["win32", "windows"].includes(raw)) return "win32";
  if (raw === "linux") return "linux";
  return raw;
}

function platformLabel(platform = process.platform, arch = process.arch) {
  const p = normalizePlatform(platform);
  const a = normalizeArch(arch);
  const names = { darwin: "macOS", win32: "Windows", linux: "Linux" };
  return `${names[p] || p} ${a}`;
}

function evaluateTarget(platform = process.platform, arch = process.arch) {
  const p = normalizePlatform(platform);
  const a = normalizeArch(arch);
  const supported = SUPPORTED_TARGETS.has(`${p}:${a}`);
  let reason = null;
  if (!supported) {
    if (p === "darwin" && ["x64", "universal"].includes(a)) {
      reason = "macOS x64/universal native builds are intentionally unsupported. Use Apple Silicon macOS arm64.";
    } else if (!["darwin", "win32", "linux"].includes(p)) {
      reason = `Unsupported operating system: ${p}. Supported: macOS arm64, Windows x64/arm64, Linux x64/arm64.`;
    } else {
      reason = `Unsupported native target ${platformLabel(p, a)}. Supported: macOS arm64, Windows x64/arm64, Linux x64/arm64.`;
    }
  }
  return { platform: p, arch: a, label: platformLabel(p, a), supported, reason };
}

function isSupportedTarget(platform = process.platform, arch = process.arch) {
  return evaluateTarget(platform, arch).supported;
}

module.exports = { SUPPORTED_TARGETS, normalizeArch, normalizePlatform, platformLabel, evaluateTarget, isSupportedTarget };
