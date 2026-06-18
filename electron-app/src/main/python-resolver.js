const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

function execCapture(command, args, options = {}) { try { return execFileSync(command, args, { encoding: "utf8", timeout: options.timeout || 2500, windowsHide: true }).trim(); } catch (err) { return (err.stdout || "").toString().trim(); } }
function getPythonCandidates({ root, isDev, platform = process.platform, resourcesPath = process.resourcesPath } = {}) {
  const candidates = [];
  if (platform === "win32") {
    candidates.push(path.join(root, "venv", "Scripts", "python.exe"), path.join(root, "venv", "Scripts", "python3.exe"), path.join(resourcesPath || root, "venv", "Scripts", "python.exe"));
    if (isDev) candidates.push("py", "python");
  } else {
    candidates.push(path.join(root, "venv", "bin", "python3"), path.join(root, "venv", "bin", "python"), path.join(resourcesPath || root, "venv", "bin", "python3"), path.join(resourcesPath || root, "venv", "bin", "python"));
    if (isDev) candidates.push("python3", "python", "/usr/bin/python3", "/usr/local/bin/python3", "/opt/homebrew/bin/python3");
  }
  return [...new Set(candidates)];
}
function pythonVersionOk(candidate, log = console) { try { const output = execCapture(candidate, ["-c", "import sys,json; print(json.dumps(list(sys.version_info[:3])))"], { timeout: 3000 }); const version = JSON.parse(output || "[]"); const ok = version.length >= 2 && version[0] === 3 && version[1] >= 10 && version[1] <= 12; log.info("PYTHON_VERSION_PROBE", { candidate, version, ok }); return ok; } catch (err) { log.warn("PYTHON_VERSION_PROBE_FAILED", { candidate, error: err.message }); return false; } }
function getPythonPath({ root, isDev, log = console, platform = process.platform, resourcesPath = process.resourcesPath } = {}) { const candidates = getPythonCandidates({ root, isDev, platform, resourcesPath }); for (const candidate of candidates) { try { if (path.isAbsolute(candidate)) { if (fs.existsSync(candidate) && pythonVersionOk(candidate, log)) { log.info(`Python found at: ${candidate}`); return candidate; } } else if (pythonVersionOk(candidate, log)) { log.info(`Using PATH Python candidate: ${candidate}`); return candidate; } } catch (_) {} } const fallback = candidates[0]; log.warn(`No supported Python found in candidate paths; will try: ${fallback}`); return fallback; }
module.exports = { execCapture, getPythonCandidates, pythonVersionOk, getPythonPath };
