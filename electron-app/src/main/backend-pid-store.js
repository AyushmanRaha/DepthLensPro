const fs = require("fs");
const path = require("path");
function getBackendPidPath(app) { return path.join(app.getPath("userData"), "backend.pid"); }
function getBackendMetadataPath(app) { return path.join(app.getPath("userData"), "backend.json"); }
function readStoredBackendPid(app) {
  try { const pid = Number.parseInt(fs.readFileSync(getBackendPidPath(app), "utf8").trim(), 10); return Number.isFinite(pid) && pid > 0 ? pid : null; } catch (_) { return null; }
}
function readStoredBackendMetadata(app) { try { return JSON.parse(fs.readFileSync(getBackendMetadataPath(app), "utf8")); } catch (_) { return null; } }
function writePrivateFile(filePath, contents) { const handle = fs.openSync(filePath, "w", 0o600); try { fs.writeFileSync(handle, contents, "utf8"); } finally { fs.closeSync(handle); } try { fs.chmodSync(filePath, 0o600); } catch (_) {} }
function writeBackendPidFiles(app, metadata, log = console) {
  fs.mkdirSync(app.getPath("userData"), { recursive: true, mode: 0o700 });
  try { fs.chmodSync(app.getPath("userData"), 0o700); } catch (_) {}
  const lifecycleMetadata = { pid: metadata.pid, backendUrl: metadata.backendUrl, port: metadata.port, host: metadata.host, startedAt: metadata.startedAt, appVersion: metadata.appVersion, isPackaged: metadata.isPackaged, platform: metadata.platform, arch: metadata.arch };
  writePrivateFile(getBackendPidPath(app), `${metadata.pid}\n`);
  writePrivateFile(getBackendMetadataPath(app), `${JSON.stringify(lifecycleMetadata, null, 2)}\n`);
  log.info("BACKEND_PID_WRITTEN", { pidFile: getBackendPidPath(app), metadataFile: getBackendMetadataPath(app), pid: metadata.pid });
}
function removeBackendPidFiles(app, log = console) { for (const filePath of [getBackendPidPath(app), getBackendMetadataPath(app)]) { try { if (fs.existsSync(filePath)) fs.unlinkSync(filePath); } catch (err) { log.warn(`Failed to remove ${filePath}: ${err.message}`); } } }
module.exports = { getBackendPidPath, getBackendMetadataPath, readStoredBackendPid, readStoredBackendMetadata, writePrivateFile, writeBackendPidFiles, removeBackendPidFiles };
