const assert = require("node:assert/strict");
const path = require("node:path");
const { isAllowedAppUrl } = require("./src/security-policy");
const { buildOwnershipCandidatePaths, isDepthLensOwnedProcess } = require("./src/backend-process-policy");

const frontendPath = path.resolve(__dirname, "..", "frontend", "index.html");
const fileUrl = `file://${frontendPath}`;

assert.equal(isAllowedAppUrl(fileUrl, { backendHost: "127.0.0.1", backendPort: 8765, frontendPath }), true);
assert.equal(isAllowedAppUrl("http://127.0.0.1:8765/live", { backendHost: "127.0.0.1", backendPort: 8765, frontendPath }), true);
assert.equal(isAllowedAppUrl("http://127.0.0.1:8766/live", { backendHost: "127.0.0.1", backendPort: 8765, frontendPath }), false);
assert.equal(isAllowedAppUrl("http://localhost:8765/live", { backendHost: "127.0.0.1", backendPort: 8765, frontendPath }), false);
assert.equal(isAllowedAppUrl("https://example.com", { backendHost: "127.0.0.1", backendPort: 8765, frontendPath }), false);

assert.deepEqual(buildOwnershipCandidatePaths({ cwd: "", backendDir: "" }), []);
assert.equal(
  isDepthLensOwnedProcess({
    commandLine: "python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765",
    cwd: "",
    backendDir: "",
  }),
  false,
);
assert.equal(
  isDepthLensOwnedProcess({
    commandLine: "python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765",
    cwd: path.resolve(__dirname, ".."),
    backendDir: path.resolve(__dirname, "..", "backend"),
  }),
  false,
);
assert.equal(
  isDepthLensOwnedProcess({
    commandLine: `${process.execPath} -m uvicorn backend.app:app --app-dir ${path.resolve(__dirname, "..")}`,
    cwd: path.resolve(__dirname, ".."),
    backendDir: path.resolve(__dirname, "..", "backend"),
  }),
  true,
);

console.log("Electron security and lifecycle policy tests passed.");
