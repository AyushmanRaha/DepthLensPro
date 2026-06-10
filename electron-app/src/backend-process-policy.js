const path = require("path");

function normalizeProcessText(value) {
  return String(value || "").toLowerCase();
}

function normalizeCandidatePath(value) {
  if (!value || typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  return path.normalize(trimmed).toLowerCase();
}

function buildOwnershipCandidatePaths({ storedMetadata = null, cwd = "", backendDir = "" } = {}) {
  return [
    storedMetadata?.backendDir,
    storedMetadata?.cwd,
    backendDir,
    cwd,
  ]
    .map(normalizeCandidatePath)
    .filter((candidate) => candidate && candidate !== path.parse(candidate).root);
}

function isDepthLensOwnedProcess({ commandLine = "", storedMetadata = null, cwd = "", backendDir = "" } = {}) {
  const normalized = normalizeProcessText(commandLine);
  if (!normalized) return false;

  if (normalized.includes("depthlens pro") || normalized.includes("depthlenspro")) return true;

  const runsDepthLensBackend = normalized.includes("uvicorn") && normalized.includes("backend.app:app");
  if (!runsDepthLensBackend) return false;

  const candidates = buildOwnershipCandidatePaths({ storedMetadata, cwd, backendDir });
  return candidates.some((candidate) => normalized.includes(candidate));
}

module.exports = { buildOwnershipCandidatePaths, isDepthLensOwnedProcess };
