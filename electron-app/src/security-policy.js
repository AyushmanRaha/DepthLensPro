const path = require("path");

function normalizeFilePath(fileUrl) {
  try {
    return path.resolve(decodeURIComponent(fileUrl.pathname));
  } catch (_) {
    return "";
  }
}

function isAllowedAppUrl(url, { backendHost = "127.0.0.1", backendPort, frontendPath } = {}) {
  let parsedUrl;
  try {
    parsedUrl = new URL(url);
  } catch (_) {
    return false;
  }

  if (parsedUrl.protocol === "http:") {
    return (
      parsedUrl.hostname === backendHost
      && String(parsedUrl.port || "80") === String(backendPort)
    );
  }

  if (parsedUrl.protocol === "file:" && frontendPath) {
    const targetPath = normalizeFilePath(parsedUrl);
    const allowedPath = path.resolve(frontendPath);
    return targetPath === allowedPath;
  }

  return false;
}

function isExternalUrl(url) {
  let parsedUrl;
  try {
    parsedUrl = new URL(url);
  } catch (_) {
    return false;
  }

  return ["https:", "mailto:"].includes(parsedUrl.protocol);
}

module.exports = { isAllowedAppUrl, isExternalUrl };
