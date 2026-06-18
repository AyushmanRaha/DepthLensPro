const path = require("path");

function getAppRoot({ app, dirname, env = process.env } = {}) {
  const isDev = !app.isPackaged || env.NODE_ENV === "development";
  return isDev ? path.resolve(dirname, "..") : process.resourcesPath;
}

function getResourcePath({ app, dirname, env = process.env } = {}, ...parts) {
  return path.join(getAppRoot({ app, dirname, env }), ...parts);
}

function logPath(log) {
  return log.transports.file.getFile().path;
}

module.exports = { getAppRoot, getResourcePath, logPath };
