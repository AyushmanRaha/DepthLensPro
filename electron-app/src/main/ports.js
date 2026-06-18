const net = require("net");
const BACKEND_HOST = "127.0.0.1";
function isPortAvailable(port, host = BACKEND_HOST) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", (err) => {
      if (err.code === "EADDRINUSE" || err.code === "EACCES") resolve(false);
      else reject(err);
    });
    server.once("listening", () => { server.close(() => resolve(true)); });
    server.listen(port, host);
  });
}
async function findAvailableBackendPort(startPort, host = BACKEND_HOST, env = process.env) {
  const envPinned = Boolean(env.DEPTHLENS_BACKEND_PORT);
  const maxAttempts = envPinned ? 1 : 25;
  for (let offset = 0; offset < maxAttempts; offset += 1) {
    const candidate = startPort + offset;
    if (await isPortAvailable(candidate, host)) return candidate;
  }
  return null;
}
module.exports = { isPortAvailable, findAvailableBackendPort, BACKEND_HOST };
