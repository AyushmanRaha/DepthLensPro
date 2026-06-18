const http = require("http");

function requestJson(url, timeoutMs = 2000) {
  return new Promise((resolve, reject) => {
    let connected = false;
    const req = http.get(url, { timeout: timeoutMs }, (res) => {
      connected = true;
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { body += chunk; });
      res.on("end", () => {
        let json = null;
        try { json = body ? JSON.parse(body) : null; } catch (_) {}
        resolve({ statusCode: res.statusCode, json, body, connected, empty: body.length === 0 });
      });
    });
    req.on("socket", (socket) => { socket.on("connect", () => { connected = true; }); });
    req.on("error", (err) => {
      err.connected = connected;
      if (err.code === "ECONNREFUSED") err.kind = "connection_refused";
      else if (connected && /timed out|timeout/i.test(err.message)) err.kind = "tcp_connected_http_timeout";
      reject(err);
    });
    req.on("timeout", () => {
      const err = new Error(`Timed out requesting ${url}`);
      err.connected = connected;
      err.kind = connected ? "tcp_connected_http_timeout" : "timeout";
      req.destroy(err);
    });
  });
}

async function probeLive(url, timeoutMs = 1200, attempt = 0, log = console) {
  const liveUrl = `${url}/live`;
  try {
    const live = await requestJson(liveUrl, timeoutMs);
    const valid = Boolean(live.statusCode === 200 && live.json?.service === "DepthLens Pro API" && live.json?.status === "ok");
    log.info("LIVE_POLL_ATTEMPT", { attempt, url: liveUrl, statusCode: live.statusCode, json: live.json, valid, empty: live.empty });
    return { ok: valid, kind: valid ? "valid_depthlens_live" : "non_depthlens_response", ...live };
  } catch (err) {
    const kind = err.kind || (err.code === "ECONNREFUSED" ? "connection_refused" : "error");
    if (kind === "tcp_connected_http_timeout") log.warn("LIVE_TCP_CONNECTED_HTTP_TIMEOUT", { attempt, url: liveUrl, error: err.message });
    log.info("LIVE_POLL_ATTEMPT", { attempt, url: liveUrl, error: err.message, kind });
    return { ok: false, kind, error: err };
  }
}

async function isLiveDepthLensBackend(url) { return (await probeLive(url, 1200, 0)).ok; }

module.exports = { requestJson, probeLive, isLiveDepthLensBackend };
