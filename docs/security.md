# Security

[← Back to README](../README.md)

DepthLens Pro is designed as a local-first desktop ML tool. The security model assumes the inference server runs on `127.0.0.1` and is accessed only by the local Electron renderer — not exposed to the network.

### Security Design

| Area | Approach |
|---|---|
| Local inference | All requests go to `127.0.0.1`; no hosted inference service is used |
| Renderer isolation | Electron `contextIsolation: true`, `sandbox: true`, `nodeIntegration: false` |
| Navigation policy | Renderer navigation is restricted to the local frontend file and `127.0.0.1:PORT` only — other localhost ports are blocked |
| External links | HTTPS and `mailto:` links open in the system browser via `shell.openExternal`; new-window requests are denied |
| Backend process ownership | Before killing any process on the backend port, Electron checks that the process command-line and stored PID metadata match a known DepthLens-owned invocation |
| Single instance | Electron prevents multiple desktop app instances from fighting over backend state |
| PID metadata | Backend PID and connection metadata are stored in platform user-data files at mode `0600` |
| Cache serialisation | Cache payloads are serialised as versioned JSON (magic prefix `DLP2\0`). Legacy pickle payloads (prefix `DLP1\0` or `\x80`) are detected, deleted, and never deserialised |
| Error handling | Client-facing 500 responses are sanitised (`"Internal server error"` only); full stack traces remain in server logs |
| Secrets | Default local flow requires no API keys, tokens, or credentials |
| Spawn safety | Backend is started with `spawn(pythonPath, args, { shell: false })` — arguments are passed as an array, not interpolated into a shell string |

### Privacy Notes

- Uploaded images are processed locally and never leave the machine.
- The backend listens on `127.0.0.1` by default; Docker mode exposes the port according to your Compose port mapping.
- First-time PyTorch model loading may download model weights from Torch Hub (GitHub/CDN) if they are not already cached at `~/.cache/torch/hub`.
- ONNX files are generated locally and stored under `models/onnx/`.

### Reporting Vulnerabilities

Please do **not** open a public GitHub issue for security-sensitive reports.

Include:

- Description of the issue
- Steps to reproduce
- Affected component
- Possible impact
- Suggested mitigation, if known

See [`SECURITY.md`](../SECURITY.md) for the full policy.

<div align="right"><sub><a href="../README.md#depthlens-pro">⬆ back to README</a></sub></div>

---
