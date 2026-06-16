const { contextBridge, ipcRenderer } = require("electron");

// Expose a narrow IPC surface; never pass ipcRenderer into the renderer.
contextBridge.exposeInMainWorld("electronAPI", {
  getBackendUrl: () => ipcRenderer.invoke("get-backend-url"),
  getAppVersion: () => ipcRenderer.invoke("get-app-version"),
  // FIX (Issue 4): expose platform so the renderer can conditionally apply
  // macOS traffic-light padding without hard-coding it in CSS
  getPlatform: () => ipcRenderer.invoke("get-platform"),
  showSaveDialog: (options) => ipcRenderer.invoke("show-save-dialog", options),
  showOpenDialog: (options) => ipcRenderer.invoke("show-open-dialog", options),
  platform: process.platform,
  arch: process.arch,
  storage: {
    path: () => ipcRenderer.invoke("storage-path"),
    settingsGet: (key) => ipcRenderer.invoke("settings-get", key),
    settingsSet: (key, value) => ipcRenderer.invoke("settings-set", key, value),
    benchmarksHistory: () => ipcRenderer.invoke("benchmarks-history"),
    modelsStatus: () => ipcRenderer.invoke("models-status"),
    cacheClear: () => ipcRenderer.invoke("cache-clear"),
  },
});