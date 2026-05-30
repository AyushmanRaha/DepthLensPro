const { contextBridge, ipcRenderer } = require("electron");

// Expose a narrow IPC surface; never pass ipcRenderer into the renderer.
contextBridge.exposeInMainWorld("electronAPI", {
  getBackendUrl: () => ipcRenderer.invoke("get-backend-url"),
  getAppVersion: () => ipcRenderer.invoke("get-app-version"),
  showSaveDialog: (options) => ipcRenderer.invoke("show-save-dialog", options),
  showOpenDialog: (options) => ipcRenderer.invoke("show-open-dialog", options),
  platform: process.platform,
  arch: process.arch,
});
