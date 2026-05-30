const { contextBridge, ipcRenderer } = require('electron');

// Expose ONLY specific safe APIs to the renderer (your frontend JS)
// This is the security boundary — never expose ipcRenderer directly
contextBridge.exposeInMainWorld('electronAPI', {
  getBackendUrl:    () => ipcRenderer.invoke('get-backend-url'),
  getAppVersion:    () => ipcRenderer.invoke('get-app-version'),
  showSaveDialog:   (options) => ipcRenderer.invoke('show-save-dialog', options),
  showOpenDialog:   (options) => ipcRenderer.invoke('show-open-dialog', options),
  // Platform info
  platform: process.platform,
  arch:     process.arch,
});