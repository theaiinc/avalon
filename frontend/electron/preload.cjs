const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('avalon', {
  getRuntimeConfig: () => ipcRenderer.invoke('avalon:runtime-config'),
  quickTest: (payload) => ipcRenderer.invoke('avalon:quick-test', payload),
  onBackendExit: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('avalon:backend-exit', listener);
    return () => ipcRenderer.removeListener('avalon:backend-exit', listener);
  },
});
