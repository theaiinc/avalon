const { app, BrowserWindow, ipcMain, shell, Tray, Menu, nativeImage } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const os = require('node:os');
const crypto = require('node:crypto');

const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);
const dashboardHost = '127.0.0.1';
const dashboardPort = Number(process.env.AVALON_DASHBOARD_PORT || 8771);
const gatewayPort = Number(process.env.AVALON_GATEWAY_PORT || 8787);
let mainWindow;
let tray;
let backendProcess;
let ownsBackend = false;
let stopping = false;

function apiKey() {
  const file = path.join(app.getPath('userData'), 'api-key');
  try {
    const value = fs.readFileSync(file, 'utf8').trim();
    if (value) return value;
  } catch {
    // Generate on first launch.
  }
  const value = crypto.randomBytes(32).toString('base64url');
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${value}\n`, { mode: 0o600 });
  return value;
}

function appRoot() {
  return app.isPackaged ? app.getAppPath() : path.resolve(__dirname, '..', '..');
}

function iconPath() {
  return path.join(__dirname, 'assets', 'avalon-icon.png');
}

function trayIconPath() {
  return path.join(__dirname, 'assets', 'avalon-tray.png');
}

function createTray() {
  if (tray) return;
  let image = nativeImage.createFromPath(trayIconPath());
  if (!image.isEmpty()) image = image.resize({ width: 18, height: 18 });
  tray = new Tray(image);
  tray.setToolTip('Avalon LLM Dashboard');
  tray.setContextMenu(Menu.buildFromTemplate([
    {
      label: 'Show Avalon',
      click: () => {
        if (!mainWindow || mainWindow.isDestroyed()) createWindow();
        else {
          mainWindow.show();
          mainWindow.focus();
        }
      },
    },
    { type: 'separator' },
    {
      label: 'Start LLM Gateway',
      click: () => postJson('/api/api-server/start', {
        model_id: '',
        port: gatewayPort,
        mode: 'both',
        device: process.env.AVALON_GATEWAY_DEVICE || '',
        gguf_backend: process.env.AVALON_GATEWAY_DEVICE || '',
        openvino_device: process.env.AVALON_OPENVINO_DEVICE || 'NPU',
        gpu_index: '',
      }).catch((error) => console.warn('Could not start gateway:', error.message)),
    },
    {
      label: 'Stop LLM Gateway',
      click: () => request('POST', '/api/api-server/stop').catch(() => {}),
    },
    { type: 'separator' },
    { label: 'Quit Avalon', click: () => app.quit() },
  ]));
  tray.on('click', () => {
    if (!mainWindow || mainWindow.isDestroyed()) createWindow();
    else {
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

function sidecarPath(name) {
  const suffix = process.platform === 'win32' ? '.exe' : '';
  const packaged = path.join(process.resourcesPath, 'sidecars', `${name}${suffix}`);
  if (app.isPackaged && fs.existsSync(packaged)) return packaged;
  return null;
}

function pythonCommand() {
  const bundled = sidecarPath('avalon-backend');
  if (bundled) return { command: bundled, args: [] };
  const root = appRoot();
  if (process.platform === 'win32') {
    return { command: path.join(root, 'backend', '.venv', 'Scripts', 'python.exe'), args: [] };
  }
  return { command: path.join(root, 'backend', '.venv', 'bin', 'python'), args: [] };
}

function spawnBackend() {
  const python = pythonCommand();
  const script = sidecarPath('avalon-backend') ? [] : [path.join(appRoot(), 'backend', 'main.py')];
  const gateway = sidecarPath('avalon-gateway');
  const dataDir = app.isPackaged
    ? path.join(app.getPath('userData'), 'data')
    : path.join(appRoot(), 'data');
  const env = {
    ...process.env,
    // The dashboard remains local-only through main.py middleware; only the
    // one-time pairing accept route is reachable from the LAN.
    AVALON_HOST: '0.0.0.0',
    AVALON_PORT: String(dashboardPort),
    AVALON_GATEWAY_PORT: String(gatewayPort),
    AVALON_GATEWAY_HOST: process.env.AVALON_GATEWAY_HOST || '0.0.0.0',
    AVALON_API_KEY: apiKey(),
    AVALON_DATA_DIR: dataDir,
  };
  if (gateway) env.AVALON_GATEWAY_EXECUTABLE = gateway;
  backendProcess = spawn(python.command, [...python.args, ...script], {
    cwd: appRoot(),
    env,
    stdio: 'ignore',
    windowsHide: true,
  });
  ownsBackend = true;
  backendProcess.once('exit', (code, signal) => {
    backendProcess = undefined;
    if (!stopping && mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('avalon:backend-exit', { code, signal });
    }
  });
}

async function ensureBackend() {
  try {
    await waitForBackend(1500);
    ownsBackend = false;
    return;
  } catch {
    spawnBackend();
    await waitForBackend();
  }
}

function request(method, requestPath) {
  return new Promise((resolve, reject) => {
    const req = http.request({
      host: dashboardHost,
      port: dashboardPort,
      method,
      path: requestPath,
      timeout: 1500,
    }, (res) => {
      res.resume();
      res.once('end', () => resolve(res.statusCode));
    });
    req.once('error', reject);
    req.once('timeout', () => req.destroy(new Error('dashboard timeout')));
    req.end();
  });
}

function postJson(requestPath, payload) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload);
    const req = http.request({
      host: dashboardHost,
      port: dashboardPort,
      method: 'POST',
      path: requestPath,
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
      timeout: 310000,
    }, (res) => {
      let text = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { text += chunk; });
      res.once('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(text ? JSON.parse(text) : {});
        } else {
          const error = new Error(`dashboard returned ${res.statusCode}: ${text}`);
          error.statusCode = res.statusCode;
          error.retryAfter = Number(res.headers['retry-after'] || 5);
          try { error.payload = text ? JSON.parse(text) : {}; } catch { error.payload = {}; }
          reject(error);
        }
      });
    });
    req.once('error', reject);
    req.once('timeout', () => req.destroy(new Error('dashboard timeout')));
    req.end(body);
  });
}

async function stopBackend() {
  if (stopping) return;
  stopping = true;
  if (ownsBackend) {
    try {
      await request('POST', '/api/api-server/stop');
    } catch {
      // The dashboard may already have exited.
    }
  }
  if (ownsBackend && backendProcess && !backendProcess.killed) {
    backendProcess.kill();
  }
  backendProcess = undefined;
  ownsBackend = false;
}

async function waitForBackend(timeoutMs = 90000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const status = await request('GET', '/api/api-server/status');
      if (status && status < 500) return;
    } catch {
      // Continue polling while Python starts.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`Avalon dashboard did not start on ${dashboardHost}:${dashboardPort}`);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 960,
    minHeight: 640,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
  mainWindow.on('close', (event) => {
    if (!stopping && tray) {
      event.preventDefault();
      mainWindow.hide();
    }
  });
  const devUrl = process.env.VITE_DEV_SERVER_URL || 'http://127.0.0.1:5173';
  if (isDev || !app.isPackaged) {
    mainWindow.loadURL(devUrl);
  } else {
    mainWindow.loadFile(path.join(appRoot(), 'dist', 'index.html'));
  }
  createTray();
  if (process.platform === 'darwin' && !nativeImage.createFromPath(iconPath()).isEmpty()) {
    app.dock.setIcon(iconPath());
  }
}

ipcMain.handle('avalon:runtime-config', () => ({
  dashboardUrl: `http://${dashboardHost}:${dashboardPort}`,
  gatewayUrl: `http://127.0.0.1:${gatewayPort}`,
  publicGatewayUrl: `http://${os.hostname()}:${gatewayPort}`,
  apiKey: apiKey(),
  packaged: app.isPackaged,
}));

ipcMain.handle('avalon:quick-test', async (_event, payload) => {
  const deadline = Date.now() + 15 * 60 * 1000;
  while (true) {
    try {
      return await postJson('/api/api-server/quick-test', payload);
    } catch (error) {
      const resourcePressure = error?.statusCode === 503
        && error?.payload?.error?.type === 'resource_pressure';
      if (!resourcePressure || Date.now() >= deadline) throw error;
      const retryAfter = Number(error.retryAfter || error.payload?.error?.resource?.retry_after || 5);
      await new Promise((resolve) => setTimeout(resolve, Math.max(1, retryAfter) * 1000));
    }
  }
});

function configureLaunchAtLogin() {
  if (process.platform !== 'darwin' || !app.isPackaged) return;
  app.setLoginItemSettings({
    openAtLogin: true,
    openAsHidden: false,
  });
}

app.whenReady().then(async () => {
  configureLaunchAtLogin();
  try {
    await ensureBackend();
    if (process.env.AVALON_AUTOSTART_GATEWAY !== 'false') {
      try {
        await postJson('/api/api-server/start', {
          model_id: '',
          port: gatewayPort,
          mode: 'both',
          device: process.env.AVALON_GATEWAY_DEVICE || '',
          gguf_backend: process.env.AVALON_GATEWAY_DEVICE || '',
          openvino_device: process.env.AVALON_OPENVINO_DEVICE || 'NPU',
          gpu_index: '',
        });
      } catch (error) {
        console.warn('Avalon gateway was not auto-started:', error.message);
      }
    }
  } catch (error) {
    console.error(error);
  }
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('before-quit', (event) => {
  if (stopping) return;
  event.preventDefault();
  stopBackend().finally(() => app.exit(0));
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
