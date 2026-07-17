import type { GPU, DriverRelease, LocalDriver, ActiveDriver, HFModel, ModelFile, LocalModel, DownloadProgress, DownloadResponse, BenchmarkResult, BenchmarkListItem, SystemStats, PCLink, PCLinkTestResult, PairingCode, PairingPeer, DiscoveredDevice } from '../types';

const BASE = window.location.protocol === 'file:' ? 'http://127.0.0.1:8771/api' : '/api';

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export const api = {
  listGPUs: () => fetchJSON<{ gpus: GPU[] }>(`${BASE}/gpu/list`),
  listDriverUpdates: () => fetchJSON<{ updates: { device_name: string; installed: string; latest: string; label: string; download_url: string }[] }>(`${BASE}/gpu/driver-updates`),

  listDrivers: () => fetchJSON<{ local: LocalDriver[]; active: ActiveDriver[] }>(`${BASE}/drivers`),
  listReleases: () => fetchJSON<{ releases: DriverRelease[] }>(`${BASE}/drivers/releases`),
  downloadDriver: (tag: string, backend: string) =>
    fetchJSON<DownloadResponse>(`${BASE}/drivers/download`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tag, backend }),
    }),
  setDriverActive: (driver_id: string, active: boolean) =>
    fetchJSON(`${BASE}/drivers/set-active`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ driver_id, active }),
    }),
  removeDriver: (driver_id: string) =>
    fetchJSON(`${BASE}/drivers/remove`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ driver_id }),
    }),

  searchModels: (q: string, limit = 20, format = 'gguf') =>
    fetchJSON<{ models: HFModel[] }>(`${BASE}/models/search?q=${encodeURIComponent(q)}&limit=${limit}&format=${format}`),
  listModelFiles: (repo_id: string, format = 'gguf') =>
    fetchJSON<{ files: ModelFile[] }>(`${BASE}/models/files?repo_id=${encodeURIComponent(repo_id)}&format=${format}`),
  listLocalModels: () => fetchJSON<{ models: LocalModel[] }>(`${BASE}/models`),
  downloadModel: (repo_id: string, filename: string) =>
    fetchJSON<DownloadResponse>(`${BASE}/models/download`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id, filename }),
    }),
  downloadOpenvinoModel: (repo_id: string) =>
    fetchJSON<DownloadResponse>(`${BASE}/models/download-openvino`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id }),
    }),
  removeModel: (model_id: string) =>
    fetchJSON(`${BASE}/models/remove`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id }),
    }),

  runBenchmark: (model_path: string, backends: string[], bench_params?: Record<string, any>) =>
    fetchJSON<{ task_id: string; status: string }>(`${BASE}/benchmark/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path, backends, bench_params }),
    }),
  benchmarkStatus: (task_id: string) =>
    fetchJSON<{ task_id: string; status: string; result?: any }>(`${BASE}/benchmark/status/${task_id}`),
  listActiveBenchmarks: () =>
    fetchJSON<{ active: { id: string }[] }>(`${BASE}/benchmark/list-active`),
  cancelBenchmark: (task_id: string) =>
    fetchJSON<{ task_id: string; cancelled: boolean }>(`${BASE}/benchmark/cancel/${task_id}`, { method: 'POST' }),
  benchmarkStats: () =>
    fetchJSON<SystemStats>(`${BASE}/benchmark/stats`),
  checkCompatibility: (backend: string, modelName?: string, gpuName?: string, driverVersion?: string) => {
    const params = new URLSearchParams();
    if (backend) params.set('backend', backend);
    if (modelName) params.set('model_name', modelName);
    if (gpuName) params.set('gpu_name', gpuName);
    if (driverVersion) params.set('driver_version', driverVersion);
    return fetchJSON<{ matches: any[] }>(`${BASE}/compatibility/check?${params}`);
  },
  listCompatibilityIssues: () => fetchJSON<{ issues: any[] }>(`${BASE}/compatibility/issues`),
  listResults: () => fetchJSON<{ results: BenchmarkListItem[] }>(`${BASE}/benchmark/results`),
  getResult: (id: string) => fetchJSON<BenchmarkResult>(`${BASE}/benchmark/results/${id}`),
  removeResult: (result_id: string) =>
    fetchJSON(`${BASE}/benchmark/results/remove`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ result_id }),
    }),

  listPCLinks: () => fetchJSON<{ links: PCLink[] }>(`${BASE}/pc-links`),
  savePCLink: (name: string, base_url: string, id = '') =>
    fetchJSON<{ link: PCLink; test: PCLinkTestResult }>(`${BASE}/pc-links`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, base_url, id }),
    }),
  testPCLink: (base_url: string) =>
    fetchJSON<PCLinkTestResult>(`${BASE}/pc-links/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_url }),
    }),
  removePCLink: (id: string) =>
    fetchJSON<{ removed: boolean }>(`${BASE}/pc-links/remove`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    }),
  pairingInfo: () => fetchJSON<{ device_id: string; device_name: string; public_key: string }>(`${BASE}/pairing/info`),
  createPairingCode: () =>
    fetchJSON<PairingCode>(`${BASE}/pairing/code`, { method: 'POST' }),
  discoverPairing: () => fetchJSON<{ devices: DiscoveredDevice[] }>(`${BASE}/pairing/discover`),
  connectPairing: (base_url: string, session_id: string, code: string, name = '') =>
    fetchJSON<{ link: PCLink; device: { device_id: string; device_name: string } }>(`${BASE}/pairing/connect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_url, session_id, code, name }),
    }),
  listPairingPeers: () => fetchJSON<{ peers: PairingPeer[] }>(`${BASE}/pairing/peers`),
  removePairingPeer: (peer_id: string) =>
    fetchJSON<{ removed: boolean }>(`${BASE}/pairing/peers/remove`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ peer_id }),
    }),
  saveServingProfile: (base_id: string, profile: Record<string, any>) =>
    fetchJSON<{ profile: Record<string, any>; status: any }>(`${BASE}/api-server/model-profile`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_id, profile }),
    }),
  resetServingProfile: (base_id: string) =>
    fetchJSON<{ removed: boolean; status: any }>(`${BASE}/api-server/model-profile/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_id }),
    }),
};
