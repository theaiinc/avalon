export interface GPU {
  vendor: string;
  index: string;
  name: string;
  memory_mb: number;
  shared_memory_mb?: number;
  driver_version: string;
  backend: string;
}

export interface DriverRelease {
  id: number;
  tag_name: string;
  name: string;
  published_at: string;
  source: "official" | "community";
  source_repo?: string;
  source_label?: string;
  backends?: string[];
  assets: ReleaseAsset[];
}

export interface ReleaseAsset {
  id: number;
  name: string;
  size: number;
  browser_download_url: string;
  content_type: string;
}

export interface LocalDriver {
  id: string;
  tag: string;
  backend: string;
  path: string;
  active: boolean;
  size: number;
  llama_bench_path: string | null;
  source_label?: string;
  has_update: boolean;
  latest_tag: string | null;
}

export interface ActiveDriver {
  id: string;
  tag: string;
  backend: string;
  path: string;
  llama_bench_path: string | null;
}

export interface HFModel {
  id: string;
  name: string;
  author: string;
  downloads: number;
  likes: number;
  pipeline_tag: string;
  last_modified: string;
}

export interface ModelFile {
  name: string;
  size: number;
  path: string;
}

export interface LocalModel {
  id: string;
  repo_id: string;
  files: string[];
  path: string;
  size: number;
  format?: string;
  openvino_path?: string | null;
}

export interface BenchmarkRun {
  backend: string;
  status: string;
  error?: string;
  elapsed_sec?: number;
  results?: any[];
  raw_stdout?: string;
  mtp?: boolean;
  npu?: boolean;
  draft_model?: string;
}

export interface BenchmarkResult {
  id: string;
  timestamp: string;
  model_name: string;
  model_path: string;
  backends: string[];
  model_params: Record<string, any>;
  bench_params: Record<string, any>;
  runs: BenchmarkRun[];
}

export interface BenchmarkListItem {
  id: string;
  timestamp: string;
  model_name: string;
  backends: string[];
  n_runs: number;
}

export interface DownloadProgress {
  status: string;
  percent: number;
  stage: string;
}

export interface DownloadResponse {
  download_id: string;
}

export interface PCLink {
  id: string;
  name: string;
  base_url: string;
  created_at?: number;
  updated_at: number;
}

export interface PCLinkTestResult {
  ok: boolean;
  base_url: string;
  model_count?: number;
  models?: { id: string; [key: string]: any }[];
  error?: string;
}

export interface PairingCode {
  session_id: string;
  code: string;
  expires_at: number;
  device_id: string;
  device_name: string;
  public_key: string;
}

export interface PairingPeer {
  id: string;
  name: string;
  public_key: string;
  paired_at: number;
}

export interface DiscoveredDevice {
  name: string;
  device_id: string;
  public_key: string;
  base_url: string;
}

export interface GPUMetric {
  name: string;
  backend: string;
  utilization_percent: number;
  memory_used_mb: number;
  memory_total_mb: number;
  temperature_c: number;
}

export interface SystemStats {
  cpu: { percent: number; count: number };
  ram: { total_gb: number; used_gb: number; percent: number };
  gpus: GPUMetric[];
  timestamp: number;
}
