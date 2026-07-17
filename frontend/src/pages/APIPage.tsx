import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../api/client';
import Spinner from '../components/Spinner';
import type { LocalModel, LocalDriver } from '../types';

type ServerStatus = 'stopped' | 'running' | 'starting' | 'error';
type ApiMode = 'openai' | 'anthropic' | 'both';

interface ServedModel {
  id: string;
  base_id?: string;
  served_id?: string;
  source?: string;
  format?: string;
  runtime?: string;
  driver?: string;
  model_file?: string;
  mtp?: boolean;
  mtp_head_path?: string | null;
  is_mtp_head?: boolean;
  dspark?: boolean;
  profile?: {
    base_id?: string;
    served_id?: string;
    model_id?: string;
    repo_id?: string;
    source?: string;
    format?: string;
    runtime?: string;
    driver?: string;
    model_file?: string;
    mtp?: boolean;
    mtp_head_path?: string | null;
    is_mtp_head?: boolean;
    dspark?: boolean;
    serving_config?: Record<string, any>;
  };
  current_request?: StreamRequest | null;
  last_request?: StreamRequest | null;
}

interface StreamRequest {
  id: string;
  model: string;
  status: 'streaming' | 'completed' | 'cancelled' | 'error' | string;
  started_at_ms?: number;
  updated_at_ms?: number;
  first_token_at_ms?: number | null;
  first_token_latency_ms?: number | null;
  finished_at_ms?: number;
  streamed_token_estimate: number;
  streamed_chars: number;
}

interface RequestLogEntry {
  at_ms: number;
  method: string;
  path: string;
  status: number;
  duration_ms: number;
  client: string;
  model_id: string;
  model_source: 'local' | 'remote' | string;
  remote_model_id: string;
  remote_pc: string;
  streamed_token_estimate: number;
  detail?: string;
}

interface ProfileDraft {
  served_id: string;
  driver: string;
  model_file: string;
  mtp: boolean;
  dspark: boolean;
  max_tokens_default: number;
  temperature_default: number;
}

export default function APIPage() {
  const [models, setModels] = useState<LocalModel[]>([]);
  const [drivers, setDrivers] = useState<LocalDriver[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [ggufBackend, setGgufBackend] = useState('');
  const [openvinoDevice, setOpenvinoDevice] = useState('NPU');
  const [gpuIndex, setGpuIndex] = useState('');
  const [port, setPort] = useState(8787);
  const [mode, setMode] = useState<ApiMode>('openai');
  const [status, setStatus] = useState<ServerStatus>('stopped');
  const [config, setConfig] = useState<any>({});
  const [error, setError] = useState('');
  const [servedModels, setServedModels] = useState<ServedModel[]>([]);
  const [editingProfiles, setEditingProfiles] = useState<Record<string, ProfileDraft>>({});
  const [profileError, setProfileError] = useState('');
  const [runtime, setRuntime] = useState<AvalonRuntimeConfig | null>(null);
  const [resourceState, setResourceState] = useState<any>({});
  const [requestLog, setRequestLog] = useState<RequestLogEntry[]>([]);
  const [showResourceWarning, setShowResourceWarning] = useState(false);

  const dashboardBase = runtime?.dashboardUrl || '';
  const gatewayBase = runtime?.gatewayUrl || `http://127.0.0.1:${port}`;
  const activeModels = servedModels.filter((model) => Boolean(model.current_request));

  useEffect(() => {
    if (resourceState.pressure) {
      setShowResourceWarning(true);
      return;
    }
    const timer = window.setTimeout(() => setShowResourceWarning(false), 3000);
    return () => window.clearTimeout(timer);
  }, [resourceState.pressure]);

  const refreshStatus = useCallback(async () => {
    try {
      const r = await fetch(`${dashboardBase}/api/api-server/status`);
      if (!r.ok) return;
      let data = await r.json();
      if (data.status === 'running') {
        const gatewayPort = data.config?.port || 8787;
        try {
          const gatewayStatus = await fetch(`${runtime?.gatewayUrl || `http://127.0.0.1:${gatewayPort}`}/v1/status`, {
            cache: 'no-store',
            headers: runtime?.apiKey ? { 'X-API-Key': runtime.apiKey } : {},
          });
          if (gatewayStatus.ok) {
            const gatewayData = await gatewayStatus.json();
            data = {
              ...data,
              config: { ...(data.config || {}), ...(gatewayData.config || {}) },
              models: Array.isArray(gatewayData.models) ? gatewayData.models : data.models,
              resources: gatewayData.resources || data.resources,
              request_log: Array.isArray(gatewayData.request_log) ? gatewayData.request_log : data.request_log,
            };
          }
        } catch { }
      }
      setConfig(data.config || {});
      setResourceState(data.resources || {});
      if (Array.isArray(data.request_log)) {
        setRequestLog((current) => {
          // Keep the visible terminal stable during a transient gateway
          // status miss. Only replace it when entries actually change.
          if (data.request_log.length === 0 && current.length > 0) return current;
          if (JSON.stringify(current) === JSON.stringify(data.request_log)) return current;
          return data.request_log;
        });
      }
      if (Array.isArray(data.models)) {
        setServedModels(data.models);
        setSelectedModel((current) => current || data.models?.[0]?.id || '');
      }
      if (data.status === 'running') {
        setStatus('running');
        setPort(data.config?.port || 8787);
        setMode(data.config?.mode || 'openai');
        setGgufBackend(data.config?.gguf_backend || data.config?.device || '');
        setOpenvinoDevice(data.config?.openvino_device || 'NPU');
        setGpuIndex(data.config?.gpu_index || '');
      } else if (data.status === 'stopped') {
        setStatus('stopped');
      } else {
        setStatus('error');
        setError(data.status);
      }
    } catch { }
  }, [dashboardBase, runtime]);

  useEffect(() => {
    window.avalon?.getRuntimeConfig().then(setRuntime).catch(() => {});
    api.listLocalModels().then((r) => {
      setModels(r.models);
      if (r.models.length > 0 && !selectedModel) {
        const m = r.models[0];
        setSelectedModel(m.id);
      }
      setServedModels((current) => current.length > 0 ? current : r.models.map((m) => ({
        id: m.id,
        base_id: m.id,
        served_id: m.id,
        source: 'local',
        format: m.format || 'gguf',
        runtime: m.format === 'openvino' ? 'openvino' : 'gguf',
        model_file: m.files?.[0] || m.openvino_path || m.path,
        mtp: `${m.id} ${m.repo_id} ${m.files?.join(' ') || ''}`.toLowerCase().includes('mtp'),
        dspark: `${m.id} ${m.repo_id} ${m.files?.join(' ') || ''}`.toLowerCase().includes('dspark'),
      })));
    }).catch(() => {});
    fetch('/api/drivers').then(r => r.json()).then(data => {
      setDrivers((data.local || []).filter((d: LocalDriver) => d.backend !== 'npu' && d.llama_bench_path));
    }).catch(() => {});
    refreshStatus();
    const timer = setInterval(refreshStatus, 1000);
    return () => clearInterval(timer);
  }, [refreshStatus]);

  const handleModelChange = (id: string) => {
    setSelectedModel(id);
  };

  const backendsWithDriver = [...new Set(drivers.map(d => d.backend))];
  const defaultGgufBackend = backendsWithDriver.includes('metal') ? 'metal' : backendsWithDriver.find((b) => b !== 'npu') || '';

  const profileKey = (m: ServedModel) => m.base_id || m.profile?.base_id || m.id;

  const draftFor = (m: ServedModel): ProfileDraft => ({
    served_id: m.profile?.served_id || m.served_id || m.id,
    driver: m.profile?.driver || m.driver || '',
    model_file: m.profile?.model_file || m.model_file || '',
    mtp: Boolean(m.profile?.mtp ?? m.mtp),
    dspark: Boolean(m.profile?.dspark ?? m.dspark),
    max_tokens_default: Number(m.profile?.serving_config?.max_tokens_default ?? 512),
    temperature_default: Number(m.profile?.serving_config?.temperature_default ?? 0.7),
  });

  const updateDraft = (key: string, patch: Partial<ProfileDraft>) => {
    setEditingProfiles((current) => ({
      ...current,
      [key]: { ...current[key], ...patch },
    }));
  };

  const startEditingProfile = (m: ServedModel) => {
    setProfileError('');
    setEditingProfiles((current) => ({ ...current, [profileKey(m)]: draftFor(m) }));
  };

  const cancelEditingProfile = (key: string) => {
    setEditingProfiles((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
  };

  const saveProfile = async (m: ServedModel) => {
    const key = profileKey(m);
    const draft = editingProfiles[key];
    if (!draft) return;
    setProfileError('');
    try {
      await api.saveServingProfile(key, {
        served_id: draft.served_id,
        driver: draft.driver,
        model_file: draft.model_file,
        mtp: draft.mtp,
        dspark: draft.dspark,
        serving_config: {
          max_tokens_default: draft.max_tokens_default,
          temperature_default: draft.temperature_default,
        },
      });
      cancelEditingProfile(key);
      await refreshStatus();
    } catch (e: any) {
      setProfileError(e.message);
    }
  };

  const resetProfile = async (m: ServedModel) => {
    const key = profileKey(m);
    setProfileError('');
    try {
      await api.resetServingProfile(key);
      cancelEditingProfile(key);
      await refreshStatus();
    } catch (e: any) {
      setProfileError(e.message);
    }
  };

  const handleStart = async () => {
    setStatus('starting');
    setError('');
    try {
      const r = await fetch(`${dashboardBase}/api/api-server/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: '',
          port,
          mode,
          device: ggufBackend || defaultGgufBackend,
          gguf_backend: ggufBackend || defaultGgufBackend,
          openvino_device: openvinoDevice,
          gpu_index: gpuIndex,
        }),
      });
      if (!r.ok) {
        const txt = await r.text();
        throw new Error(txt);
      }
      await refreshStatus();
    } catch (e: any) {
      setStatus('error');
      setError(e.message);
    }
  };

  const handleStop = async () => {
    try {
      const response = await fetch(`${dashboardBase}/api/api-server/stop`, { method: 'POST' });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        const active = body.detail?.active_requests?.length;
        throw new Error(
          active
            ? `Gateway is serving ${active} request${active === 1 ? '' : 's'}; stop was not performed.`
            : body.detail?.message || 'Gateway could not be stopped.',
        );
      }
      setStatus('stopped');
    } catch (e: any) {
      setError(e.message);
    }
  };

  const endpoints = mode === 'openai' ? ['/v1/chat/completions', '/v1/models']
    : mode === 'anthropic' ? ['/v1/messages']
    : ['/v1/chat/completions', '/v1/models', '/v1/messages'];

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">API Server</h2>
      <p className="text-sm text-gray-500 mb-6">Run a local OpenAI / Anthropic-compatible API endpoint for your downloaded models.</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-medium mb-3">Configuration</h3>

          <p className="text-xs text-gray-500 mb-4">
            The API server is a gateway. Each served model carries its own profile, and clients choose the model per request.
          </p>

          <div className="mb-4">
            <label className="block text-xs text-gray-400 mb-1">Port</label>
            <input
              type="number"
              value={port}
              onChange={(e) => setPort(Number(e.target.value))}
              disabled={status === 'running'}
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm outline-none focus:border-blue-500 disabled:opacity-50"
            />
          </div>

          <div className="mb-4">
            <label className="block text-xs text-gray-400 mb-1">API Mode</label>
            <div className="flex gap-2">
              {(['openai', 'anthropic', 'both'] as ApiMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  disabled={status === 'running'}
                  className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                    mode === m
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                  } disabled:opacity-50`}
                >
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {status === 'running' ? (
            <button onClick={handleStop} className="w-full px-4 py-2 bg-red-700 rounded hover:bg-red-600 text-sm font-medium">
              Stop Server
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={status === 'starting'}
              className="w-full px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 text-sm font-medium disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {status === 'starting' ? <><Spinner className="w-4 h-4" /> Starting...</> : 'Start Gateway'}
            </button>
          )}

          {error && <p className="text-red-400 text-xs mt-2">{error}</p>}
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-medium mb-3">Status</h3>

          <div className="flex items-center gap-2 mb-4">
            <span className={`w-2.5 h-2.5 rounded-full ${
              status === 'running' ? 'bg-green-500' : status === 'starting' ? 'bg-yellow-500' : 'bg-gray-500'
            }`} />
            <span className="text-sm capitalize">{status}</span>
          </div>

          {status === 'running' && (
            <>
              <div className="relative border-b border-gray-700 pb-4 mb-4">
                <div
                  aria-hidden={!showResourceWarning}
                  className={`absolute inset-0 z-10 flex min-h-full items-start rounded border border-yellow-700 bg-yellow-950/95 px-3 py-3 text-xs text-yellow-200 shadow-lg transition-opacity duration-500 ease-out ${
                    showResourceWarning ? 'opacity-100' : 'pointer-events-none opacity-0'
                  }`}
                >
                    Resource pressure detected. New local requests are being delayed until capacity returns.
                    <div className="mt-1 text-yellow-300/80">
                      Available: {resourceState.available_mb ?? '?'} MB · Reserved: {resourceState.reserved_mb ?? 0} MB ·
                      Safety reserve: {resourceState.safety_reserve_mb ?? '?'} MB
                    </div>
                </div>
                <h4 className="text-xs font-medium text-gray-400 mb-2">Models in use</h4>
                {activeModels.length > 0 ? (
                  <div className="space-y-2">
                    {activeModels.map((model) => {
                      const request = model.current_request!;
                      return (
                        <div key={model.id} className="rounded bg-gray-900 px-3 py-2 text-xs">
                          <div className="flex items-center justify-between gap-3">
                            <span className="text-gray-200 truncate">
                              {model.profile?.served_id || model.served_id || model.id}
                            </span>
                            <span className="text-green-300 shrink-0">{request.status}</span>
                          </div>
                          <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-gray-500">
                            <span>{model.source || 'local'} · {model.driver || model.profile?.driver || 'default'}</span>
                            <span>{request.streamed_token_estimate} estimated tokens</span>
                            <span>
                              {typeof request.first_token_latency_ms === 'number'
                                ? `${formatDuration(request.first_token_latency_ms)} TTFT`
                                : 'waiting for first token'}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="rounded bg-gray-900 px-3 py-3 text-xs text-gray-600">
                    No model in use
                  </div>
                )}
              </div>
              <div className="text-xs text-gray-400 space-y-1 mb-4">
                <div className="flex justify-between">
                  <span>Served Models</span>
                  <span className="text-gray-300">{servedModels.length}</span>
                </div>
                <div className="flex justify-between">
                  <span>GGUF Backend</span>
                  <span className="text-gray-300 font-mono uppercase">
                    {config.gguf_backend || config.device || '?'}{config.gpu_index ? ` (GPU ${config.gpu_index})` : ''}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>OpenVINO Device</span>
                  <span className="text-gray-300 font-mono uppercase">
                    {config.openvino_device || 'NPU'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Port</span>
                  <span className="text-gray-300">{config.port}</span>
                </div>
                <div className="flex justify-between">
                  <span>Mode</span>
                  <span className="text-gray-300 capitalize">{config.mode}</span>
                </div>
              </div>

              {runtime?.apiKey && (
                <div className="border-t border-gray-700 pt-3 mt-3">
                  <h4 className="text-xs font-medium text-gray-400 mb-2">LAN LLM API access</h4>
                  <div className="text-[11px] text-gray-500 mb-2">
                    Only the authenticated LLM endpoints are exposed. Dashboard controls remain local to this machine.
                  </div>
                  <div className="flex items-center gap-2">
                    <code className="text-[11px] bg-gray-900 px-2 py-1 rounded text-green-300 flex-1 truncate">
                      {runtime.publicGatewayUrl}/v1
                    </code>
                    <button
                      onClick={() => navigator.clipboard.writeText(`${runtime.publicGatewayUrl}/v1`)}
                      className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-400"
                    >
                      Copy URL
                    </button>
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    <code className="text-[11px] bg-gray-900 px-2 py-1 rounded text-blue-300 flex-1 truncate">
                      {runtime.apiKey}
                    </code>
                    <button
                      onClick={() => navigator.clipboard.writeText(runtime.apiKey)}
                      className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-400"
                    >
                      Copy key
                    </button>
                  </div>
                </div>
              )}

              <div className="border-t border-gray-700 pt-3 mt-3">
                <h4 className="text-xs font-medium text-gray-400 mb-2">Endpoints</h4>
                {endpoints.map((ep) => (
                  <div key={ep} className="flex items-center gap-2 mb-1">
                    <code className="text-xs bg-gray-900 px-2 py-0.5 rounded text-green-400 flex-1">
                      {gatewayBase}{ep}
                    </code>
                    <button
                      onClick={() => navigator.clipboard.writeText(`${gatewayBase}${ep}`)}
                      className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-400"
                    >
                      Copy
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      {status === 'running' && (
        <RequestTerminal entries={requestLog} />
      )}

      {status === 'running' && servedModels.length > 0 && (
        <div className="mt-6 bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between gap-3 mb-3">
            <h3 className="text-sm font-medium">Model Catalog</h3>
            <span className="text-xs text-gray-500">{servedModels.length} served</span>
          </div>
          {profileError && <p className="mb-3 text-xs text-red-400">{profileError}</p>}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
            {servedModels.map((m) => {
              const key = profileKey(m);
              const draft = editingProfiles[key];
              const isEditing = Boolean(draft);
              const streamRequest = m.current_request || m.last_request;
              const isStreaming = Boolean(m.current_request);
              return (
                <div key={key} className="bg-gray-900 rounded px-2 py-2 text-xs">
                  <div className="flex items-start gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="text-gray-200 truncate">{m.profile?.served_id || m.served_id || m.id}</div>
                      <div className="mt-0.5 text-[11px] text-gray-500 truncate">
                        File: {modelFileLabel(m.profile?.model_file || m.model_file)}
                      </div>
                    </div>
                    <span className="text-[10px] uppercase text-gray-500 shrink-0">{m.source || 'local'}</span>
                  </div>

                  {isEditing ? (
                    <div className="mt-3 space-y-2">
                      <ProfileInput label="Served ID" value={draft.served_id} onChange={(value) => updateDraft(key, { served_id: value })} />
                      <ProfileInput label="Driver" value={draft.driver} onChange={(value) => updateDraft(key, { driver: value })} />
                      <ProfileInput label="Model File" value={draft.model_file} onChange={(value) => updateDraft(key, { model_file: value })} />
                      <div className="grid grid-cols-2 gap-2">
                        <ProfileNumberInput
                          label="Default Tokens"
                          value={draft.max_tokens_default}
                          onChange={(value) => updateDraft(key, { max_tokens_default: value })}
                        />
                        <ProfileNumberInput
                          label="Default Temp"
                          value={draft.temperature_default}
                          step="0.1"
                          onChange={(value) => updateDraft(key, { temperature_default: value })}
                        />
                      </div>
                      <div className="flex flex-wrap gap-3 text-[11px] text-gray-400">
                        <label className="flex items-center gap-1">
                          <input type="checkbox" checked={draft.mtp} onChange={(e) => updateDraft(key, { mtp: e.target.checked })} className="accent-purple-500" />
                          MTP
                        </label>
                        <label className="flex items-center gap-1">
                          <input type="checkbox" checked={draft.dspark} onChange={(e) => updateDraft(key, { dspark: e.target.checked })} className="accent-green-500" />
                          dSpark
                        </label>
                      </div>
                      <div className="flex gap-2 pt-1">
                        <button onClick={() => saveProfile(m)} className="px-2 py-1 rounded bg-blue-700 hover:bg-blue-600 text-[11px]">
                          Save
                        </button>
                        <button onClick={() => cancelEditingProfile(key)} className="px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-[11px]">
                          Cancel
                        </button>
                        <button onClick={() => resetProfile(m)} className="ml-auto px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-[11px] text-gray-400">
                          Reset
                        </button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="mt-2 flex flex-wrap gap-1">
                        <ProfileBadge label={(m.runtime || m.format || 'model').toUpperCase()} tone="blue" />
                        <ProfileBadge label={`Driver: ${(m.driver || m.profile?.driver || 'request').toUpperCase()}`} tone="gray" />
                        <ProfileBadge
                          label={mtpLabel(m)}
                          tone={(m.profile?.mtp ?? m.mtp) ? 'purple' : (m.profile?.is_mtp_head ?? m.is_mtp_head) ? 'yellow' : 'gray'}
                        />
                        <ProfileBadge label={(m.profile?.dspark ?? m.dspark) ? 'dSpark' : 'No dSpark'} tone={(m.profile?.dspark ?? m.dspark) ? 'green' : 'gray'} />
                      </div>
                      <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] text-gray-500">
                        <span>Served ID</span>
                        <span className="text-gray-400 truncate">{m.profile?.served_id || m.served_id || m.id}</span>
                        <span>API Mode</span>
                        <span className="text-gray-400 uppercase">{m.profile?.serving_config?.api_mode || config.mode || 'openai'}</span>
                        <span>Port</span>
                        <span className="text-gray-400">{m.profile?.serving_config?.port || config.port}</span>
                        <span>GPU</span>
                        <span className="text-gray-400">{m.profile?.serving_config?.gpu_index || 'default'}</span>
                        {(m.profile?.mtp_head_path || m.mtp_head_path) && (
                          <>
                            <span>MTP Head</span>
                            <span className="text-gray-400 truncate" title={m.profile?.mtp_head_path || m.mtp_head_path || ''}>
                              {modelFileLabel(m.profile?.mtp_head_path || m.mtp_head_path || '')}
                            </span>
                          </>
                        )}
                        <span>{isStreaming ? 'Current Stream' : 'Last Stream'}</span>
                        <span className={isStreaming ? 'text-green-300' : 'text-gray-400'}>
                          {streamRequest
                            ? `${streamRequest.streamed_token_estimate} est. model tokens`
                            : 'none'}
                        </span>
                        {streamRequest && (
                          <>
                            <span>Stream Status</span>
                            <span className={isStreaming ? 'text-green-300' : 'text-gray-400'}>
                              {streamRequest.status}
                              {isStreaming && streamRequest.updated_at_ms ? `, ${formatStreamAge(streamRequest.updated_at_ms)}` : ''}
                            </span>
                            <span>First Token</span>
                            <span className="text-gray-400">
                              {typeof streamRequest.first_token_latency_ms === 'number'
                                ? `${formatDuration(streamRequest.first_token_latency_ms)} TTFT`
                                : 'pending'}
                            </span>
                          </>
                        )}
                      </div>
                      <button onClick={() => startEditingProfile(m)} className="mt-3 px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-[11px] text-gray-300">
                        Edit Configuration
                      </button>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {status === 'running' && (
        <div className="mt-6 bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-medium mb-3">Quick Test</h3>
          <QuickTest
            dashboardBase={dashboardBase}
            mode={mode}
            models={servedModels.length > 0 ? servedModels : models.map((m) => ({ id: m.id, format: m.format }))}
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
          />
        </div>
      )}
    </div>
  );
}

function RequestTerminal({ entries }: { entries: RequestLogEntry[] }) {
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [entries.length]);

  return (
    <div className="mt-6 bg-gray-950 rounded-lg border border-gray-700 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <h3 className="text-sm font-medium">API Request Terminal</h3>
        <span className="text-[11px] text-gray-500">{entries.length} recent requests</span>
      </div>
      <div ref={terminalRef} className="h-48 overflow-y-auto p-3 font-mono text-[11px] leading-5">
        {entries.length === 0 ? (
          <div className="text-gray-600">Waiting for API requests...</div>
        ) : (
          entries.map((entry, index) => (
            <div key={`${entry.at_ms}-${index}`} className="whitespace-nowrap">
              <span className="text-gray-600">{formatLogTime(entry.at_ms)}</span>
              <span className={entry.status >= 400 ? 'text-red-400' : 'text-green-400'}>
                {' '}{entry.status}
              </span>
              <span className="text-blue-300"> {entry.method}</span>
              <span className="text-gray-300"> {entry.path}</span>
              <span className="text-gray-600"> {entry.duration_ms}ms from {entry.client}</span>
              {entry.model_id && (
                <span className="text-purple-300">
                  {' '}model={entry.model_id} [{entry.model_source || 'unknown'}]
                  {entry.remote_pc && ` pc=${entry.remote_pc}`}
                  {entry.remote_model_id && ` remote=${entry.remote_model_id}`}
                  {` tokens=${entry.streamed_token_estimate ?? 0}`}
                </span>
              )}
              {entry.detail && <span className="text-yellow-400"> — {entry.detail}</span>}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function ProfileBadge({ label, tone }: { label: string; tone: 'blue' | 'gray' | 'green' | 'purple' | 'yellow' }) {
  const classes = {
    blue: 'bg-blue-900/50 text-blue-200',
    gray: 'bg-gray-800 text-gray-400',
    green: 'bg-green-900/50 text-green-200',
    purple: 'bg-purple-900/50 text-purple-200',
    yellow: 'bg-yellow-900/50 text-yellow-200',
  };
  return <span className={`rounded px-1.5 py-0.5 text-[10px] ${classes[tone]}`}>{label}</span>;
}

function mtpLabel(model: ServedModel) {
  if (model.profile?.mtp ?? model.mtp) return 'MTP ready';
  if (model.profile?.is_mtp_head ?? model.is_mtp_head) return 'MTP head';
  return 'No MTP';
}

function modelFileLabel(path?: string) {
  if (!path) return 'unknown';
  const parts = path.split('/');
  return parts[parts.length - 1] || path;
}

function formatStreamAge(updatedAtMs: number) {
  const ageSeconds = Math.max(0, Math.round((Date.now() - updatedAtMs) / 1000));
  return ageSeconds <= 1 ? 'live' : `${ageSeconds}s ago`;
}

function formatDuration(ms: number) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatLogTime(ms: number) {
  return new Date(ms).toLocaleTimeString([], { hour12: false });
}

function ProfileInput({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="block">
      <span className="block text-[11px] text-gray-500 mb-1">{label}</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-xs outline-none focus:border-blue-500"
      />
    </label>
  );
}

function ProfileNumberInput({
  label,
  value,
  step = '1',
  onChange,
}: {
  label: string;
  value: number;
  step?: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span className="block text-[11px] text-gray-500 mb-1">{label}</span>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-xs outline-none focus:border-blue-500"
      />
    </label>
  );
}

function QuickTest({
  dashboardBase,
  mode,
  models,
  selectedModel,
  onModelChange,
}: {
  dashboardBase: string;
  mode: ApiMode;
  models: ServedModel[];
  selectedModel: string;
  onModelChange: (id: string) => void;
}) {
  const [prompt, setPrompt] = useState('What is the meaning of life?');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const requestBase = window.location.protocol === 'file:' ? dashboardBase : '';

  const testOpenAI = async () => {
    setLoading(true);
    setResponse('');
    try {
      const payload = {
        model: selectedModel,
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 128,
        format: 'openai' as const,
      };
      let data;
      if (window.avalon?.quickTest) {
        data = await window.avalon.quickTest(payload);
      } else {
        const r = await fetch(`${requestBase}/api/api-server/quick-test`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!r.ok) { throw new Error(await r.text()); }
        data = await r.json();
      }
      setResponse(data.choices?.[0]?.message?.content || JSON.stringify(data));
    } catch (e: any) {
      setResponse(`Error: ${e.message}`);
    }
    setLoading(false);
  };

  const testAnthropic = async () => {
    setLoading(true);
    setResponse('');
    try {
      const payload = {
        model: selectedModel,
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 128,
        format: 'anthropic' as const,
      };
      let data;
      if (window.avalon?.quickTest) {
        data = await window.avalon.quickTest(payload);
      } else {
        const r = await fetch(`${requestBase}/api/api-server/quick-test`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!r.ok) { throw new Error(await r.text()); }
        data = await r.json();
      }
      setResponse(data.content?.[0]?.text || JSON.stringify(data));
    } catch (e: any) {
      setResponse(`Error: ${e.message}`);
    }
    setLoading(false);
  };

  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1">Model</label>
      <select
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm outline-none focus:border-blue-500 mb-3"
      >
        <option value="">Select a model...</option>
        {models.map((m) => (
          <option key={m.id} value={m.id}>{m.id}</option>
        ))}
      </select>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={2}
        className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm outline-none focus:border-blue-500 resize-none"
      />
      <div className="flex gap-2 mt-2">
        {(mode === 'openai' || mode === 'both') && (
          <button onClick={testOpenAI} disabled={loading || !selectedModel}
            className="px-3 py-1.5 text-xs bg-green-700 rounded hover:bg-green-600 disabled:opacity-50">
            {loading ? 'Testing...' : 'Test OpenAI'}
          </button>
        )}
        {(mode === 'anthropic' || mode === 'both') && (
          <button onClick={testAnthropic} disabled={loading || !selectedModel}
            className="px-3 py-1.5 text-xs bg-purple-700 rounded hover:bg-purple-600 disabled:opacity-50">
            {loading ? 'Testing...' : 'Test Anthropic'}
          </button>
        )}
      </div>
      {response && (
        <pre className="mt-3 bg-gray-900 rounded p-3 text-xs text-gray-300 whitespace-pre-wrap max-h-48 overflow-y-auto">
          {response}
        </pre>
      )}
    </div>
  );
}
