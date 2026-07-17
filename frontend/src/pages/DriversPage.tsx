import { useEffect, useState, useRef } from 'react';
import { api } from '../api/client';
import type { DriverRelease, LocalDriver, ActiveDriver, DownloadProgress } from '../types';
import Spinner from '../components/Spinner';

function useDownloadProgress() {
  const [dl, setDl] = useState<Record<string, DownloadProgress>>({});
  const timers = useRef<Record<string, number>>({});

  const poll = (id: string) => {
    const fn = async () => {
      try {
        const res = await fetch(`/api/downloads/progress/${id}`);
        const p: DownloadProgress = await res.json();
        setDl((prev) => ({ ...prev, [id]: p }));
        if (p.status !== 'done' && p.status !== 'error') {
          timers.current[id] = window.setTimeout(fn, 500);
        } else {
          delete timers.current[id];
        }
      } catch { delete timers.current[id]; }
    };
    fn();
  };

  const start = (id: string) => {
    setDl((prev) => ({ ...prev, [id]: { status: 'starting', percent: 0, stage: '' } }));
    poll(id);
  };

  const clear = (id: string) => {
    if (timers.current[id]) clearTimeout(timers.current[id]);
    delete timers.current[id];
    setDl((prev) => { const n = { ...prev }; delete n[id]; return n; });
  };

  useEffect(() => () => { Object.values(timers.current).forEach(clearTimeout); }, []);

  return { downloads: dl, start, clear };
}

export default function DriversPage() {
  const [releases, setReleases] = useState<DriverRelease[]>([]);
  const [local, setLocal] = useState<LocalDriver[]>([]);
  const [active, setActive] = useState<ActiveDriver[]>([]);
  const [driverUpdates, setDriverUpdates] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const dl = useDownloadProgress();

  const load = () => {
    setLoading(true);
    Promise.all([api.listReleases(), api.listDrivers(), api.listDriverUpdates()])
      .then(([r, d, u]) => {
        setReleases(r.releases);
        setLocal(d.local);
        setActive(d.active);
        setDriverUpdates(u.updates);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const [keyToDlId, setKeyToDlId] = useState<Record<string, string>>({});

  const handleDownload = async (tag: string, backend: string) => {
    const key = `${backend}_${tag}`;
    setBusy((prev) => ({ ...prev, [key]: true }));
    try {
      const res = await api.downloadDriver(tag, backend);
      setKeyToDlId((prev) => ({ ...prev, [key]: res.download_id }));
      dl.start(res.download_id);
    } catch (e: any) {
      alert(e.message);
    } finally {
      setBusy((prev) => ({ ...prev, [key]: false }));
    }
  };

  useEffect(() => {
    const entries = Object.entries(dl.downloads);
    const done = entries.filter(([, p]) => p.status === 'done' || p.status === 'error');
    if (done.length > 0) {
      load();
      done.forEach(([id]) => {
        setTimeout(() => {
          dl.clear(id);
          setKeyToDlId((prev) => {
            const n = { ...prev };
            Object.entries(n).forEach(([k, v]) => { if (v === id) delete n[k]; });
            return n;
          });
          setBusy((prev) => {
            const n = { ...prev };
            Object.keys(n).forEach((k) => { delete n[k]; });
            return n;
          });
        }, 2000);
      });
    }
  }, [dl.downloads]);

  const handleToggleActive = async (id: string, currentlyActive: boolean) => {
    try {
      await api.setDriverActive(id, !currentlyActive);
      load();
    } catch (e: any) {
      alert(e.message);
    }
  };

  const handleRemove = async (id: string) => {
    if (!confirm('Remove this driver?')) return;
    try {
      await api.removeDriver(id);
      load();
    } catch (e: any) {
      alert(e.message);
    }
  };

  const handleUpdate = async (tag: string, backend: string, oldId: string) => {
    if (!confirm(`Update to ${tag}? The old driver will be removed.`)) return;
    const key = `${backend}_${tag}`;
    setBusy((prev) => ({ ...prev, [key]: true }));
    try {
      const res = await api.downloadDriver(tag, backend);
      setKeyToDlId((prev) => ({ ...prev, [key]: res.download_id }));
      dl.start(res.download_id);
      await api.removeDriver(oldId);
      load();
    } catch (e: any) {
      alert(e.message);
      load();
    }
  };

  const backends = ['cuda', 'vulkan', 'sycl', 'openvino', 'directml', 'hip', 'cpu', 'metal'];

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">Drivers</h2>
        <button onClick={load} className="text-sm text-blue-400 hover:text-blue-300 disabled:opacity-50 flex items-center gap-1" disabled={loading}>
          {loading ? <><Spinner className="w-3 h-3" /> Refreshing</> : 'Refresh'}
        </button>
      </div>

      {driverUpdates.map((u, i) => (
        <div key={i} className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-3 mb-4 text-sm flex items-center justify-between">
          <div>
            <span className="text-yellow-300 font-medium">GPU driver update available</span>
            <span className="text-gray-400 ml-2">{u.device_name}: {u.installed} → {u.latest}</span>
          </div>
          <a href={u.download_url} target="_blank" rel="noopener noreferrer"
            className="px-3 py-1 bg-yellow-700 rounded hover:bg-yellow-600 text-xs whitespace-nowrap ml-3">
            Download
          </a>
        </div>
      ))}

      {active.length > 0 && (
        <div className="bg-green-900/30 border border-green-700 rounded-lg p-3 mb-6 text-sm">
          Active drivers: {active.map((a) => (
            <span key={a.id} className="inline-block bg-green-800 rounded px-2 py-0.5 ml-1 text-xs">
              {a.backend}@{a.tag}
            </span>
          ))}
        </div>
      )}

      <div className="mb-8">
        <h3 className="font-semibold mb-3">Local Drivers</h3>
        {local.length === 0 ? (
          <p className="text-gray-500 text-sm">No drivers downloaded. Download one from the releases below.</p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {local.map((d) => (
              <div key={d.id} className={`bg-gray-900 rounded-lg p-4 border ${d.active ? 'border-green-600' : 'border-gray-800'}`}>
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <span className="font-medium">{d.id}</span>
                    <span className="ml-2 text-xs text-gray-500">{d.backend}</span>
                  </div>
                  <span className={`px-2 py-0.5 rounded text-xs ${d.active ? 'bg-green-900 text-green-300' : 'bg-gray-700 text-gray-400'}`}>
                    {d.active ? 'active' : 'inactive'}
                  </span>
                </div>
                <div className="text-xs text-gray-500 mb-3">
                  {d.tag} · {(d.size / 1024 / 1024).toFixed(1)} MB
                  {d.llama_bench_path ? ' · bench ready' : ' · no bench'}
                  {d.source_label && <span className="ml-2 text-yellow-500">from {d.source_label}</span>}
                </div>
                {(() => { const dlId = keyToDlId[d.id]; const prog = dlId ? dl.downloads[dlId] : undefined; return prog ? (
                  <div className="mt-2">
                    <div className="bg-gray-700 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${prog.status === 'error' ? 'bg-red-500' : 'bg-blue-500'}`}
                        style={{ width: `${prog.percent}%` }}
                      />
                    </div>
                    <div className="text-[10px] text-gray-500 mt-0.5">{prog.stage}</div>
                  </div>
                ) : (
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleToggleActive(d.id, d.active)}
                      className={`px-3 py-1 text-xs rounded ${
                        d.active
                          ? 'bg-yellow-700 hover:bg-yellow-600'
                          : 'bg-blue-600 hover:bg-blue-500'
                      }`}
                    >
                      {d.active ? 'Deactivate' : 'Activate'}
                    </button>
                    <button onClick={() => handleRemove(d.id)} className="px-3 py-1 text-xs bg-red-700 rounded hover:bg-red-600">
                      Remove
                    </button>
                    {d.has_update && d.latest_tag && (
                      <button
                        onClick={() => handleUpdate(d.latest_tag!, d.backend, d.id)}
                        className="px-3 py-1 text-xs bg-green-700 rounded hover:bg-green-600 disabled:opacity-50 animate-pulse"
                      >
                        Update to {d.latest_tag}
                      </button>
                    )}
                  </div>
                )})()}
              </div>
            ))}
          </div>
        )}
      </div>

      <div>
        <h3 className="font-semibold mb-3">Releases</h3>
        {loading ? (
          <div className="flex items-center gap-2 text-gray-500 text-sm py-4"><Spinner className="w-4 h-4" /> Loading releases...</div>
        ) : (
          <>
            {['official', 'community'].map((group) => {
              const groupReleases = releases.filter((r) => r.source === group);
              if (groupReleases.length === 0) return null;
              return (
                <div key={group} className="mb-6">
                  <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3 border-b border-gray-800 pb-1">
                    {group === 'official' ? 'Official (ggml-org/llama.cpp)' : 'Community Forks'}
                  </h4>
                  <div className="space-y-3">
                    {groupReleases.map((rel) => {
                      const repoLink = rel.source_repo
                        ? `https://github.com/${rel.source_repo}/releases/tag/${rel.tag_name}`
                        : `https://github.com/ggml-org/llama.cpp/releases/tag/${rel.tag_name}`;
                      return (
                        <div key={rel.id} className="bg-gray-900 rounded-lg p-4 border border-gray-800">
                          <div className="flex items-start justify-between mb-2">
                            <div>
                              <a
                                href={repoLink}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="font-medium hover:text-blue-400"
                              >
                                {rel.tag_name}
                              </a>
                              <p className="text-xs text-gray-500 mt-0.5">{rel.name}</p>
                              {rel.source_label && (
                                <p className="text-xs text-yellow-500 mt-0.5">{rel.source_label}</p>
                              )}
                            </div>
                            <span className="text-xs text-gray-500">{new Date(rel.published_at).toLocaleDateString()}</span>
                          </div>
                          <div className="flex flex-wrap gap-2 mt-2">
                            {backends.map((b) => {
                              const hasAsset = rel.backends
                                ? rel.backends.includes(b)
                                : rel.assets.some((a) => {
                                    const n = a.name;
                                    if (b === 'metal') return /llama-b.*-bin-macos-.*\.tar\.gz/.test(n);
                                    if (b === 'openvino') return /llama-b.*-bin-win-openvino-.*-x64\.zip/.test(n);
                                    if (b === 'directml') return /llama-cpp-directml.*\.zip/.test(n);
                                    if (b === 'hip') return /llama-b.*-bin-win-hip-.*\.zip/.test(n);
                                    return new RegExp(`llama-b.*-bin-win-${b}.*-x64\\.zip`).test(n);
                                  });
                              const already = local.find((l) => l.tag === rel.tag_name && l.backend === b);
                              const dk = `${b}_${rel.tag_name}`;
                              const dlId = keyToDlId[dk];
                              const prog = dlId ? dl.downloads[dlId] : undefined;
                              if (prog) {
                                return (
                                  <div key={b} className="w-24">
                                    <div className="bg-gray-700 rounded-full h-1.5 overflow-hidden">
                                      <div className={`h-full rounded-full ${prog.status === 'error' ? 'bg-red-500' : 'bg-blue-500'}`} style={{ width: `${prog.percent}%` }} />
                                    </div>
                                    <div className="text-[10px] text-gray-500 text-center">{prog.stage}</div>
                                  </div>
                                );
                              }
                              return (
                                <button
                                  key={b}
                                  disabled={!hasAsset || !!already || !!busy[dk]}
                                  onClick={() => handleDownload(rel.tag_name, b)}
                                  className={`px-3 py-1 text-xs rounded ${
                                    already
                                      ? 'bg-gray-700 text-gray-500 cursor-default'
                                      : hasAsset
                                      ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                                      : 'bg-gray-800 text-gray-600 cursor-not-allowed'
                                  }`}
                                >
                                  {b} {already ? '✓' : hasAsset ? '' : '—'}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </>
        )}
      </div>
    </div>
  );
}
