import { useEffect, useState, useRef } from 'react';
import { api } from '../api/client';
import type { HFModel, LocalModel, ModelFile, DownloadProgress } from '../types';
import Spinner from '../components/Spinner';

function useDownloadProgress() {
  const [dl, setDl] = useState<Record<string, DownloadProgress>>({});
  const timers = useRef<Record<string, number>>({});

  const poll = (id: string) => {
    const fn = async () => {
      try {
        const res = await fetch(`/api/downloads/progress/${id}`);
        if (!res.ok) { delete timers.current[id]; return; }
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

const RECENT_KEY = 'model_search_history';

export default function ModelsPage() {
  const [query, setQuery] = useState(() => localStorage.getItem(RECENT_KEY) || 'personaplex');
  const [searchResults, setSearchResults] = useState<HFModel[]>(() => {
    const saved = sessionStorage.getItem('hf_search_results');
    return saved ? JSON.parse(saved) : [];
  });
  const [modelFormat, setModelFormat] = useState(() => sessionStorage.getItem('model_format') || 'gguf');
  const [local, setLocal] = useState<LocalModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [busyDl, setBusyDl] = useState<Record<string, boolean>>({});
  const [showHistory, setShowHistory] = useState(false);
  const dl = useDownloadProgress();

  const loadLocal = () => {
    api.listLocalModels()
      .then((d) => setLocal(d.models))
      .catch(console.error);
  };

  useEffect(() => { loadLocal(); }, []);

  useEffect(() => {
    if (searchResults.length > 0) {
      sessionStorage.setItem('hf_search_results', JSON.stringify(searchResults));
    }
  }, [searchResults]);

  useEffect(() => {
    if (!searchResults.length && query.trim()) {
      handleSearch();
    }
  }, []);

  const getHistory = (): string[] => {
    try { return JSON.parse(localStorage.getItem(RECENT_KEY + '_list') || '[]'); } catch { return []; }
  };
  const addHistory = (q: string) => {
    const list = [q, ...getHistory().filter((x) => x !== q)].slice(0, 10);
    localStorage.setItem(RECENT_KEY + '_list', JSON.stringify(list));
  };

  const handleSearch = async (fmt?: string) => {
    const f = fmt ?? modelFormat;
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setModelFormat(f);
    sessionStorage.setItem('model_format', f);
    localStorage.setItem(RECENT_KEY, q);
    addHistory(q);
    try {
      const d = await api.searchModels(q, 20, f);
      setSearchResults(d.models);
    } catch (e: any) {
      alert(e.message);
    } finally {
      setLoading(false);
      setShowHistory(false);
    }
  };

  const [keyToDlId, setKeyToDlId] = useState<Record<string, string>>({});

  const handleDownload = async (repoId: string, filename: string) => {
    const key = `${repoId}/${filename}`;
    setBusyDl((prev) => ({ ...prev, [key]: true }));
    try {
      const res = await api.downloadModel(repoId, filename);
      setKeyToDlId((prev) => ({ ...prev, [key]: res.download_id }));
      dl.start(res.download_id);
      setBusyDl((prev) => ({ ...prev, [key]: false }));
    } catch (e: any) {
      alert(e.message);
    }
  };

  const handleRemove = async (id: string) => {
    if (!confirm('Remove this model?')) return;
    try {
      await api.removeModel(id);
      loadLocal();
    } catch (e: any) {
      alert(e.message);
    }
  };

  useEffect(() => {
    const entries = Object.entries(dl.downloads);
    const done = entries.filter(([, p]) => p.status === 'done' || p.status === 'error');
    if (done.length > 0) {
      loadLocal();
      done.forEach(([id]) => {
        setTimeout(() => {
          dl.clear(id);
          setKeyToDlId((prev) => {
            const n = { ...prev };
            Object.entries(n).forEach(([k, v]) => { if (v === id) delete n[k]; });
            return n;
          });
          setBusyDl((prev) => {
            const n = { ...prev };
            Object.keys(n).forEach((k) => { if (Object.values(keyToDlId).includes(id)) delete n[k]; });
            return n;
          });
        }, 2000);
      });
    }
  }, [dl.downloads]);

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Models</h2>

      <div className="bg-gray-900 rounded-lg p-4 mb-6">
        <h3 className="font-semibold mb-3">Local Models</h3>
        {local.length === 0 ? (
          <p className="text-gray-500 text-sm">No models downloaded. Search and download from HuggingFace below.</p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {local.map((m) => (
              <div key={m.id} className="bg-gray-800 rounded p-3 border border-gray-700">
                <div className="font-medium text-sm truncate">{m.repo_id}</div>
                <div className="text-xs text-gray-500 mt-1">
                  {m.files.join(', ')} · {(m.size / 1024 / 1024 / 1024).toFixed(2)} GB
                </div>
                <button onClick={() => handleRemove(m.id)} className="mt-2 px-2 py-0.5 text-xs bg-red-700 rounded hover:bg-red-600">
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="bg-gray-900 rounded-lg p-4">
        <h3 className="font-semibold mb-3">Search HuggingFace</h3>
          <div className="flex gap-2 mb-2 relative">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              onFocus={() => setShowHistory(true)}
              onBlur={() => setTimeout(() => setShowHistory(false), 200)}
              placeholder="Search models..."
              className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm outline-none focus:border-blue-500"
            />
            <button onClick={() => handleSearch()} disabled={loading} className="px-4 py-2 bg-blue-600 rounded text-sm hover:bg-blue-500 disabled:opacity-50 flex items-center gap-1.5">
              {loading ? <><Spinner className="w-3.5 h-3.5" /> Searching</> : 'Search'}
            </button>
            {showHistory && getHistory().length > 0 && (
              <div className="absolute top-full left-0 mt-1 bg-gray-800 border border-gray-700 rounded shadow-lg z-10 w-64">
                {getHistory().map((h) => (
                  <button
                    key={h}
                    className="block w-full text-left px-3 py-1.5 text-xs hover:bg-gray-700"
                    onMouseDown={() => { setQuery(h); handleSearch(); }}
                  >
                    {h}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="flex gap-2 mb-3">
            <button
              onClick={() => handleSearch('gguf')}
              className={`px-3 py-1 text-xs rounded ${modelFormat === 'gguf' ? 'bg-blue-700 text-white' : 'bg-gray-700 text-gray-400 hover:text-white'}`}
            >GGUF (llama.cpp)</button>
            <button
              onClick={() => handleSearch('openvino')}
              className={`px-3 py-1 text-xs rounded ${modelFormat === 'openvino' ? 'bg-blue-700 text-white' : 'bg-gray-700 text-gray-400 hover:text-white'}`}
            >OpenVINO (NPU)</button>
          </div>

        {searchResults.length > 0 && (
          <div className="space-y-2">
            {searchResults.map((m) => (
              <ModelRow key={m.id} model={m} onDownload={handleDownload} busyDl={busyDl} downloads={dl.downloads} keyToDlId={keyToDlId} format={modelFormat} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function fmtSize(bytes: number): string {
  if (bytes === 0) return '?';
  const gb = bytes / 1024 / 1024 / 1024;
  if (gb >= 1) return gb.toFixed(2) + ' GB';
  const mb = bytes / 1024 / 1024;
  return mb.toFixed(0) + ' MB';
}

type SortKey = 'name' | 'size';
type SortDir = 'asc' | 'desc';

function ModelRow({ model, onDownload, busyDl, downloads, keyToDlId, format }: { model: HFModel; onDownload: (repo: string, file: string) => void; busyDl: Record<string, boolean>; downloads: Record<string, DownloadProgress>; keyToDlId: Record<string, string>; format: string }) {
  const [files, setFiles] = useState<ModelFile[]>([]);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>('name');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const [filter, setFilter] = useState('');

  const loadFiles = async () => {
    setLoadingFiles(true);
    try {
      const d = await api.listModelFiles(model.id);
      setFiles(d.files);
    } catch { setFiles([]); }
    finally { setLoadingFiles(false); }
  };

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const filtered = files
    .filter((f) => !filter || f.name.toLowerCase().includes(filter.toLowerCase()))
    .sort((a, b) => {
      const cmp = sortKey === 'name'
        ? a.name.localeCompare(b.name)
        : a.size - b.size;
      return sortDir === 'asc' ? cmp : -cmp;
    });

  const sortArrow = (key: SortKey) => {
    if (sortKey !== key) return '';
    return sortDir === 'asc' ? ' \u25B2' : ' \u25BC';
  };

  return (
    <div className="bg-gray-800 rounded p-3 border border-gray-700">
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm">{model.id}</span>
            {format === 'openvino' && <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-900 text-cyan-300">OpenVINO</span>}
          </div>
          <div className="text-xs text-gray-500 mt-0.5">
            {model.downloads.toLocaleString()} downloads · {model.likes} likes
          </div>
        </div>
        {format === 'openvino' ? (
          <OpenvinoDownloadButton modelId={model.id} />
        ) : (
          <button onClick={loadFiles} disabled={loadingFiles} className="px-2 py-0.5 text-xs bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50 flex items-center gap-1">
            {loadingFiles ? <><Spinner className="w-3 h-3" /> Loading...</> : files.length > 0 ? 'Refresh' : 'List files'}
          </button>
        )}
      </div>
      {files.length > 0 && (
        <div className="mt-3">
          <div className="flex gap-2 mb-2">
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter files..."
              className="flex-1 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs outline-none focus:border-blue-500"
            />
            <button onClick={() => toggleSort('name')} className="px-2 py-1 text-xs bg-gray-700 rounded hover:bg-gray-600">
              Name{sortArrow('name')}
            </button>
            <button onClick={() => toggleSort('size')} className="px-2 py-1 text-xs bg-gray-700 rounded hover:bg-gray-600">
              Size{sortArrow('size')}
            </button>
          </div>
          <div className="max-h-64 overflow-y-auto space-y-1">
            {filtered.map((f) => {
              const dk = `${model.id}/${f.name}`;
              const dlId = keyToDlId[dk];
              const prog = dlId ? downloads[dlId] : undefined;
              const isBusy = busyDl[dk];
              return (
                <div key={f.name} className="flex items-center gap-2 px-2 py-1 rounded hover:bg-gray-700 text-xs">
                  <span className="flex-1 truncate">{f.name}</span>
                  <span className="text-gray-400 shrink-0 w-20 text-right">{fmtSize(f.size)}</span>
                  {prog ? (
                    <div className="shrink-0 w-32">
                      <div className="bg-gray-700 rounded-full h-2 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${prog.status === 'error' ? 'bg-red-500' : 'bg-blue-500'}`}
                          style={{ width: `${prog.percent}%` }}
                        />
                      </div>
                      <div className="text-[10px] text-gray-500 text-right mt-0.5">{prog.stage}</div>
                    </div>
                  ) : (
                    <button
                      onClick={() => onDownload(model.id, f.name)}
                      disabled={isBusy}
                      className="shrink-0 px-2 py-0.5 bg-blue-700 rounded hover:bg-blue-600 disabled:opacity-50"
                    >
                      Download
                    </button>
                  )}
                </div>
              );
            })}
            {filtered.length === 0 && (
              <p className="text-gray-500 text-xs px-2">No files match filter</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function OpenvinoDownloadButton({ modelId }: { modelId: string }) {
  const [busy, setBusy] = useState(false);
  const [prog, setProg] = useState<DownloadProgress | null>(null);

  const poll = (id: string) => {
    fetch(`/api/downloads/progress/${id}`)
      .then((r) => r.json())
      .then((p: DownloadProgress) => {
        setProg(p);
        if (p.status !== 'done' && p.status !== 'error') {
          setTimeout(() => poll(id), 500);
        } else {
          setBusy(false);
        }
      })
      .catch(() => setBusy(false));
  };

  const handleDownload = async () => {
    setBusy(true);
    try {
      const res = await api.downloadOpenvinoModel(modelId);
      poll(res.download_id);
    } catch (e: any) {
      alert(e.message);
      setBusy(false);
    }
  };

  if (prog) {
    return (
      <div className="w-32">
        <div className="bg-gray-700 rounded-full h-2 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${prog.status === 'error' ? 'bg-red-500' : 'bg-blue-500'}`}
            style={{ width: `${prog.percent}%` }}
          />
        </div>
        <div className="text-[10px] text-gray-500 text-right mt-0.5">{prog.stage}</div>
      </div>
    );
  }

  return (
    <button onClick={handleDownload} disabled={busy}
      className="px-3 py-1 text-xs bg-cyan-700 rounded hover:bg-cyan-600 disabled:opacity-50 shrink-0">
      {busy ? 'Downloading...' : 'Download (snapshot)'}
    </button>
  );
}
