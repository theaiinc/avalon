import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { LocalDriver, LocalModel, ActiveDriver, GPU } from '../types';
import Spinner from '../components/Spinner';
import SystemStatsChart from '../components/SystemStatsChart';
import CompatibilityWarning from '../components/CompatibilityWarning';

export default function BenchmarkPage() {
  const navigate = useNavigate();
  const [drivers, setDrivers] = useState<LocalDriver[]>([]);
  const [activeDrivers, setActiveDrivers] = useState<ActiveDriver[]>([]);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [gpus, setGpus] = useState<GPU[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const [currentBackend, setCurrentBackend] = useState<string | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);

  const [selectedModel, setSelectedModel] = useState('');
  const [selectedDevices, setSelectedDevices] = useState<string[]>([]);
  const [nCtx, setNCtx] = useState(2048);
  const [nPrompt, setNPrompt] = useState(512);
  const [nGen, setNGen] = useState(128);
  const [nBatch, setNBatch] = useState(512);
  const [repetitions, setRepetitions] = useState(3);
  const [mtpMode, setMtpMode] = useState('auto');

  const load = () => {
    setLoading(true);
    Promise.all([api.listDrivers(), api.listLocalModels(), api.listGPUs(), api.listActiveBenchmarks()])
      .then(([d, m, g, active]) => {
        setDrivers(d.local);
        setActiveDrivers(d.active);
        setModels(m.models);
        setGpus(g.gpus);
        if (active.active.length > 0) {
          const tid = active.active[0].id;
          setRunning(true);
          setTaskId(tid);
          poll(tid);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const driverForBackend = (backend: string): LocalDriver | undefined =>
    drivers.find((d) => d.backend === backend && (d.llama_bench_path || backend === 'npu'));

  const toggleDevice = async (backend: string) => {
    const driver = driverForBackend(backend);
    if (!driver) return;

    setSelectedDevices((prev) => {
      if (prev.includes(backend)) {
        return prev.filter((x) => x !== backend);
      }
      return [...prev, backend];
    });

    if (!activeDrivers.find((a) => a.backend === backend)) {
      await api.setDriverActive(driver.id, true).catch(() => {});
      load();
    }
  };

  const selectedBackends = selectedDevices;

  const poll = (tid: string) => {
    let attempts = 0;
    const startTime = Date.now();
    const elapsedTimer = setInterval(() => setElapsedSec(Math.floor((Date.now() - startTime) / 1000)), 1000);
    pollRef.current = window.setInterval(async () => {
      attempts++;
      try {
        const res = await api.benchmarkStatus(tid);
        if (res.status === 'done') {
          clearInterval(pollRef.current!);
          clearInterval(elapsedTimer);
          pollRef.current = null;
          setRunning(false);
          setTaskId(null);
          setCurrentBackend(null);
          setElapsedSec(0);
          navigate('/results');
        } else if (res.status === 'not_found') {
          if (attempts > 15) {
            clearInterval(pollRef.current!);
            clearInterval(elapsedTimer);
            pollRef.current = null;
            setRunning(false);
            setTaskId(null);
            setCurrentBackend(null);
            setElapsedSec(0);
            alert('Benchmark failed to start');
          }
        }
      } catch { }
    }, 1000);
  };

  const handleRun = async () => {
    if (!selectedModel) { alert('Select a model'); return; }
    if (selectedBackends.length === 0) { alert('Select at least one device'); return; }

    setRunning(true);
    setCurrentBackend(selectedBackends[0]);
    setElapsedSec(0);
    try {
      const res = await api.runBenchmark(selectedModel, selectedBackends, {
        n_ctx: nCtx,
        n_prompt: nPrompt,
        n_gen: nGen,
        n_batch: nBatch,
        repetitions,
        mtp_mode: mtpMode,
      });
      setTaskId(res.task_id);
      poll(res.task_id);
    } catch (e: any) {
      alert(e.message);
      setRunning(false);
    }
  };

  const handleStop = async () => {
    if (!taskId) return;
    try {
      await api.cancelBenchmark(taskId);
    } catch { }
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    setRunning(false);
    setTaskId(null);
    setCurrentBackend(null);
    setElapsedSec(0);
  };

  if (loading) {
    return <div className="flex items-center gap-3 text-gray-400 py-8"><Spinner className="w-5 h-5" /> Loading...</div>;
  }

  const modelFiles = models.flatMap((m) => m.files.map((f) => ({
    id: m.id,
    file: f,
    fullPath: `${m.path}\\${f}`,
    isHead: f.startsWith('MTP\\') || f.startsWith('MTP/') || f.includes('\\MTP\\') || f.includes('/MTP/'),
  })));

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Run Benchmark</h2>

      {running && (
        <div className="bg-blue-900/30 border border-blue-700 rounded-lg p-3 mb-6 text-sm flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Spinner className="w-4 h-4 text-blue-400" />
            <span>Benchmark running{currentBackend ? ` (${currentBackend})` : ''} ... {elapsedSec}s</span>
          </div>
          <button onClick={handleStop} className="px-3 py-1 bg-red-700 rounded hover:bg-red-600 text-xs">Stop</button>
        </div>
      )}

      {running && (
        <div className="mb-6">
          <SystemStatsChart running={running} />
        </div>
      )}

      {drivers.length === 0 && (
        <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-3 mb-6 text-sm">
          No drivers downloaded. Go to <button onClick={() => navigate('/drivers')} className="text-blue-400 hover:underline">Drivers page</button> to download one first.
        </div>
      )}

      {activeDrivers.length > 0 && (
        <div className="bg-green-900/30 border border-green-700 rounded-lg p-3 mb-6 text-sm">
          Active: {activeDrivers.map((a) => a.backend).join(', ')}
        </div>
      )}

      <div className="bg-gray-900 rounded-lg p-5 border border-gray-800">
        <div className="mb-5">
          <label className="block text-sm font-medium mb-2">Model</label>
          {modelFiles.length === 0 ? (
            <p className="text-gray-500 text-sm">No models downloaded. Go to <button onClick={() => navigate('/models')} className="text-blue-400 hover:underline">Models page</button>.</p>
          ) : (
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm outline-none focus:border-blue-500"
              >
                <option value="">Select a model file...</option>
                {models.map((m) => {
                  if (m.format === 'openvino') {
                    return (
                      <optgroup key={m.id} label={m.id}>
                        <option value={m.path}>🧠 {m.id} (OpenVINO IR)</option>
                      </optgroup>
                    );
                  }
                  const baseFiles = m.files.filter((f) => !(f.startsWith('MTP\\') || f.startsWith('MTP/') || f.includes('\\MTP\\') || f.includes('/MTP/')));
                  const headFiles = m.files.filter((f) => f.startsWith('MTP\\') || f.startsWith('MTP/') || f.includes('\\MTP\\') || f.includes('/MTP/'));
                  return (
                    <optgroup key={m.id} label={m.id}>
                      {baseFiles.map((f, i) => (
                        <option key={`base-${i}`} value={`${m.path}\\${f}`}>📦 {f} (base)</option>
                      ))}
                      {headFiles.map((f, i) => (
                        <option key={`head-${i}`} value={`${m.path}\\${f}`}>🎯 {f.replace('MTP\\', '').replace('MTP/', '')} (head)</option>
                      ))}
                    </optgroup>
                  );
                })}
              </select>
          )}
        </div>

        <div className="mb-5">
          <label className="block text-sm font-medium mb-2">Devices (select GPU(s) to benchmark on)</label>
          <div className="grid gap-3 md:grid-cols-2">
            {(() => {
              const groups = new Map<string, GPU[]>();
              for (const g of gpus) {
                const list = groups.get(g.name) ?? [];
                list.push(g);
                groups.set(g.name, list);
              }
              return Array.from(groups.entries()).map(([name, entries]) => {
                const hasDriver = (b: string) => !!driverForBackend(b);
                return (
                  <div key={name} className="bg-gray-900 rounded-lg p-4 border border-gray-800">
                    <div className="font-medium text-sm mb-2 truncate">{name}</div>
                    <div className="flex flex-wrap gap-2">
                      {entries.map((g) => {
                        const driver = driverForBackend(g.backend);
                        const avail = !!driver;
                        const sel = selectedDevices.includes(g.backend);
                        const active = !!activeDrivers.find((a) => a.backend === g.backend);
                        return (
                          <label
                            key={g.backend}
                            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs cursor-pointer border transition-colors ${
                              !avail
                                ? 'border-gray-700 opacity-40 cursor-not-allowed bg-gray-850'
                                : sel
                                ? 'border-blue-600 bg-blue-900/30'
                                : 'border-gray-700 hover:border-gray-500 bg-gray-800'
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={sel}
                              disabled={!avail || running}
                              onChange={() => toggleDevice(g.backend)}
                              className="accent-blue-600"
                            />
                            <span className="font-medium">{g.backend}</span>
                            <span className="text-gray-500">{g.memory_mb}MB{g.shared_memory_mb ? ` + ${(g.shared_memory_mb / 1024).toFixed(1)}GB` : ''}</span>
                            {!avail && <span className="text-gray-600">(no driver)</span>}
                            {avail && active && <span className="text-green-500">active</span>}
                          </label>
                        );
                      })}
                    </div>
                  </div>
                );
              });
            })()}
            {gpus.length === 0 && <span className="text-gray-500 text-sm">No GPUs detected</span>}
          </div>
        </div>

        <CompatibilityWarning
          backends={selectedBackends}
          modelName={selectedModel.split('\\').pop()?.split('/').pop() || selectedModel}
          gpuName={gpus.find((g) => g.backend === selectedBackends[0])?.name}
          driverVersion={gpus.find((g) => g.backend === selectedBackends[0])?.driver_version}
        />

        <div className="mb-5">
          <label className="block text-sm font-medium mb-2">Preset</label>
          <div className="flex flex-wrap gap-2">
            {[
              { label: 'Quick test', n_ctx: 2048, n_prompt: 512, n_gen: 128, n_batch: 512, reps: 3 },
              { label: 'Chat', n_ctx: 8192, n_prompt: 2048, n_gen: 512, n_batch: 512, reps: 3 },
              { label: 'Coding (start)', n_ctx: 100000, n_prompt: 10000, n_gen: 512, n_batch: 512, reps: 3 },
              { label: 'Coding (near full)', n_ctx: 100000, n_prompt: 90000, n_gen: 512, n_batch: 512, reps: 3 },
              { label: 'Document analysis', n_ctx: 32000, n_prompt: 28000, n_gen: 1024, n_batch: 512, reps: 3 },
              { label: 'Long ctx (max)', n_ctx: 128000, n_prompt: 120000, n_gen: 1024, n_batch: 512, reps: 3 },
            ].map((p, i) => (
              <button
                key={i}
                onClick={() => { setNCtx(p.n_ctx); setNPrompt(p.n_prompt); setNGen(p.n_gen); setNBatch(p.n_batch); setRepetitions(p.reps); }}
                className="px-3 py-1.5 text-xs rounded border border-gray-700 bg-gray-800 hover:bg-gray-700 hover:border-gray-500 transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">MTP mode (Multi-Token Prediction)</label>
          <div className="flex flex-wrap gap-3">
            {[
              { value: 'auto', label: 'Auto', desc: 'Use MTP if model has MTP head' },
              { value: 'force', label: 'MTP only', desc: 'Force MTP, skip llama-bench' },
              { value: 'off', label: 'No MTP', desc: 'Use llama-bench only' },
              { value: 'both', label: 'Both', desc: 'Run both MTP and non-MTP' },
            ].map((opt) => (
              <label key={opt.value} className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="radio"
                  name="mtp_mode"
                  value={opt.value}
                  checked={mtpMode === opt.value}
                  onChange={() => setMtpMode(opt.value)}
                  disabled={running}
                  className="accent-blue-600"
                />
                <div>
                  <span className="text-gray-200">{opt.label}</span>
                  <span className="text-gray-500 ml-2 text-xs">{opt.desc}</span>
                </div>
              </label>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-5">
          <div>
            <label className="block text-xs font-medium mb-1 text-gray-400">n_ctx (context)</label>
            <input type="number" value={nCtx} onChange={(e) => setNCtx(Number(e.target.value))} disabled={running} className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm outline-none focus:border-blue-500 disabled:opacity-50" />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1 text-gray-400">n_prompt</label>
            <input type="number" value={nPrompt} onChange={(e) => setNPrompt(Number(e.target.value))} disabled={running} className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm outline-none focus:border-blue-500 disabled:opacity-50" />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1 text-gray-400">n_gen</label>
            <input type="number" value={nGen} onChange={(e) => setNGen(Number(e.target.value))} disabled={running} className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm outline-none focus:border-blue-500 disabled:opacity-50" />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1 text-gray-400">n_batch</label>
            <input type="number" value={nBatch} onChange={(e) => setNBatch(Number(e.target.value))} disabled={running} className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm outline-none focus:border-blue-500 disabled:opacity-50" />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1 text-gray-400">repetitions</label>
            <input type="number" value={repetitions} onChange={(e) => setRepetitions(Number(e.target.value))} disabled={running} className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm outline-none focus:border-blue-500 disabled:opacity-50" />
          </div>
        </div>

        <button
          onClick={handleRun}
          disabled={running || !selectedModel || selectedBackends.length === 0}
          className="px-6 py-2 bg-blue-600 rounded text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {running ? 'Starting...' : 'Run Benchmark'}
        </button>
      </div>
    </div>
  );
}
