import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { BenchmarkListItem, BenchmarkResult } from '../types';
import Spinner from '../components/Spinner';

export default function ResultsPage() {
  const [results, setResults] = useState<BenchmarkListItem[]>([]);
  const [selected, setSelected] = useState<BenchmarkResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [sortAsc, setSortAsc] = useState(false);

  const sorted = [...results].sort((a, b) => {
    const diff = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
    return sortAsc ? diff : -diff;
  });

  const load = () => {
    setLoading(true);
    api.listResults()
      .then((d) => setResults(d.results))
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const handleSelect = async (id: string) => {
    try {
      const r = await api.getResult(id);
      setSelected(r);
    } catch (e: any) {
      alert(e.message);
    }
  };

  const handleRemove = async (id: string) => {
    if (!confirm('Delete this result?')) return;
    try {
      await api.removeResult(id);
      if (selected?.id === id) setSelected(null);
      load();
    } catch (e: any) {
      alert(e.message);
    }
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Benchmark Results</h2>

      {loading ? (
        <div className="flex items-center gap-3 text-gray-400 py-8"><Spinner className="w-5 h-5" /> Loading...</div>
      ) : results.length === 0 ? (
        <p className="text-gray-500">No benchmark results yet. Run one from the <a href="/benchmark" className="text-blue-400 hover:underline">Benchmark page</a>.</p>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch">
          <div className="lg:col-span-1 space-y-2 h-full">
            <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
              <span>{results.length} result{results.length !== 1 ? 's' : ''}</span>
              <button onClick={() => setSortAsc(!sortAsc)} className="hover:text-gray-300 transition-colors">
                {sortAsc ? '↑ oldest first' : '↓ newest first'}
              </button>
            </div>
            {sorted.map((r) => (
              <button
                key={r.id}
                onClick={() => handleSelect(r.id)}
                className={`w-full text-left bg-gray-900 rounded p-3 border text-sm hover:bg-gray-800 ${
                  selected?.id === r.id ? 'border-blue-500' : 'border-gray-800'
                }`}
              >
                <div className="font-medium truncate">{r.model_name}</div>
                <div className="text-xs text-gray-500 mt-1">{r.backends.join(', ')} · {new Date(r.timestamp).toLocaleString()}</div>
              </button>
            ))}
          </div>

          <div className="lg:col-span-2 h-full">
            {selected ? (
              <BenchmarkDetail result={selected} onRemove={handleRemove} />
            ) : (
              <div className="bg-gray-900 rounded-lg p-6 border border-gray-800 text-center text-gray-500 h-full">
                Select a result to view details
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function BenchmarkDetail({ result, onRemove }: { result: BenchmarkResult; onRemove: (id: string) => void }) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [chartMetric, setChartMetric] = useState<'prompt' | 'gen'>('gen');

  const promptTs = (run: any) => {
    if (!run.results || run.results.length === 0) return null;
    if (run.mtp) {
      const tp = run.results[0]?.tokens_per_sec;
      return tp?.prompt_tps ?? tp?.prompt ?? null;
    }
    if (run.npu) {
      const tp = run.results[0]?.tokens_per_sec;
      return tp?.prompt_tps ?? null;
    }
    const r = run.results.find((x: any) => x.n_prompt > 0 && (x.n_gen === 0 || x.n_gen === undefined));
    return r?.avg_ts ?? null;
  };

  const genTs = (run: any) => {
    if (!run.results || run.results.length === 0) return null;
    if (run.mtp) {
      const tp = run.results[0]?.tokens_per_sec;
      return tp?.generation_tps ?? null;
    }
    if (run.npu) {
      const tp = run.results[0]?.tokens_per_sec;
      return tp?.generation_tps ?? null;
    }
    const r = run.results.find((x: any) => x.n_gen > 0);
    return r?.avg_ts ?? null;
  };

  const runMetric = (run: any) => chartMetric === 'prompt' ? promptTs(run) : genTs(run);

  const formatVal = (val: number | null) => val !== null ? val.toFixed(2) : '—';

  const firstResult = (run: any) => run.results?.[0] ?? null;

  const fmtBytes = (bytes: number) => {
    const gb = bytes / (1024 ** 3);
    return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(bytes / (1024 ** 2)).toFixed(1)} MB`;
  };

  const fmtParams = (n: number) => {
    if (n >= 1e12) return `${(n / 1e12).toFixed(2)} T`;
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)} B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)} M`;
    return `${n}`;
  };

  const infoLabels: Record<string, string> = {
    n_threads: 'Threads',
    n_gpu_layers: 'GPU Layers',
    n_batch: 'Batch',
    n_ubatch: 'UBatch',
    cpu_info: 'CPU',
    gpu_info: 'GPU',
    flash_attn: 'Flash Attn',
    use_mmap: 'mmap',
    type_k: 'K type',
    type_v: 'V type',
    build_commit: 'Build',
    backends: 'Backend',
  };

  return (
    <div className="h-full">
      <div className="flex items-start justify-between mb-3">
        <div className="min-w-0 flex-1">
          <h3 className="font-semibold truncate">{result.model_name}</h3>
          <p className="text-xs text-gray-500 mt-0.5">{new Date(result.timestamp).toLocaleString()}</p>
          {(firstResult(result.runs[0])?.model_type || firstResult(result.runs[0])?.model_n_params) && (
            <p className="text-xs text-gray-400 mt-1">
              {firstResult(result.runs[0])?.model_type ?? ''}
              {firstResult(result.runs[0])?.model_n_params ? ` · ${fmtParams(firstResult(result.runs[0]).model_n_params)} params` : ''}
              {firstResult(result.runs[0])?.model_size ? ` · ${fmtBytes(firstResult(result.runs[0]).model_size)}` : ''}
            </p>
          )}
          <p className="text-xs text-gray-600 mt-0.5">
            ctx={result.bench_params?.n_ctx ?? '?'} · p={result.bench_params?.n_prompt ?? '?'} · n={result.bench_params?.n_gen ?? '?'} · b={result.bench_params?.n_batch ?? '?'} · r={result.bench_params?.repetitions ?? '?'}
          </p>
        </div>
        <button onClick={() => onRemove(result.id)} className="shrink-0 px-3 py-1 text-xs bg-red-700 rounded hover:bg-red-600">Delete</button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left py-2 px-3 text-gray-400 font-medium">Backend</th>
              <th className="text-right py-2 px-3 text-gray-400 font-medium">Prompt t/s</th>
              <th className="text-right py-2 px-3 text-gray-400 font-medium">Gen t/s</th>
              <th className="text-right py-2 px-3 text-gray-400 font-medium">Time (s)</th>
              <th className="text-center py-2 px-3 text-gray-400 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {result.runs.map((run, i) => (
              <>
                <tr
                  key={i}
                  className={`border-b border-gray-800 cursor-pointer hover:bg-gray-800/50 ${run.status === 'success' ? '' : 'text-red-400'}`}
                  onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                >
                  <td className="py-2 px-3 font-medium">
                    <div className="flex items-center gap-1.5">
                      <span className={`text-gray-500 text-xs transition-transform ${expandedIdx === i ? 'rotate-90' : ''}`}>▶</span>
                      {run.backend}
                      {run.mtp && <span className="text-[10px] px-1 py-0.5 rounded bg-blue-900 text-blue-300">MTP</span>}
                      {run.npu && <span className="text-[10px] px-1 py-0.5 rounded bg-purple-900 text-purple-300">NPU</span>}
                    </div>
                  </td>
                  <td className="py-2 px-3 text-right font-mono">{formatVal(promptTs(run))}</td>
                  <td className="py-2 px-3 text-right font-mono">{formatVal(genTs(run))}</td>
                  <td className="py-2 px-3 text-right text-gray-400">{run.elapsed_sec?.toFixed(1) ?? '—'}</td>
                  <td className="py-2 px-3 text-center">
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      run.status === 'success' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                    }`}>
                      {run.status}
                    </span>
                  </td>
                </tr>
                {expandedIdx === i && run.status === 'success' && (
                  <tr key={`${i}-detail`} className="border-b border-gray-800">
                    <td colSpan={5} className="py-3 px-6">
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-x-6 gap-y-1.5 text-xs">
                        {Object.entries(infoLabels).map(([key, label]) => {
                          const val = firstResult(run)?.[key] ?? null;
                          if (val === null || val === undefined || val === '' || val === 0) return null;
                          const display = typeof val === 'boolean' ? (val ? 'yes' : 'no') : String(val);
                          return (
                            <div key={key} className="flex justify-between gap-2">
                              <span className="text-gray-500">{label}</span>
                              <span className="text-gray-300 font-mono truncate max-w-[160px]" title={display}>{display}</span>
                            </div>
                          );
                        })}
                        {run.mtp && run.draft_model && (
                          <div className="flex justify-between gap-2">
                            <span className="text-gray-500">Draft model</span>
                            <span className="text-gray-300 font-mono truncate max-w-[160px]" title={run.draft_model}>{run.draft_model.split('\\').pop()}</span>
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>

      {result.runs.every((r) => r.status === 'success') && (
        <div className="mt-6 bg-gray-800 rounded p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold text-sm">Comparison</h4>
            <div className="flex gap-1 text-xs">
              <button
                onClick={() => setChartMetric('gen')}
                className={`px-2 py-1 rounded ${chartMetric === 'gen' ? 'bg-blue-700 text-white' : 'bg-gray-700 text-gray-400 hover:text-white'}`}
              >Gen t/s</button>
              <button
                onClick={() => setChartMetric('prompt')}
                className={`px-2 py-1 rounded ${chartMetric === 'prompt' ? 'bg-blue-700 text-white' : 'bg-gray-700 text-gray-400 hover:text-white'}`}
              >Prompt t/s</button>
            </div>
          </div>
          <div className="flex items-end gap-3" style={{ height: 160 }}>
            {result.runs.map((run, i) => {
              const ts = runMetric(run);
              const allTs = result.runs.map((r) => runMetric(r) ?? 0);
              const maxTs = Math.max(...allTs);
              const barMax = 120;
              const barH = ts && maxTs ? Math.round((ts / maxTs) * barMax) : 4;
              return (
                <div key={i} className="flex flex-col items-center flex-1">
                  <span className="text-xs text-gray-400 mb-1">{formatVal(ts)}</span>
                  <div
                    className="w-full bg-blue-600 rounded-t transition-all duration-200"
                    style={{ height: barH, minHeight: 4 }}
                  />
                  <span className="text-xs mt-1 text-gray-500 truncate max-w-full">{run.backend}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
