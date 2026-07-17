import { useEffect, useState, useMemo } from 'react';
import { api } from '../api/client';
import type { GPU } from '../types';
import Spinner from '../components/Spinner';
import CompatibilityWarning from '../components/CompatibilityWarning';

function groupGpus(gpus: GPU[]): { name: string; vendor: string; index: string; memory_mb: number; shared_memory_mb?: number; driver_version: string; backends: GPU[] }[] {
  const map = new Map<string, GPU[]>();
  for (const g of gpus) {
    const list = map.get(g.name) ?? [];
    list.push(g);
    map.set(g.name, list);
  }
  return Array.from(map.entries()).map(([name, entries]) => ({
    name,
    vendor: entries[0].vendor,
    index: entries[0].index,
    memory_mb: entries[0].memory_mb,
    shared_memory_mb: entries[0].shared_memory_mb,
    driver_version: entries[0].driver_version,
    backends: entries,
  }));
}

export default function GPUPage() {
  const [gpus, setGpus] = useState<GPU[]>([]);
  const [loading, setLoading] = useState(true);

  const groups = useMemo(() => groupGpus(gpus), [gpus]);

  useEffect(() => {
    api.listGPUs()
      .then((d) => setGpus(d.gpus))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const bgClass = (b: string) =>
    b === 'cuda' ? 'bg-green-900 text-green-300' :
    b === 'sycl' ? 'bg-blue-900 text-blue-300' :
    b === 'vulkan' ? 'bg-purple-900 text-purple-300' :
    b === 'openvino' ? 'bg-amber-900 text-amber-300' :
    b === 'npu' ? 'bg-cyan-900 text-cyan-300' :
    'bg-gray-700 text-gray-300';

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">GPUs</h2>
      {loading ? (
        <div className="flex items-center gap-3 text-gray-400 py-8"><Spinner className="w-5 h-5" /> Detecting GPUs...</div>
      ) : groups.length === 0 ? (
        <p className="text-gray-500">No GPUs detected. Make sure drivers are installed.</p>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {groups.map((g, i) => (
            <div key={i} className="bg-gray-900 rounded-lg p-5 border border-gray-800">
              <div className="flex items-start justify-between mb-3">
                <div className="min-w-0 flex-1">
                  <h3 className="font-semibold text-lg truncate">{g.name}</h3>
                  <span className="text-xs text-gray-500">{g.vendor} · Index {g.index}</span>
                </div>
                <div className="flex flex-wrap gap-1 ml-3 shrink-0">
                  {g.backends.map((b) => (
                    <span key={b.backend} className={`px-2 py-0.5 rounded text-xs font-medium ${bgClass(b.backend)}`}>
                      {b.backend}
                    </span>
                  ))}
                </div>
              </div>
              <div className="space-y-1 text-sm text-gray-400">
                <div className="flex justify-between">
                  <span>Memory</span>
                  <span>{g.memory_mb > 0 ? `${(g.memory_mb / 1024).toFixed(1)} GB` : 'Unknown'}
                    {g.shared_memory_mb ? <span className="text-gray-500 ml-1">(+{(g.shared_memory_mb / 1024).toFixed(1)} GB shared)</span> : ''}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Driver</span>
                  <span>{g.driver_version || 'N/A'}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-8">
        <h3 className="text-lg font-semibold mb-3">Known Compatibility Issues</h3>
        {groups.map((g, i) => (
          <CompatibilityWarning
            key={i}
            backends={g.backends.map((b) => b.backend)}
            gpuName={g.name}
            driverVersion={g.driver_version}
          />
        ))}
        {groups.length === 0 && <p className="text-gray-500">Load GPUs first to see relevant issues.</p>}
      </div>
    </div>
  );
}
