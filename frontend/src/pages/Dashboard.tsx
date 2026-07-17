import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { GPU, LocalDriver, LocalModel, BenchmarkListItem } from '../types';
import Spinner from '../components/Spinner';

export default function Dashboard() {
  const navigate = useNavigate();
  const [gpus, setGpus] = useState<GPU[]>([]);
  const [drivers, setDrivers] = useState<LocalDriver[]>([]);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [results, setResults] = useState<BenchmarkListItem[]>([]);
  const [driverUpdates, setDriverUpdates] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.listGPUs(),
      api.listDrivers(),
      api.listLocalModels(),
      api.listResults(),
      api.listDriverUpdates(),
    ]).then(([g, d, m, r, u]) => {
      setGpus(g.gpus);
      setDrivers(d.local);
      setModels(m.models);
      setResults(r.results);
      setDriverUpdates(u.updates);
    }).catch(console.error)
    .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="flex items-center gap-3 text-gray-400 py-8"><Spinner className="w-5 h-5" /> Loading...</div>;
  }

  const gpuGroups = gpus.reduce<Record<string, GPU[]>>((acc, g) => {
    (acc[g.name] ??= []).push(g);
    return acc;
  }, {});
  const gpuGroupList = Object.entries(gpuGroups);

  const cards = [
    { label: 'GPUs Detected', value: gpuGroupList.length, color: 'border-blue-500', to: '/gpus' },
    { label: 'Drivers Downloaded', value: drivers.length, color: 'border-green-500', to: '/drivers' },
    { label: 'Models Downloaded', value: models.length, color: 'border-yellow-500', to: '/models' },
    { label: 'Benchmark Runs', value: results.length, color: 'border-purple-500', to: '/results' },
  ];

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Dashboard</h2>

      {driverUpdates.map((u, i) => (
        <div key={i} className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-3 mb-4 text-sm flex items-center justify-between">
          <div>
            <span className="text-yellow-300 font-medium">Driver update available</span>
            <span className="text-gray-400 ml-2">{u.device_name}: {u.installed} → {u.latest}</span>
          </div>
          <a href={u.download_url} target="_blank" rel="noopener noreferrer"
            className="px-3 py-1 bg-yellow-700 rounded hover:bg-yellow-600 text-xs whitespace-nowrap ml-3">
            Download
          </a>
        </div>
      ))}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {cards.map((card) => (
          <button
            key={card.label}
            onClick={() => navigate(card.to)}
            className={`bg-gray-900 border-l-4 ${card.color} rounded-lg p-4 text-left hover:bg-gray-800 transition-colors`}
          >
            <div className="text-3xl font-bold">{card.value}</div>
            <div className="text-sm text-gray-400 mt-1">{card.label}</div>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="font-semibold mb-3">GPUs</h3>
          {gpus.length === 0 ? (
            <p className="text-gray-500 text-sm">No GPUs detected</p>
          ) : (
            <div className="space-y-2">
              {gpuGroupList.map(([name, entries], i) => (
                <div key={i} className="flex justify-between text-sm">
                  <span className="truncate mr-2">{name}</span>
                  <span className="text-gray-400 shrink-0">{entries[0].vendor} · {entries.map(e => e.backend).join(', ')}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="font-semibold mb-3">Recent Results</h3>
          {results.length === 0 ? (
            <p className="text-gray-500 text-sm">No benchmark results yet</p>
          ) : (
            <div className="space-y-2">
              {results.slice(0, 5).map((r) => (
                <button
                  key={r.id}
                  onClick={() => navigate(`/results`)}
                  className="w-full flex justify-between text-sm hover:text-blue-400"
                >
                  <span>{r.model_name}</span>
                  <span className="text-gray-400">{r.backends.join(', ')}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
