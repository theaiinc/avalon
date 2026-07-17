import { useState, useEffect, useRef } from 'react';
import { api } from '../api/client';
import type { SystemStats } from '../types';
import Spinner from './Spinner';

const MAX_HISTORY = 60;

export default function SystemStatsChart({ running }: { running: boolean }) {
  const [history, setHistory] = useState<{ cpu: number; ram: number; gpuMap: Record<string, number>; t: number }[]>([]);
  const [current, setCurrent] = useState<SystemStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (!running) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }

    setError(null);

    const poll = async () => {
      try {
        const stats = await api.benchmarkStats();
        setCurrent(stats);
        setError(null);
        const gpuMap: Record<string, number> = {};
        for (const g of stats.gpus) {
          gpuMap[g.name] = g.utilization_percent;
        }
        setHistory((prev) => {
          const next = [...prev, { cpu: stats.cpu.percent, ram: stats.ram.percent, gpuMap, t: stats.timestamp }];
          if (next.length > MAX_HISTORY) next.splice(0, next.length - MAX_HISTORY);
          return next;
        });
      } catch (e: any) {
        const msg = e?.message || '';
        if (msg.includes('Not Found')) {
          setError('Stats endpoint not found — restart the backend server');
        } else {
          setError(msg || 'Failed to fetch stats');
        }
      }
    };

    poll();
    intervalRef.current = window.setInterval(poll, 2000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [running]);

  if (!running && !current) return null;

  const W = 280;
  const H = 80;
  const pad = { left: 4, right: 4, top: 4, bottom: 4 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;

  const toX = (i: number) => pad.left + (i / Math.max(MAX_HISTORY - 1, 1)) * chartW;

  const sparkline = (data: number[], color: string) => {
    if (data.length < 2) return null;
    const pts = data.map((v, i) => `${toX(i)},${pad.top + chartH - (v / 100) * chartH}`);
    return <polyline fill="none" stroke={color} strokeWidth="1.5" points={pts.join(' ')} />;
  };

  const cpu = history.map((h) => h.cpu);
  const ram = history.map((h) => h.ram);
  const gpuNames = current?.gpus.map((g) => g.name) || [];

  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <h3 className="font-semibold text-sm text-gray-300 mb-3 flex items-center gap-2">
        <Spinner className="w-4 h-4" /> Live System Stats
      </h3>

      {error && (
        <div className="text-xs text-red-400 mb-2">Stats unavailable: {error}</div>
      )}

      {!current && !error && (
        <div className="flex items-center gap-2 text-gray-500 text-xs">
          <Spinner className="w-3 h-3" /> Collecting system stats...
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {current && (
          <>
            <div className="space-y-2">
              <StatBar label="CPU" value={current.cpu.percent} color="bg-blue-500" />
              <StatBar label="RAM" value={current.ram.percent} color="bg-green-500" />
              {current.gpus.map((g, i) => (
                <div key={i}>
                  <StatBar
                    label={`GPU: ${g.name.split(' ').slice(0, 3).join(' ')}`}
                    value={g.utilization_percent}
                    color="bg-purple-500"
                  />
                  {g.memory_total_mb > 0 && (
                    <div className="text-xs text-gray-500 ml-1">
                      Mem: {(g.memory_used_mb / 1024).toFixed(1)} / {(g.memory_total_mb / 1024).toFixed(1)} GB
                      {g.temperature_c > 0 && ` · ${g.temperature_c}°C`}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div>
              <svg width={W} height={H} className="bg-gray-950 rounded">
                {sparkline(cpu, '#3b82f6')}
                {sparkline(ram, '#22c55e')}
                {gpuNames.map((_, i) => {
                  const gpuData = history.map((h) => h.gpuMap[gpuNames[i]] ?? 0);
                  return sparkline(gpuData, ['#a855f7', '#f59e0b', '#ec4899'][i % 3]);
                })}
              </svg>
              <div className="flex gap-3 mt-1 text-[10px] text-gray-500">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500" /> CPU</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-500" /> RAM</span>
                {gpuNames.length > 0 && <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-500" /> GPU</span>}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function StatBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-36 truncate shrink-0" title={label}>{label}</span>
      <div className="flex-1 bg-gray-800 rounded-full h-2">
        <div className={`${color} h-2 rounded-full transition-all duration-500`} style={{ width: `${Math.min(value, 100)}%` }} />
      </div>
      <span className="text-xs text-gray-400 w-10 text-right">{value.toFixed(0)}%</span>
    </div>
  );
}
