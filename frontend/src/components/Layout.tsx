import { useEffect, useState, useRef } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import Spinner from './Spinner';
import avalonLogo from '../assets/avalon-logo-rounded.png';

const links = [
  { to: '/', label: 'Dashboard', icon: '◉' },
  { to: '/hardware', label: 'Hardware', icon: '⚙' },
  { to: '/models', label: 'Models', icon: '▦' },
  { to: '/benchmark', label: 'Benchmark', icon: '⚡' },
  { to: '/results', label: 'Results', icon: '☰' },
  { to: '/api-server', label: 'API Server', icon: '⇌' },
  { to: '/pc-links', label: 'PC Links', icon: '⇄' },
];

export default function Layout() {
  const [running, setRunning] = useState<{ id: string; elapsed: number } | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('/api/benchmark/list-active');
        const data = await res.json();
        if (data.active?.length > 0) {
          setRunning({ id: data.active[0].id, elapsed: 0 });
          if (!pollRef.current) {
            const start = Date.now();
            pollRef.current = window.setInterval(() => {
              setRunning((prev) => prev ? { ...prev, elapsed: Math.floor((Date.now() - start) / 1000) } : null);
            }, 1000);
          }
        } else {
          setRunning(null);
          if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
        }
      } catch { }
    };
    check();
    const timer = setInterval(check, 5000);
    return () => { clearInterval(timer); if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const handleStop = async () => {
    if (!running) return;
    try {
      await fetch(`/api/benchmark/cancel/${running.id}`, { method: 'POST' });
    } catch { }
  };

  return (
    <div className="flex h-screen">
      <nav className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">
        <div className="p-4 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <img src={avalonLogo} alt="Avalon" className="h-10 w-10 rounded-lg object-cover border border-amber-200/20" />
            <div>
              <h1 className="text-lg font-bold tracking-tight">Avalon</h1>
              <p className="text-xs text-gray-500 mt-0.5">Model Serving Dashboard</p>
            </div>
          </div>
        </div>
        <div className="flex-1 p-2 space-y-1">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded text-sm transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800'
                }`
              }
            >
              <span className="w-5 text-center">{link.icon}</span>
              {link.label}
            </NavLink>
          ))}
        </div>
      </nav>
      <main className="flex-1 overflow-auto p-6 relative">
        {running && (
          <div className="bg-blue-900/30 border border-blue-700 rounded-lg p-3 mb-4 text-sm flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Spinner className="w-4 h-4 text-blue-400" />
              <span>Benchmark running ... {running.elapsed}s</span>
            </div>
            <button onClick={handleStop} className="px-3 py-1 bg-red-700 rounded hover:bg-red-600 text-xs">Stop</button>
          </div>
        )}
        <Outlet />
      </main>
    </div>
  );
}
