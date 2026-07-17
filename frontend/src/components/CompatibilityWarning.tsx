import { useEffect, useState } from 'react';
import { api } from '../api/client';

interface CompatibilityMatch {
  id: string;
  backends: string[];
  severity: string;
  title: string;
  detail: string;
  workaround?: string;
  ref?: string;
}

interface Props {
  backends: string[];
  modelName?: string;
  gpuName?: string;
  driverVersion?: string;
}

export default function CompatibilityWarning({ backends, modelName, gpuName, driverVersion }: Props) {
  const [matches, setMatches] = useState<CompatibilityMatch[]>([]);
  const [allIssues, setAllIssues] = useState<CompatibilityMatch[]>([]);

  useEffect(() => {
    api.listCompatibilityIssues().then((r) => setAllIssues(r.issues)).catch(() => {});
  }, []);

  useEffect(() => {
    if (backends.length === 0) { setMatches([]); return; }
    const filtered = allIssues.filter((issue) => {
      if (!backends.some((b) => issue.backends.includes(b))) return false;
      const m = issue as any;
      const namePatterns = m.match?.model_name_contains ?? [];
      if (namePatterns.length > 0 && modelName) {
        if (!namePatterns.some((p: string) => modelName.toLowerCase().includes(p.toLowerCase()))) return false;
      }
      const gpuPatterns = m.match?.gpu_name_contains ?? [];
      if (gpuPatterns.length > 0 && gpuName) {
        if (!gpuPatterns.some((p: string) => gpuName.toLowerCase().includes(p.toLowerCase()))) return false;
      }
      if (m.match?.driver_version_below && driverVersion) {
        if (driverVersion >= m.match.driver_version_below) return false;
      }
      return true;
    });
    setMatches(filtered);
  }, [backends, modelName, gpuName, driverVersion, allIssues]);

  if (matches.length === 0) return null;

  return (
    <div className="space-y-2 mb-4">
      {matches.map((m) => (
        <div
          key={m.id}
          className={`rounded p-3 text-sm border ${
            m.severity === 'error'
              ? 'bg-red-900/30 border-red-700 text-red-200'
              : m.severity === 'warning'
              ? 'bg-yellow-900/30 border-yellow-700 text-yellow-200'
              : 'bg-blue-900/30 border-blue-700 text-blue-200'
          }`}
        >
          <div className="flex items-start gap-2">
            <span className="mt-0.5 shrink-0">
              {m.severity === 'error' ? '⛔' : m.severity === 'warning' ? '⚠️' : 'ℹ️'}
            </span>
            <div className="min-w-0">
              <div className="font-semibold text-sm">{m.title}</div>
              <p className="text-xs mt-1 opacity-90">{m.detail}</p>
              {m.workaround && (
                <p className="text-xs mt-1 font-medium">Workaround: {m.workaround}</p>
              )}
              {m.ref && (
                <a href={m.ref} target="_blank" rel="noopener noreferrer" className="text-xs underline mt-1 inline-block opacity-80 hover:opacity-100">
                  Reference ↗
                </a>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
