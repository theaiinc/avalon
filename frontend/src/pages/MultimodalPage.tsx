import { useEffect, useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { api } from '../api/client';
import type { LocalModel, MultimodalCase, MultimodalModality, MultimodalProfile, MultimodalRun } from '../types';

const modalities: { value: MultimodalModality; label: string }[] = [
  { value: 'tts', label: 'TTS' },
  { value: 'stt', label: 'STT' },
  { value: 'imagegen', label: 'Image generation' },
  { value: 'videogen', label: 'Video generation' },
];
const PROMPTS_KEY = 'avalon_multimodal_prompts';

function gatewayAdapterUrl(modality: MultimodalModality): string {
  const path = {
    tts: 'audio/speech',
    stt: 'audio/transcriptions',
    imagegen: 'images/generations',
    videogen: 'videos/generations',
  }[modality];
  return `http://127.0.0.1:8787/v1/${path}`;
}

function artifactUrl(url: string): string {
  if (/^https?:\/\//i.test(url)) return url;
  const base = window.location.protocol === 'file:' ? 'http://127.0.0.1:8771' : '';
  return `${base}${url.startsWith('/') ? url : `/${url}`}`;
}

function modelCapabilities(model: LocalModel): MultimodalModality[] {
  if (model.capabilities?.length) return model.capabilities.filter((value): value is MultimodalModality =>
    modalities.some((item) => item.value === value));
  const text = `${model.id} ${model.repo_id} ${model.files.join(' ')}`.toLowerCase();
  const markers: Record<MultimodalModality, string[]> = {
    tts: ['tts', 'text-to-speech', 'speech-synthesis', 'kokoro', 'bark', 'piper'],
    stt: ['whisper', 'speech-to-text', 'automatic-speech-recognition', 'wav2vec', 'asr'],
    imagegen: ['flux', 'stable-diffusion', 'stable_diffusion', 'sdxl', 'qwen-image', 'qwen_image', 'text-to-image', 'imagegen', 'diffusion'],
    videogen: ['video-generation', 'video_generation', 'cogvideo', 'hunyuan-video', 'mochi', 'animatediff', 'text-to-video'],
  };
  return modalities.filter(({ value }) => markers[value].some((marker) => text.includes(marker))).map(({ value }) => value);
}

export default function MultimodalPage() {
  const location = useLocation();
  const [profiles, setProfiles] = useState<MultimodalProfile[]>([]);
  const [cases, setCases] = useState<MultimodalCase[]>([]);
  const [runs, setRuns] = useState<MultimodalRun[]>([]);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [capabilities, setCapabilities] = useState<{ protocol: string; approved_executables: string[] } | null>(null);
  const [modality, setModality] = useState<MultimodalModality>('tts');
  const [mode, setMode] = useState<'local' | 'http' | 'builtin'>('http');
  const [name, setName] = useState('');
  const [model, setModel] = useState('');
  const [modelPath, setModelPath] = useState('');
  const [executableId, setExecutableId] = useState('');
  const [input, setInput] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [imageBase64, setImageBase64] = useState('');
  const [imageName, setImageName] = useState('');
  const [imageWidth, setImageWidth] = useState(1024);
  const [imageHeight, setImageHeight] = useState(1024);
  const [imageSteps, setImageSteps] = useState(28);
  const [imageGuidance, setImageGuidance] = useState(2.5);
  const [imageSeed, setImageSeed] = useState(-1);
  const [promptHistory, setPromptHistory] = useState<string[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [savedPage, setSavedPage] = useState(1);
  const [historyPage, setHistoryPage] = useState(1);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');

  const reload = async () => {
    const [p, c, r, caps, localModels] = await Promise.all([
      api.listMultimodalProfiles(), api.listMultimodalCases(), api.listMultimodalRuns(),
      api.multimodalCapabilities(), api.listLocalModels(),
    ]);
    setProfiles(p.profiles); setCases(c.cases); setRuns(r.runs);
    setCapabilities(caps);
    setModels(localModels.models);
  };
  const reloadRuns = async () => {
    const result = await api.listMultimodalRuns();
    setRuns(result.runs);
  };
  useEffect(() => { reload().catch((e) => setMessage(e.message)); }, []);
  useEffect(() => {
    const timer = window.setInterval(() => reloadRuns().catch(() => {}), 1500);
    return () => clearInterval(timer);
  }, []);

  const compatibleProfiles = useMemo(() => profiles.filter((p) => p.modality === modality), [profiles, modality]);
  const compatibleCases = useMemo(() => cases.filter((c) => c.modality === modality), [cases, modality]);
  const savedTests = useMemo(() => {
    const seen = new Set<string>();
    return [...compatibleCases].reverse()
      .map((testCase) => ({
        testCase,
        profile: compatibleProfiles.find((profile) => profile.id === testCase.profile_id) || compatibleProfiles[0],
      }))
      .filter((item) => {
        if (!item.profile) return false;
        const key = JSON.stringify([item.profile.id, item.testCase.modality, item.testCase.input, item.testCase.assertions]);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
  }, [compatibleCases, compatibleProfiles]);
  const pageSize = 5;
  const savedPageCount = Math.max(1, Math.ceil(savedTests.length / pageSize));
  const historyPageCount = Math.max(1, Math.ceil(runs.length / pageSize));
  const visibleSavedTests = savedTests.slice((savedPage - 1) * pageSize, savedPage * pageSize);
  const sortedRuns = useMemo(
    () => [...runs].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()),
    [runs],
  );
  const visibleRuns = sortedRuns.slice((historyPage - 1) * pageSize, historyPage * pageSize);
  useEffect(() => {
    setSavedPage(1);
    setHistoryPage(1);
  }, [modality]);
  useEffect(() => {
    setSavedPage((page) => Math.min(page, savedPageCount));
  }, [savedPageCount]);
  useEffect(() => {
    setHistoryPage((page) => Math.min(page, historyPageCount));
  }, [historyPageCount]);
  const compatibleModels = useMemo(
    () => models.filter((localModel) => modelCapabilities(localModel).includes(modality)),
    [models, modality],
  );
  const generatedUrl = gatewayAdapterUrl(modality);
  const activeRun = runs.find((run) => run.id === activeRunId)
    || runs.find((run) => ['queued', 'running', 'cancelling'].includes(run.state))
    || [...runs].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())[0]
    || null;
  useEffect(() => {
    if (!compatibleModels.some((localModel) => localModel.id === name)) {
      setName('');
      setModel('');
    }
  }, [compatibleModels, name]);
  useEffect(() => {
    const requestedId = new URLSearchParams(location.search).get('model');
    const requested = models.find((localModel) => localModel.id === requestedId);
    if (!requested) return;
    const supportedModality = modalities.find((item) => modelCapabilities(requested).includes(item.value));
    if (supportedModality) setModality(supportedModality.value);
    setName(requested.id);
    setModel(requested.repo_id || requested.id);
    setModelPath(requested.path);
  }, [models, location.search]);
  useEffect(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(PROMPTS_KEY) || '{}');
      setPromptHistory(Array.isArray(stored[modality]) ? stored[modality] : []);
    } catch {
      setPromptHistory([]);
    }
  }, [modality]);
  useEffect(() => {
    setMode(modality === 'imagegen' ? 'builtin' : 'http');
  }, [modality]);

  const saveAndRun = async () => {
    if (!name.trim() || !input.trim()) { setMessage('Enter a profile name and test input.'); return; }
    const prompt = input.trim();
    try {
      const stored = JSON.parse(localStorage.getItem(PROMPTS_KEY) || '{}');
      const previous = Array.isArray(stored[modality]) ? stored[modality] : [];
      stored[modality] = [prompt, ...previous.filter((item: string) => item !== prompt)].slice(0, 20);
      localStorage.setItem(PROMPTS_KEY, JSON.stringify(stored));
      setPromptHistory(stored[modality]);
    } catch { /* local storage is optional */ }
    setBusy(true); setMessage('');
    try {
      const profile = await api.saveMultimodalProfile({
        name, model, model_path: modelPath, modality, mode, url: mode === 'http' ? generatedUrl : undefined,
        allow_private_network: mode === 'http',
        executable_id: mode === 'local' ? executableId : undefined,
      });
      const testCase = await api.saveMultimodalCase({
        name: `${name} test`, modality,
        input: modality === 'stt' ? { audio_base64: input } : modality === 'imagegen' ? {
          prompt: input,
          text: input,
          ...(imageBase64 ? { image_base64: imageBase64 } : {}),
          options: {
            negative_prompt: negativePrompt,
            width: imageWidth,
            height: imageHeight,
            steps: imageSteps,
            guidance: imageGuidance,
            seed: imageSeed,
          },
        } : { prompt: input, text: input },
        assertions: {},
        profile_id: profile.profile.id,
      });
      const created = await api.createMultimodalRun(profile.profile.id, testCase.case.id);
      setActiveRunId(created.run.id);
      await reloadRuns();
      setMessage(
        mode === 'builtin'
          ? 'Run queued. Avalon will provision the local image runtime and required model files.'
          : mode === 'local'
          ? 'Run queued. Local plugins must be approved by the server allowlist.'
          : 'Run queued. The configured HTTP adapter will be contacted.',
      );
    } catch (e: any) {
      setMessage(e.message);
    } finally { setBusy(false); }
  };

  const runExisting = async (profile: MultimodalProfile, testCase: MultimodalCase) => {
    try {
      const created = await api.createMultimodalRun(profile.id, testCase.id);
      setActiveRunId(created.run.id);
      await reloadRuns();
    }
    catch (e: any) { setMessage(e.message); }
  };

  const removeCase = async (testCase: MultimodalCase) => {
    if (!confirm(`Delete saved test "${testCase.name}"?`)) return;
    try { await api.removeMultimodalCase(testCase.id); await reload(); }
    catch (e: any) { setMessage(e.message); }
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-2">Multimodal Testing</h2>
      <p className="text-sm text-gray-500 mb-6">Test TTS, STT, image-generation, and video-generation adapters without changing the LLM benchmark path.</p>
      {capabilities && <p className="text-xs text-gray-600 mb-4">Protocol {capabilities.protocol} · approved local adapters: {capabilities.approved_executables.length || 'none'}</p>}

      <div className="flex flex-wrap gap-2 mb-5">
        {modalities.map((item) => (
          <button key={item.value} onClick={() => setModality(item.value)}
            className={`px-3 py-1.5 rounded text-sm ${modality === item.value ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'}`}>
            {item.label}
          </button>
        ))}
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 mb-6">
        <h3 className="font-semibold mb-4">Create profile and test case</h3>
        <div className="flex flex-row gap-4 items-start" style={{ display: 'flex', flexDirection: 'row' }}>
        <div className="min-w-0" style={{ flex: '1 1 0%', minWidth: 0 }}>
          <label className="text-sm">Downloaded model
            <select value={name} onChange={(e) => {
              const selected = models.find((model) => model.id === e.target.value);
              setName(e.target.value);
              if (selected) {
                setModel(selected.repo_id || selected.id);
                setModelPath(selected.path);
              }
            }} className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-3 py-2">
              <option value="">Select a downloaded model...</option>
              {compatibleModels.map((localModel) => (
                <option key={localModel.id} value={localModel.id}>
                  {localModel.repo_id || localModel.id} ({localModel.format || 'local'})
                </option>
              ))}
            </select>
            {models.length === 0 && <span className="block text-xs text-yellow-500 mt-1">No downloaded models. Download one from the Models page first.</span>}
            {models.length > 0 && compatibleModels.length === 0 && (
              <span className="block text-xs text-yellow-500 mt-1">No downloaded {modality} capable models were detected.</span>
            )}
          </label>
        <div className="flex gap-5 my-4 text-sm">
          {modality === 'imagegen' && <label><input type="radio" checked={mode === 'builtin'} onChange={() => setMode('builtin')} className="mr-2 accent-blue-600" />Avalon auto runtime</label>}
          <label><input type="radio" checked={mode === 'http'} onChange={() => setMode('http')} className="mr-2 accent-blue-600" />HTTP adapter</label>
          <label><input type="radio" checked={mode === 'local'} onChange={() => setMode('local')} className="mr-2 accent-blue-600" />Approved local plugin</label>
        </div>
        {mode === 'builtin' ? (
          <div className="bg-blue-900/20 border border-blue-800 rounded p-3 mb-4 text-sm text-blue-200">
            Avalon will download stable-diffusion.cpp and the required FLUX files on first run, then generate locally.
          </div>
        ) : mode === 'http' ? (
          <label className="block text-sm mb-4">Adapter URL
            <input value={generatedUrl} readOnly
              className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-gray-400" />
            <span className="block text-xs text-gray-600 mt-1">Generated from the selected modality and local gateway. The selected model is sent in the request body.</span>
          </label>
        ) : (
          <label className="block text-sm mb-4">Approved executable ID
            <input value={executableId} onChange={(e) => setExecutableId(e.target.value)} placeholder="whisper-local"
              className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-3 py-2" />
            <span className="block text-xs text-gray-600 mt-1">Shell commands are not accepted; IDs resolve through the server-side allowlist.</span>
          </label>
        )}
        <label className="block text-sm mb-4">Test input
          <textarea value={input} onChange={(e) => setInput(e.target.value)} rows={3}
            placeholder={modality === 'stt' ? 'Base64 audio input' : modality === 'tts' ? 'Text to synthesize' : 'Prompt'}
            className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-3 py-2" />
        </label>
        {modality === 'imagegen' && (
          <div className="grid grid-cols-2 gap-3 mb-4 text-sm">
            <label className="col-span-2">Image to edit (optional)
              <input type="file" accept="image/png,image/jpeg,image/webp"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  if (file.size > 25 * 1024 * 1024) {
                    setMessage('Reference images must be 25 MB or smaller.');
                    e.currentTarget.value = '';
                    return;
                  }
                  const reader = new FileReader();
                  reader.onload = () => {
                    const dataUrl = String(reader.result || '');
                    setImageBase64(dataUrl.split(',')[1] || '');
                    setImageName(file.name);
                    setMessage('');
                  };
                  reader.readAsDataURL(file);
                }}
                className="mt-1 block w-full text-xs text-gray-400 file:mr-3 file:rounded file:border-0 file:bg-gray-700 file:px-3 file:py-2 file:text-gray-200 hover:file:bg-gray-600" />
              <span className="block text-xs text-gray-600 mt-1">
                {imageName ? `${imageName} selected` : 'PNG, JPEG, or WebP; Avalon converts it to base64 before sending.'}
              </span>
            </label>
            <label>Negative prompt
              <input value={negativePrompt} onChange={(e) => setNegativePrompt(e.target.value)}
                className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-3 py-2" />
            </label>
            <label>Seed
              <input type="number" value={imageSeed} onChange={(e) => setImageSeed(Number(e.target.value))}
                className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-3 py-2" />
            </label>
            <label>Width
              <input type="number" min={64} max={4096} value={imageWidth} onChange={(e) => setImageWidth(Number(e.target.value))}
                className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-3 py-2" />
            </label>
            <label>Height
              <input type="number" min={64} max={4096} value={imageHeight} onChange={(e) => setImageHeight(Number(e.target.value))}
                className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-3 py-2" />
            </label>
            <label>Steps
              <input type="number" min={1} max={100} value={imageSteps} onChange={(e) => setImageSteps(Number(e.target.value))}
                className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-3 py-2" />
            </label>
            <label>Guidance
              <input type="number" min={0} max={20} step={0.1} value={imageGuidance} onChange={(e) => setImageGuidance(Number(e.target.value))}
                className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-3 py-2" />
            </label>
          </div>
        )}
        {promptHistory.length > 0 && (
          <div className="mb-4">
            <div className="text-xs text-gray-500 mb-2">Previous examples</div>
            <div className="flex flex-wrap gap-2">
              {promptHistory.map((prompt) => (
                <button key={prompt} onClick={() => setInput(prompt)} title={prompt}
                  className="max-w-full truncate px-3 py-1.5 text-xs rounded-full border border-gray-700 bg-gray-800 text-gray-300 hover:border-blue-500 hover:text-white">
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}
        <button onClick={saveAndRun} disabled={busy} className="px-4 py-2 bg-blue-600 rounded text-sm hover:bg-blue-500 disabled:opacity-50">
          {busy ? 'Starting...' : 'Save and run'}
        </button>
        {message && <p className="text-sm text-yellow-400 mt-3">{message}</p>}
        </div>
        <div className="min-w-0" style={{ flex: '1 1 0%', minWidth: 0 }}>
          <RunCanvas run={activeRun} />
        </div>
      </div>
      </div>

      {savedTests.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 mb-6">
          <h3 className="font-semibold mb-3">Saved {modality} tests</h3>
          <div className="space-y-2">
            {visibleSavedTests.map(({ profile, testCase }) => (
              <div key={testCase.id} className="flex items-center justify-between bg-gray-800 rounded p-3 text-sm">
                <span>{profile.name} · {testCase.name}</span>
                <div className="flex gap-2">
                  <button onClick={() => runExisting(profile, testCase)} className="px-2 py-1 bg-gray-700 rounded hover:bg-gray-600">Run</button>
                  <button onClick={() => removeCase(testCase)} className="px-2 py-1 bg-red-700 rounded hover:bg-red-600">Delete</button>
                </div>
              </div>
            ))}
          </div>
          <Pagination page={savedPage} pageCount={savedPageCount} onChange={setSavedPage} />
        </div>
      )}

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5">
        <h3 className="font-semibold mb-3">Run history</h3>
        <div className="space-y-2">
          {runs.length === 0 && <p className="text-sm text-gray-500">No multimodal runs yet.</p>}
          {visibleRuns.map((run) => <RunRow key={run.id} run={run} onCancel={() => api.cancelMultimodalRun(run.id).then(reloadRuns)} />)}
        </div>
        <Pagination page={historyPage} pageCount={historyPageCount} onChange={setHistoryPage} />
      </div>
    </div>
  );
}

function RunRow({ run, onCancel }: { run: MultimodalRun; onCancel: () => void }) {
  const artifact = run.result?.artifacts?.[0];
  return (
    <div className="border-b border-gray-800 py-3 text-sm">
      <div className="flex items-center justify-between">
        <span><span className="font-medium">{run.modality}</span> <span className="text-gray-500">· {new Date(run.created_at).toLocaleString()}</span></span>
        {['queued', 'running'].includes(run.state) && <button onClick={onCancel} className="text-xs text-red-400 hover:text-red-300">Cancel</button>}
      </div>
      <div className="text-xs text-gray-400 mt-1">{run.state}{run.error ? ` — ${run.error}` : ''}</div>
      {run.result && <div className="text-xs text-green-400 mt-1">{Object.entries(run.result.metrics).map(([k, v]) => `${k}: ${v}`).join(' · ')}</div>}
      {run.result?.assertions && <div className={`text-xs mt-1 ${run.result.assertions.passed ? 'text-green-400' : 'text-red-400'}`}>
        Quality assertions: {run.result.assertions.passed ? 'passed' : 'failed'}
      </div>}
      {run.result?.transcript && <p className="text-sm text-gray-300 mt-2">{run.result.transcript}</p>}
      {artifact && <a href={artifactUrl(artifact.url)} target="_blank" rel="noreferrer" className="inline-block text-xs text-blue-400 hover:underline mt-2">Open {artifact.filename}</a>}
    </div>
  );
}

function Pagination({ page, pageCount, onChange }: {
  page: number;
  pageCount: number;
  onChange: (page: number) => void;
}) {
  if (pageCount <= 1) return null;
  return (
    <div className="flex items-center justify-end gap-2 mt-4 text-xs">
      <button disabled={page === 1} onClick={() => onChange(page - 1)}
        className="px-2 py-1 rounded bg-gray-800 text-gray-300 disabled:opacity-40">
        Previous
      </button>
      <span className="text-gray-500">Page {page} of {pageCount}</span>
      <button disabled={page === pageCount} onClick={() => onChange(page + 1)}
        className="px-2 py-1 rounded bg-gray-800 text-gray-300 disabled:opacity-40">
        Next
      </button>
    </div>
  );
}

function RunCanvas({ run }: { run: MultimodalRun | null }) {
  const [now, setNow] = useState(Date.now());
  const artifact = run?.result?.artifacts?.[0];
  const isActive = run && ['queued', 'running', 'cancelling'].includes(run.state);
  useEffect(() => {
    if (!isActive) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [isActive]);
  const elapsedSeconds = run
    ? Math.max(0, Math.round(((run.finished_at ? new Date(run.finished_at).getTime() : now)
      - new Date(run.started_at || run.created_at).getTime()) / 1000))
    : 0;
  const duration = `${Math.floor(elapsedSeconds / 60)}m ${elapsedSeconds % 60}s`;
  const timeout = run?.timeout_sec || 120;
  const progressWidth = Math.min(100, Math.max(4, (elapsedSeconds / timeout) * 100));

  return (
    <div className={`min-h-64 rounded-lg border p-5 flex flex-col md:sticky md:top-4 ${
      isActive ? 'border-blue-700 bg-blue-950/20 shadow-lg shadow-blue-950/20' : 'border-gray-800 bg-gray-950'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <h4 className="font-semibold">Test canvas</h4>
        {run && <span className="text-xs text-gray-500">{run.modality}</span>}
      </div>
      {!run && (
        <div className="flex-1 flex items-center justify-center text-sm text-gray-600 text-center">
          Run a test to open the live result canvas.
        </div>
      )}
      {run && isActive && (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center">
          <div className="h-10 w-10 rounded-full border-2 border-blue-400 border-t-transparent animate-spin" />
          <div>
            <p className="text-blue-300 capitalize">{run.progress?.stage || run.state}…</p>
            <p className="text-sm text-gray-300 mt-1">{run.progress?.detail || 'Preparing the adapter'}</p>
            <p className="text-xs text-gray-500 mt-1">Elapsed {duration} · timeout {timeout}s</p>
            <div className="w-full h-1.5 bg-gray-800 rounded-full mt-3 overflow-hidden">
              <div className="h-full bg-blue-500 transition-all" style={{ width: `${progressWidth}%` }} />
            </div>
            <p className="text-[11px] text-gray-600 mt-1">Progress is time-based; generation speed depends on the model and hardware.</p>
          </div>
        </div>
      )}
      {run && !isActive && (
        <div className="space-y-3 text-sm">
          <div className={`inline-flex rounded-full px-2 py-1 text-xs ${
            run.state === 'succeeded' ? 'bg-green-900/40 text-green-300' : 'bg-red-900/40 text-red-300'
          }`}>
            {run.state}
          </div>
          <p className="text-xs text-gray-500">Completed in {duration}</p>
          {run.error && <p className="text-red-300">{run.error}</p>}
          {run.result && (
            <>
              <div className="rounded border border-gray-800 p-3 text-xs text-gray-300">
                {Object.entries(run.result.metrics).map(([key, value]) => (
                  <div key={key} className="flex justify-between gap-4 py-1">
                    <span className="text-gray-500">{key}</span><span>{String(value)}</span>
                  </div>
                ))}
              </div>
              {run.result.transcript && <p className="text-gray-300 whitespace-pre-wrap">{run.result.transcript}</p>}
              {run.result.assertions && (
                <p className={run.result.assertions.passed ? 'text-green-400' : 'text-red-400'}>
                  Quality assertions: {run.result.assertions.passed ? 'passed' : 'failed'}
                </p>
              )}
              {artifact && (
                <div>
                  {artifact.mime.startsWith('image/') && (
                    <img src={artifactUrl(artifact.url)} alt={artifact.filename} className="max-h-56 w-full rounded object-contain bg-black" />
                  )}
                  <a href={artifactUrl(artifact.url)} target="_blank" rel="noreferrer" className="inline-block text-xs text-blue-400 hover:underline mt-2">
                    Open {artifact.filename}
                  </a>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
