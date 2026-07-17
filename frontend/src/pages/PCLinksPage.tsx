import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { PCLink, PCLinkTestResult, PairingCode, DiscoveredDevice } from '../types';

export default function PCLinksPage() {
  const [links, setLinks] = useState<PCLink[]>([]);
  const [name, setName] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState('');
  const [testResult, setTestResult] = useState<PCLinkTestResult | null>(null);
  const [pairing, setPairing] = useState<PairingCode | null>(null);
  const [pairingUrl, setPairingUrl] = useState('');
  const [pairingCode, setPairingCode] = useState('');
  const [pairingBusy, setPairingBusy] = useState(false);
  const [discovered, setDiscovered] = useState<DiscoveredDevice[]>([]);

  const loadLinks = async () => {
    const data = await api.listPCLinks();
    setLinks(data.links);
  };

  useEffect(() => {
    loadLinks().catch((e) => setMessage(e.message));
  }, []);

  const handleTest = async () => {
    if (!baseUrl.trim()) {
      setMessage('Enter a PC URL first.');
      return;
    }
    setTesting(true);
    setMessage('');
    setTestResult(null);
    try {
      const result = await api.testPCLink(baseUrl);
      setTestResult(result);
    } catch (e: any) {
      setMessage(e.message);
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    if (!name.trim() || !baseUrl.trim()) {
      setMessage('Name and URL are required.');
      return;
    }
    setLoading(true);
    setMessage('');
    try {
      const result = await api.savePCLink(name, baseUrl);
      setTestResult(result.test);
      setName('');
      setBaseUrl('');
      await loadLinks();
      setMessage(`Linked ${result.link.name}.`);
    } catch (e: any) {
      setMessage(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async (id: string) => {
    setLoading(true);
    setMessage('');
    try {
      await api.removePCLink(id);
      await loadLinks();
    } catch (e: any) {
      setMessage(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCreatePairingCode = async () => {
    setPairingBusy(true);
    setMessage('');
    try {
      setPairing(await api.createPairingCode());
      setMessage('Show this code on the other Avalon device, then connect from that device.');
    } catch (e: any) {
      setMessage(e.message);
    } finally {
      setPairingBusy(false);
    }
  };

  const handlePair = async () => {
    if (!pairingUrl.trim() || !pairingCode.trim()) {
      setMessage('Enter the other device URL and its pairing code.');
      return;
    }
    setPairingBusy(true);
    setMessage('');
    try {
      const result = await api.connectPairing(pairingUrl, '', pairingCode, name);
      await loadLinks();
      setPairingCode('');
      setPairing(null);
      setMessage(`Paired with ${result.device.device_name}.`);
    } catch (e: any) {
      setMessage(e.message);
    } finally {
      setPairingBusy(false);
    }
  };

  const handleDiscover = async () => {
    setPairingBusy(true);
    setMessage('');
    try {
      const result = await api.discoverPairing();
      setDiscovered(result.devices);
      if (result.devices.length === 0) setMessage('No Avalon devices found on this network.');
    } catch (e: any) {
      setMessage(e.message);
    } finally {
      setPairingBusy(false);
    }
  };

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">PC Links</h2>
      <p className="text-sm text-gray-500 mb-6">
        Link another Avalon or OpenAI-compatible PC so its models appear in this Avalon gateway as
        <span className="font-mono text-gray-300"> pc:&lt;pc-id&gt;:&lt;model-id&gt;</span>.
      </p>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <section className="bg-gray-800 rounded-lg p-4 border border-blue-800 xl:col-span-2">
          <h3 className="text-sm font-medium mb-2">Accountless Device Pairing</h3>
          <p className="text-xs text-gray-400 mb-4">
            Pair directly over your local network with a short-lived code. No account or cloud service is used.
            Private credentials are kept in the operating system secure storage.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <button
                onClick={handleDiscover}
                disabled={pairingBusy}
                className="mr-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded text-sm"
              >
                Find Devices
              </button>
              <button
                onClick={handleCreatePairingCode}
                disabled={pairingBusy}
                className="px-3 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded text-sm"
              >
                {pairingBusy ? 'Working...' : 'Generate Pairing Code'}
              </button>
              {pairing && (
                <div className="mt-3 rounded bg-gray-900 p-3">
                  <div className="text-xs text-gray-500">Give this code to the other device</div>
                  <div className="font-mono text-2xl tracking-widest text-blue-300 mt-1">{pairing.code}</div>
                  <div className="text-[11px] text-gray-500 mt-1">
                    Expires {new Date(pairing.expires_at * 1000).toLocaleTimeString()}
                  </div>
                </div>
              )}
              {discovered.length > 0 && (
                <div className="mt-3 space-y-1">
                  {discovered.map((device) => (
                    <button
                      key={device.device_id}
                      onClick={() => setPairingUrl(device.base_url)}
                      className="block w-full rounded bg-gray-900 px-3 py-2 text-left text-xs hover:bg-gray-700"
                    >
                      <span className="text-gray-200">{device.name}</span>
                      <span className="ml-2 font-mono text-gray-500">{device.base_url}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className="space-y-2">
              <input
                value={pairingUrl}
                onChange={(e) => setPairingUrl(e.target.value)}
                placeholder="http://other-device:8771"
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm outline-none focus:border-blue-500"
              />
              <input
                value={pairingCode}
                onChange={(e) => setPairingCode(e.target.value.toUpperCase())}
                placeholder="Enter the other device's code"
                maxLength={8}
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm font-mono tracking-widest outline-none focus:border-blue-500"
              />
              <button
                onClick={handlePair}
                disabled={pairingBusy}
                className="px-3 py-2 bg-green-700 hover:bg-green-600 disabled:opacity-50 rounded text-sm"
              >
                Pair Device
              </button>
            </div>
          </div>
        </section>

        <section className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-medium mb-3">Add Linked PC</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Name</label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Gaming PC"
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Base URL</label>
              <input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="http://192.168.1.50:8787"
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm outline-none focus:border-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">The remote PC must expose an OpenAI-compatible <span className="font-mono">/v1/models</span> endpoint.</p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleTest}
                disabled={testing || loading}
                className="px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded text-sm"
              >
                {testing ? 'Testing...' : 'Test'}
              </button>
              <button
                onClick={handleSave}
                disabled={testing || loading}
                className="px-3 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded text-sm"
              >
                {loading ? 'Saving...' : 'Save Link'}
              </button>
            </div>
          </div>

          {testResult && (
            <div className={`mt-4 rounded border p-3 text-sm ${testResult.ok ? 'border-green-700 bg-green-900/20 text-green-200' : 'border-red-700 bg-red-900/20 text-red-200'}`}>
              {testResult.ok ? (
                <div>Connected. Found {testResult.model_count || 0} model(s).</div>
              ) : (
                <div>{testResult.error || 'Connection failed.'}</div>
              )}
            </div>
          )}

          {message && <p className="text-sm text-gray-400 mt-4">{message}</p>}
        </section>

        <section className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-medium mb-3">Linked PCs</h3>
          {links.length === 0 ? (
            <p className="text-sm text-gray-500">No linked PCs yet.</p>
          ) : (
            <div className="space-y-3">
              {links.map((link) => (
                <div key={link.id} className="bg-gray-900 border border-gray-700 rounded p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="font-medium">{link.name}</div>
                      <div className="text-xs text-gray-500 font-mono mt-1">{link.base_url}</div>
                      <div className="text-xs text-gray-600 mt-1">Prefix: <span className="font-mono">pc:{link.id}:</span></div>
                    </div>
                    <button
                      onClick={() => handleRemove(link.id)}
                      disabled={loading}
                      className="px-2 py-1 bg-red-800 hover:bg-red-700 disabled:opacity-50 rounded text-xs"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
