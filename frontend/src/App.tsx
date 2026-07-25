import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import HardwarePage from './pages/HardwarePage';
import ModelsPage from './pages/ModelsPage';
import BenchmarkPage from './pages/BenchmarkPage';
import APIPage from './pages/APIPage';
import PCLinksPage from './pages/PCLinksPage';
import MultimodalPage from './pages/MultimodalPage';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/hardware" element={<HardwarePage />} />
        <Route path="/gpus" element={<Navigate to="/hardware" replace />} />
        <Route path="/drivers" element={<Navigate to="/hardware" replace />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/benchmark" element={<BenchmarkPage />} />
        <Route path="/multimodal" element={<MultimodalPage />} />
        <Route path="/results" element={<Navigate to="/benchmark?tab=results" replace />} />
        <Route path="/api-server" element={<APIPage />} />
        <Route path="/agents" element={<Navigate to="/api-server" replace />} />
        <Route path="/pc-links" element={<PCLinksPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
