import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import GPUPage from './pages/GPUPage';
import DriversPage from './pages/DriversPage';
import ModelsPage from './pages/ModelsPage';
import BenchmarkPage from './pages/BenchmarkPage';
import ResultsPage from './pages/ResultsPage';
import APIPage from './pages/APIPage';
import AgentsPage from './pages/AgentsPage';
import PCLinksPage from './pages/PCLinksPage';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/gpus" element={<GPUPage />} />
        <Route path="/drivers" element={<DriversPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/benchmark" element={<BenchmarkPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="/api-server" element={<APIPage />} />
        <Route path="/agents" element={<AgentsPage />} />
        <Route path="/pc-links" element={<PCLinksPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
