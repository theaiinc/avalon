import GPUPage from './GPUPage';
import DriversPage from './DriversPage';

export default function HardwarePage() {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Hardware</h2>
      <GPUPage />
      <div className="mt-10 border-t border-gray-800 pt-8">
        <DriversPage />
      </div>
    </div>
  );
}
