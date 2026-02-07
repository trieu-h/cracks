import { Routes, Route } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import Training from './pages/Training';
import Prediction from './pages/Prediction';
import Datasets from './pages/Datasets';
import Models from './pages/Models';
import Settings from './pages/Settings';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/training" element={<Training />} />
        <Route path="/prediction" element={<Prediction />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/models" element={<Models />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  );
}

export default App;
