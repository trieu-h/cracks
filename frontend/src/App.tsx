import { Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { Layout } from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import Training from './pages/Training';
import Detection from './pages/Detection';
import Datasets from './pages/Datasets';
import Models from './pages/Models';
import Settings from './pages/Settings';
import Evaluation from './pages/Evaluation';
import Demo from './pages/Demo';

function App() {
  return (
    <ThemeProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/training" element={<Training />} />
          <Route path="/detection" element={<Detection />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/models" element={<Models />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/demo" element={<Demo />} />
        </Routes>
      </Layout>
    </ThemeProvider>
  );
}

export default App;
