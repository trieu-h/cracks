import React from 'react';
import { Settings, Info, Github } from 'lucide-react';
import { Panel } from '../components/ui/Panel';

const SettingsPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <Panel title="Application Settings">
        <div className="space-y-6">
          <div className="flex items-center justify-between py-3 border-b border-gray-800">
            <div>
              <div className="text-gray-200">Backend URL</div>
              <div className="text-sm text-gray-500 font-mono">
                {import.meta.env.VITE_API_URL || 'http://localhost:8000/api'}
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between py-3 border-b border-gray-800">
            <div>
              <div className="text-gray-200">WebSocket URL</div>
              <div className="text-sm text-gray-500 font-mono">
                {import.meta.env.VITE_WS_URL || 'ws://localhost:8000'}
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between py-3 border-b border-gray-800">
            <div>
              <div className="text-gray-200">Theme</div>
              <div className="text-sm text-gray-500">
                Vintage Industrial (Dark)
              </div>
            </div>
            <div className="text-xs text-vintage-orange bg-vintage-orange/20 px-2 py-1 rounded">
              ACTIVE
            </div>
          </div>
        </div>
      </Panel>

      <Panel title="About">
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-lg bg-vintage-orange/20 flex items-center justify-center">
              <Settings size={32} className="text-vintage-orange" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-200">Crack Detection Lab</h3>
              <p className="text-sm text-gray-500">Version 1.0.0</p>
            </div>
          </div>

          <p className="text-gray-400 text-sm leading-relaxed">
            A vintage industrial-style laboratory interface for training and deploying 
            crack detection models using YOLO and RF-DETR architectures. Features real-time 
            GPU monitoring, WebSocket updates, and 3D visualization.
          </p>

          <div className="flex gap-4 pt-4">
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-gray-400 hover:text-vintage-orange transition-colors"
            >
              <Github size={18} />
              <span className="text-sm">GitHub Repository</span>
            </a>
          </div>

          <div className="mt-6 pt-6 border-t border-gray-800">
            <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
              <Info size={16} />
              System Information
            </h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Framework:</span>
                <span className="text-gray-300 ml-2">FastAPI + React + Three.js</span>
              </div>
              <div>
                <span className="text-gray-500">ML Backend:</span>
                <span className="text-gray-300 ml-2">Ultralytics YOLO + RF-DETR</span>
              </div>
              <div>
                <span className="text-gray-500">UI Style:</span>
                <span className="text-gray-300 ml-2">Vintage Industrial (1970s)</span>
              </div>
              <div>
                <span className="text-gray-500">Storage:</span>
                <span className="text-gray-300 ml-2">In-Memory (No Database)</span>
              </div>
            </div>
          </div>
        </div>
      </Panel>
    </div>
  );
};

export default SettingsPage;
