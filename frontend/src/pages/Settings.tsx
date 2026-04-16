import React from 'react';
import { Settings, Info, Github } from 'lucide-react';
import { Panel } from '../components/ui/Panel';

const SettingsPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <Panel title="Application Settings">
        <div className="space-y-6">
          <div 
            className="flex items-center justify-between py-3"
            style={{ borderBottom: '1px solid var(--border-primary)' }}
          >
            <div>
              <div style={{ color: 'var(--text-secondary)' }}>Backend URL</div>
              <div className="text-sm font-mono" style={{ color: 'var(--text-muted)' }}>
                {import.meta.env.VITE_API_URL || 'http://localhost:8000/api'}
              </div>
            </div>
          </div>

          <div 
            className="flex items-center justify-between py-3"
            style={{ borderBottom: '1px solid var(--border-primary)' }}
          >
            <div>
              <div style={{ color: 'var(--text-secondary)' }}>WebSocket URL</div>
              <div className="text-sm font-mono" style={{ color: 'var(--text-muted)' }}>
                {import.meta.env.VITE_WS_URL || 'ws://localhost:8000'}
              </div>
            </div>
          </div>
        </div>
      </Panel>

      <Panel title="About">
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <div 
              className="w-16 h-16 rounded-lg flex items-center justify-center"
              style={{ background: 'var(--accent-glow)' }}
            >
              <Settings size={32} style={{ color: 'var(--accent-primary)' }} />
            </div>
            <div>
              <h3 className="text-xl font-bold" style={{ color: 'var(--text-secondary)' }}>Crack Lab</h3>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Version 1.0.0</p>
            </div>
          </div>

          <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            An interface for training and deploying 
            crack detection models using YOLO architectures. Features real-time 
            GPU monitoring and WebSocket updates.
          </p>

          <div className="flex gap-4 pt-4">
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center gap-2 transition-colors"
              style={{ color: 'var(--text-muted)' }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = 'var(--accent-primary)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = 'var(--text-muted)';
              }}
            >
              <Github size={18} />
              <span className="text-sm">GitHub Repository</span>
            </a>
          </div>

          <div 
            className="mt-6 pt-6"
            style={{ borderTop: '1px solid var(--border-primary)' }}
          >
            <h4 
              className="text-sm font-semibold mb-3 flex items-center gap-2"
              style={{ color: 'var(--text-secondary)' }}
            >
              <Info size={16} />
              System Information
            </h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Framework:</span>
                <span className="ml-2" style={{ color: 'var(--text-secondary)' }}>FastAPI + React 18</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>ML Backend:</span>
                <span className="ml-2" style={{ color: 'var(--text-secondary)' }}>Ultralytics YOLO</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Storage:</span>
                <span className="ml-2" style={{ color: 'var(--text-secondary)' }}>SQLite Database</span>
              </div>
            </div>
          </div>
        </div>
      </Panel>
    </div>
  );
};

export default SettingsPage;
