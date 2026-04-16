import React, { useState, useEffect } from 'react';
import { Play, Square } from 'lucide-react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { LED } from '../components/ui/LED';
import {
  startTraining,
  stopTraining,
  resumeTraining,
  getDatasets,
  getTrainingSessions
} from '../api';
import { LiveMonitor } from '../components/training/LiveMonitor';

const Training: React.FC = () => {
  const [config, setConfig] = useState({
    model_type: 'yolov26',
    model: 'yolo26n-seg.pt',
    dataset_yaml: '',
    epochs: 100,
    imgsz: 640,
    batch: 16,
    lr0: 0.01,
  });

  const [datasets, setDatasets] = useState<any[]>([]);
  const [sessions, setSessions] = useState<any[]>([]);
  const [activeSession, setActiveSession] = useState<any>(null);

  useEffect(() => {
    getDatasets().then(res => setDatasets(res.data));
    loadSessions();
  }, []);

  const loadSessions = () => {
    getTrainingSessions().then(res => {
      setSessions(res.data);
      const active = res.data.find((s: any) => s.status === 'running');
      setActiveSession(active);
    });
  };

  const handleStart = async () => {
    const res = await startTraining(config);
    if (res.data.success) {
      loadSessions();
    }
  };

  const handleStop = async () => {
    if (activeSession) {
      await stopTraining(activeSession.id);
      loadSessions();
    }
  };

  const handleResume = async (sessionId: string) => {
    const res = await resumeTraining(sessionId);
    if (res.data.success) {
      loadSessions();
    }
  };

  return (
    <div className="space-y-6 h-full overflow-auto">
      <div className="grid grid-cols-2 gap-6">
        {/* Configuration Panel */}
        <Panel title="Training Configuration">
          <div className="space-y-4">
            {/* Model Type Toggle */}
            <div>
              <label className="text-sm mb-2 block" style={{ color: 'var(--text-muted)' }}>Model Type</label>
              <div 
                className="flex gap-2 p-1 rounded-xl"
                style={{ background: 'var(--bg-tertiary)' }}
              >
                <button
                  onClick={() => setConfig({
                    ...config,
                    model_type: 'yolov11',
                    model: 'yolo11n-seg.pt',
                    epochs: 100,
                    batch: 16,
                    lr0: 0.01
                  })}
                  className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all ${
                    config.model_type === 'yolov11'
                      ? 'shadow-lg'
                      : ''
                  }`}
                  style={
                    config.model_type === 'yolov11'
                      ? { background: 'var(--accent-primary)', color: 'white' }
                      : { color: 'var(--text-muted)', background: 'transparent' }
                  }
                  onMouseEnter={(e) => {
                    if (config.model_type !== 'yolov11') {
                      e.currentTarget.style.background = 'var(--bg-hover)';
                      e.currentTarget.style.color = 'var(--text-secondary)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (config.model_type !== 'yolov11') {
                      e.currentTarget.style.background = 'transparent';
                      e.currentTarget.style.color = 'var(--text-muted)';
                    }
                  }}
                >
                  YOLOv11
                </button>
                <button
                  onClick={() => setConfig({
                    ...config,
                    model_type: 'yolov26',
                    model: 'yolo26n.pt',
                    epochs: 100,
                    batch: 16,
                    lr0: 0.01
                  })}
                  className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all ${
                    config.model_type === 'yolov26'
                      ? 'shadow-lg'
                      : ''
                  }`}
                  style={
                    config.model_type === 'yolov26'
                      ? { background: 'var(--accent-primary)', color: 'white' }
                      : { color: 'var(--text-muted)', background: 'transparent' }
                  }
                  onMouseEnter={(e) => {
                    if (config.model_type !== 'yolov26') {
                      e.currentTarget.style.background = 'var(--bg-hover)';
                      e.currentTarget.style.color = 'var(--text-secondary)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (config.model_type !== 'yolov26') {
                      e.currentTarget.style.background = 'transparent';
                      e.currentTarget.style.color = 'var(--text-muted)';
                    }
                  }}
                >
                  YOLOv26
                </button>
              </div>
            </div>

            {/* Model Selection and Tips - YOLOv11 */}
            {config.model_type === 'yolov11' ? (
              <div className="space-y-4">
                <div>
                  <label className="text-sm mb-2 block" style={{ color: 'var(--text-muted)' }}>YOLOv11 Model</label>
                  <select
                    value={config.model}
                    onChange={(e) => setConfig({ ...config, model: e.target.value })}
                    className="input-clean w-full"
                  >
                    <option value="yolo11n-seg.pt">YOLO11 Nano Segment</option>
                    <option value="yolo11s-seg.pt">YOLO11 Small Segment</option>
                    <option value="yolo11m-seg.pt">YOLO11 Medium Segment</option>
                    <option value="yolo11l-seg.pt">YOLO11 Large Segment</option>
                    <option value="yolo11x-seg.pt">YOLO11 Extra Large Segment</option>
                  </select>
                </div>

                {/* YOLOv11 Best Practices - Fixed Contrast */}
                <div 
                  className="p-4 rounded-xl"
                  style={{ 
                    background: 'var(--success-bg)', 
                    border: '1px solid var(--border-primary)'
                  }}
                >
                  <div className="flex items-start gap-3">
                    <div 
                      className="p-1.5 rounded-lg mt-0.5"
                      style={{ background: 'var(--success-bg)' }}
                    >
                      <svg 
                        className="w-4 h-4" 
                        style={{ color: 'var(--success-text)' }}
                        fill="none" 
                        viewBox="0 0 24 24" 
                        stroke="currentColor"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div>
                      <h4 
                        className="text-sm font-medium mb-2"
                        style={{ color: 'var(--success-text)' }}
                      >
                        YOLO Training Best Practices
                      </h4>
                      <div className="space-y-3 text-xs" style={{ color: 'var(--text-secondary)' }}>
                        <div>
                          <span className="font-semibold" style={{ color: 'var(--success-text)' }}>Model Size Selection:</span>
                          <ul className="mt-1 ml-3 space-y-0.5">
                            <li>• Nano: Fastest, lowest accuracy - good for testing</li>
                            <li>• Small/Medium: Balance of speed and accuracy</li>
                            <li>• Large/XLarge: Best accuracy, requires more GPU memory</li>
                          </ul>
                        </div>

                        <div>
                          <span className="font-semibold" style={{ color: 'var(--success-text)' }}>Recommended Settings:</span>
                          <ul className="mt-1 ml-3 space-y-0.5">
                            <li>• Epochs: 100-300 (segmentation needs more than detection)</li>
                            <li>• Batch Size: 8-16 (reduce if out of memory)</li>
                            <li>• Image Size: 640 (standard) or 1280 (higher quality)</li>
                            <li>• Learning Rate: 0.01 (default works well)</li>
                          </ul>
                        </div>

                        <div>
                          <span className="font-semibold" style={{ color: 'var(--success-text)' }}>Pro Tips:</span>
                          <ul className="mt-1 ml-3 space-y-0.5">
                            <li>• Start with 100 epochs, increase if loss is still decreasing</li>
                            <li>• Use smaller batches (4-8) for larger models (L/XL)</li>
                            <li>• Higher image size = better accuracy but slower training</li>
                            <li>• Monitor the loss curve - should steadily decrease</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : null}

            {/* YOLOv26 Model Block */}
            {config.model_type === 'yolov26' && (
              <div className="space-y-4">
                <div>
                  <label className="text-sm mb-2 block" style={{ color: 'var(--text-muted)' }}>YOLOv26 Model</label>
                  <select
                    value={config.model}
                    onChange={(e) => setConfig({ ...config, model: e.target.value })}
                    className="input-clean w-full"
                  >
                    <option value="yolo26n-seg.pt">YOLOv26 Nano Segment</option>
                    <option value="yolo26s-seg.pt">YOLOv26 Small Segment</option>
                    <option value="yolo26m-seg.pt">YOLOv26 Medium Segment</option>
                    <option value="yolo26l-seg.pt">YOLOv26 Large Segment</option>
                    <option value="yolo26x-seg.pt">YOLOv26 Extra Large Segment</option>
                  </select>
                </div>

                {/* YOLOv26 Best Practices - Fixed Contrast */}
                <div 
                  className="p-4 rounded-xl"
                  style={{ 
                    background: 'var(--bg-tertiary)', 
                    border: '1px solid var(--border-primary)'
                  }}
                >
                  <div className="flex items-start gap-3">
                    <div 
                      className="p-1.5 rounded-lg mt-0.5"
                      style={{ background: 'var(--bg-secondary)' }}
                    >
                      <svg 
                        className="w-4 h-4" 
                        style={{ color: 'var(--accent-primary)' }}
                        fill="none" 
                        viewBox="0 0 24 24" 
                        stroke="currentColor"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div>
                      <h4 
                        className="text-sm font-medium mb-2"
                        style={{ color: 'var(--accent-primary)' }}
                      >
                        YOLOv26 Training Best Practices
                      </h4>
                      <div className="space-y-3 text-xs" style={{ color: 'var(--text-secondary)' }}>
                        <div>
                          <span className="font-semibold" style={{ color: 'var(--accent-primary)' }}>Model Size Selection:</span>
                          <ul className="mt-1 ml-3 space-y-0.5">
                            <li>• Nano: Fastest, lowest accuracy - perfect for edge devices</li>
                            <li>• Small/Medium: Best balance of speed and precision</li>
                            <li>• Large/XLarge: Maximum accuracy, needs high GPU memory</li>
                          </ul>
                        </div>
                        
                        <div>
                          <span className="font-semibold" style={{ color: 'var(--accent-primary)' }}>Recommended Settings:</span>
                          <ul className="mt-1 ml-3 space-y-0.5">
                            <li>• Epochs: 100-300 (segmentation needs more iterations)</li>
                            <li>• Batch Size: 8-32 (maximize this until out of memory)</li>
                            <li>• Image Size: 640 (standard) or 1280 (high quality)</li>
                            <li>• Learning Rate: 0.01 is highly recommended</li>
                          </ul>
                        </div>

                        <div>
                          <span className="font-semibold" style={{ color: 'var(--accent-primary)' }}>Pro Tips:</span>
                          <ul className="mt-1 ml-3 space-y-0.5">
                            <li>• Uses end-to-end NMS-free paths (faster than older YOLO)</li>
                            <li>• Use smaller batches (4-8) only for XLarge models</li>
                            <li>• Automatically drops learning rate towards end of epochs</li>
                            <li>• Monitor the loss curve - it should drop very sharply initially</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Dataset Selection */}
            <div>
              <label className="text-sm mb-2 block" style={{ color: 'var(--text-muted)' }}>Dataset</label>
              <select
                value={config.dataset_yaml}
                onChange={(e) => setConfig({ ...config, dataset_yaml: e.target.value })}
                className="input-clean w-full"
              >
                <option value="">Select a dataset...</option>
                {datasets.map(ds => (
                  <option key={ds.id} value={ds.yaml_path}>
                    {ds.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Training Parameters */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm mb-2 block" style={{ color: 'var(--text-muted)' }}>Epochs</label>
                <input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                  className="input-clean w-full"
                />
                <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>Total training passes. More epochs = better training, until overfitting starts.</p>
              </div>
              <div>
                <label className="text-sm mb-2 block" style={{ color: 'var(--text-muted)' }}>Batch Size</label>
                <input
                  type="number"
                  value={config.batch}
                  onChange={(e) => setConfig({ ...config, batch: parseInt(e.target.value) })}
                  className="input-clean w-full"
                />
                <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>Images processed together. Lower this value if you encounter "Out of Memory" errors.</p>
              </div>
              <div>
                <label className="text-sm mb-2 block" style={{ color: 'var(--text-muted)' }}>Image Size</label>
                <input
                  type="number"
                  value={config.imgsz}
                  onChange={(e) => setConfig({ ...config, imgsz: parseInt(e.target.value) })}
                  className="input-clean w-full"
                />
                <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>Input resolution. 640 is standard; higher values detect smaller cracks but use more VRAM.</p>
              </div>
              <div>
                <label className="text-sm mb-2 block" style={{ color: 'var(--text-muted)' }}>Learning Rate</label>
                <input
                  type="number"
                  step="0.0001"
                  value={config.lr0}
                  onChange={(e) => setConfig({ ...config, lr0: parseFloat(e.target.value) })}
                  className="input-clean w-full"
                />
                <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>Initial step size. 0.01 is good for YOLO; use 0.0001 or lower for RF-DETR fine-tuning.</p>
              </div>
            </div>

            {/* Control Buttons */}
            <div className="flex gap-3 pt-4">
              {!activeSession ? (
                <Button primary onClick={handleStart} disabled={!config.dataset_yaml} className="flex-1">
                  <Play size={18} className="inline mr-2" />
                  Start Training
                </Button>
              ) : (
                <Button variant="danger" onClick={handleStop} className="flex-1">
                  <Square size={18} className="inline mr-2" />
                  Stop Training
                </Button>
              )}
            </div>
          </div>
        </Panel>

        {/* Live Monitor component */}
        <LiveMonitor activeSession={activeSession} />
      </div>

      {/* Training History */}
      <Panel title="Training History">
        <div className="space-y-2">
          {sessions.length === 0 ? (
            <p className="text-center py-8" style={{ color: 'var(--text-muted)' }}>No training sessions yet</p>
          ) : (
            sessions.map(session => (
              <div
                key={session.id}
                className="flex items-center justify-between py-3 px-4 rounded-lg"
                style={{ background: 'var(--bg-tertiary)' }}
              >
                <div className="flex items-center gap-3">
                  <LED
                    color={
                      session.status === 'running' ? 'orange' :
                        session.status === 'completed' ? 'green' : 'red'
                    }
                    pulse={session.status === 'running'}
                  />
                  <div>
                    <div className="font-mono text-sm leading-none mb-1" style={{ color: 'var(--text-primary)' }}>Session #{session.id}</div>
                    <div className="text-[10px] uppercase font-bold tracking-tight" style={{ color: 'var(--text-muted)' }}>
                      {session.model_type} • {session.current_epoch}/{session.total_epochs} epochs
                    </div>
                  </div>
                </div>
                
                {/* Status and Resume Button Group - Aligned Right */}
                <div className="flex items-center gap-3">
                  <span 
                    className="text-[10px] font-bold px-2 py-0.5 rounded uppercase"
                    style={
                      session.status === 'running'
                        ? { background: 'var(--warning-bg)', color: 'var(--warning-text)' }
                        : session.status === 'completed'
                        ? { background: 'var(--success-bg)', color: 'var(--success-text)' }
                        : { background: 'var(--error-bg)', color: 'var(--error-text)' }
                    }
                  >
                    {session.status}
                  </span>

                  {/* Resume Button for stopped/error sessions */}
                  {session.status !== 'running' && session.current_epoch < session.total_epochs && (
                    <Button
                      variant="ghost"
                      onClick={() => handleResume(session.id)}
                      className="opacity-50 hover:opacity-100 text-xs py-1 px-3 h-auto"
                      disabled={activeSession !== undefined && activeSession !== null}
                    >
                      Resume
                    </Button>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </Panel>
    </div>
  );
};

export default Training;
