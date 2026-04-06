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
    model_type: 'yolo',
    model: 'yolo11n-seg.pt',
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
              <label className="text-sm text-stone-400 mb-2 block">Model Type</label>
              <div className="flex gap-2 p-1 bg-stone-800 rounded-xl">
                <button
                  onClick={() => setConfig({
                    ...config, 
                    model_type: 'yolo',
                    model: 'yolo11n-seg.pt',
                    epochs: 100,
                    batch: 16,
                    lr0: 0.01
                  })}
                  className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all ${
                    config.model_type === 'yolo'
                      ? 'bg-green-500 text-stone-950 shadow-lg'
                      : 'text-stone-400 hover:text-stone-200 hover:bg-stone-700/50'
                  }`}
                >
                  YOLO
                </button>
                <button
                  onClick={() => setConfig({
                    ...config, 
                    model_type: 'rfdetr',
                    model: 'RFDETRSegMedium',
                    epochs: 10,
                    batch: 4,
                    lr0: 0.0001
                  })}
                  className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all ${
                    config.model_type === 'rfdetr'
                      ? 'bg-green-500 text-stone-950 shadow-lg'
                      : 'text-stone-400 hover:text-stone-200 hover:bg-stone-700/50'
                  }`}
                >
                  RF-DETR
                </button>
              </div>
            </div>

            {/* Model Selection and Tips */}
            {config.model_type === 'yolo' ? (
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-stone-500 mb-2 block">YOLO Model</label>
                  <select
                    value={config.model}
                    onChange={(e) => setConfig({...config, model: e.target.value})}
                    className="input-clean w-full"
                  >
                    <option value="yolo11n-seg.pt">YOLO11n-seg (Nano)</option>
                    <option value="yolo11s-seg.pt">YOLO11s-seg (Small)</option>
                    <option value="yolo11m-seg.pt">YOLO11m-seg (Medium)</option>
                    <option value="yolo11l-seg.pt">YOLO11l-seg (Large)</option>
                    <option value="yolo11x-seg.pt">YOLO11x-seg (Extra Large)</option>
                  </select>
                </div>

                <div className="p-4 bg-green-900/20 border border-green-800 rounded-xl">
                  <div className="flex items-start gap-3">
                    <div className="p-1.5 rounded-lg bg-green-500/20 mt-0.5">
                      <svg className="w-4 h-4 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-green-300 mb-2">YOLO Training Best Practices</h4>
                      <div className="space-y-3 text-xs text-green-400/80">
                        <div>
                          <span className="font-semibold text-green-300">Model Size Selection:</span>
                          <ul className="mt-1 ml-3 space-y-0.5">
                            <li>• Nano: Fastest, lowest accuracy - good for testing</li>
                            <li>• Small/Medium: Balance of speed and accuracy</li>
                            <li>• Large/XLarge: Best accuracy, requires more GPU memory</li>
                          </ul>
                        </div>
                        
                        <div>
                          <span className="font-semibold text-green-300">Recommended Settings:</span>
                          <ul className="mt-1 ml-3 space-y-0.5">
                            <li>• Epochs: 100-300 (segmentation needs more than detection)</li>
                            <li>• Batch Size: 8-16 (reduce if out of memory)</li>
                            <li>• Image Size: 640 (standard) or 1280 (higher quality)</li>
                            <li>• Learning Rate: 0.01 (default works well)</li>
                          </ul>
                        </div>

                        <div>
                          <span className="font-semibold text-green-300">Pro Tips:</span>
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
            ) : (
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-stone-500 mb-2 block">RF-DETR Model</label>
                  <select
                    value={config.model}
                    onChange={(e) => setConfig({...config, model: e.target.value})}
                    className="input-clean w-full"
                  >
                    <option value="RFDETRSegNano">RF-DETR Seg Nano</option>
                    <option value="RFDETRSegSmall">RF-DETR Seg Small</option>
                    <option value="RFDETRSegMedium">RF-DETR Seg Medium</option>
                    <option value="RFDETRSegLarge">RF-DETR Seg Large</option>
                  </select>
                </div>

                <div className="p-4 bg-blue-900/20 border border-blue-800 rounded-xl">
                  <div className="flex items-start gap-3">
                    <div className="p-1.5 rounded-lg bg-blue-500/20 mt-0.5">
                      <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-blue-300 mb-1">RF-DETR Segmentation Model</h4>
                      <ul className="text-xs text-blue-400/80 space-y-1">
                        <li>• State-of-the-art transformer-based instance segmentation</li>
                        <li>• Pixel-level mask prediction for precise object boundaries</li>
                        <li>• Typically converges in fewer epochs than YOLO (5-20 recommended)</li>
                        <li>• Requires lower batch size (2-8) due to memory requirements</li>
                        <li>• Lower learning rate (1e-4) works best</li>
                      </ul>
                      <p className="text-xs text-amber-400/80 mt-2 italic">
                        Note: On macOS with Apple Silicon, RF-DETR will use CPU to avoid compatibility issues.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Dataset Selection */}
            <div>
              <label className="text-sm text-stone-500 mb-2 block">Dataset</label>
              <select
                value={config.dataset_yaml}
                onChange={(e) => setConfig({...config, dataset_yaml: e.target.value})}
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
                <label className="text-sm text-stone-500 mb-2 block">Epochs</label>
                <input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                  className="input-clean w-full"
                />
                <p className="text-[10px] text-stone-500 mt-1">Total training passes. More epochs = better training, until overfitting starts.</p>
              </div>
              <div>
                <label className="text-sm text-stone-500 mb-2 block">Batch Size</label>
                <input
                  type="number"
                  value={config.batch}
                  onChange={(e) => setConfig({...config, batch: parseInt(e.target.value)})}
                  className="input-clean w-full"
                />
                <p className="text-[10px] text-stone-500 mt-1">Images processed together. Lower this value if you encounter "Out of Memory" errors.</p>
              </div>
              <div>
                <label className="text-sm text-stone-500 mb-2 block">Image Size</label>
                <input
                  type="number"
                  value={config.imgsz}
                  onChange={(e) => setConfig({...config, imgsz: parseInt(e.target.value)})}
                  className="input-clean w-full"
                />
                <p className="text-[10px] text-stone-500 mt-1">Input resolution. 640 is standard; higher values detect smaller cracks but use more VRAM.</p>
              </div>
              <div>
                <label className="text-sm text-stone-500 mb-2 block">Learning Rate</label>
                <input
                  type="number"
                  step="0.0001"
                  value={config.lr0}
                  onChange={(e) => setConfig({...config, lr0: parseFloat(e.target.value)})}
                  className="input-clean w-full"
                />
                <p className="text-[10px] text-stone-500 mt-1">Initial step size. 0.01 is good for YOLO; use 0.0001 or lower for RF-DETR fine-tuning.</p>
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

        {/* Live Monitor component replaces the original inline panel content */}
        <LiveMonitor activeSession={activeSession} />
      </div>

      {/* Training History */}
      <Panel title="Training History">
        <div className="space-y-2">
          {sessions.length === 0 ? (
            <p className="text-stone-600 text-center py-8">No training sessions yet</p>
          ) : (
            sessions.map(session => (
              <div 
                key={session.id}
                className="flex items-center justify-between py-3 px-4 bg-stone-800/50 rounded-lg"
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
                    <div className="font-mono text-sm leading-none mb-1">Session #{session.id}</div>
                    <div className="text-[10px] text-stone-500 uppercase font-bold tracking-tight">
                      {session.model_type} • {session.current_epoch}/{session.total_epochs} epochs
                    </div>
                  </div>
                </div>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                  session.status === 'running' ? 'bg-amber-900/20 text-amber-400' :
                  session.status === 'completed' ? 'bg-green-900/20 text-green-400' :
                  'bg-red-900/20 text-red-400'
                }`}>
                  {session.status}
                </span>
                
                {/* Resume Button for stopped/error sessions */}
                {session.status !== 'running' && session.current_epoch < session.total_epochs && (
                  <Button 
                    variant="ghost" 
                    onClick={() => handleResume(session.id)}
                    className="ml-4 opacity-50 hover:opacity-100 text-xs py-1 px-3 h-auto"
                    disabled={activeSession !== undefined && activeSession !== null} // disable if another training is active
                  >
                    Resume
                  </Button>
                )}
              </div>
            ))
          )}
        </div>
      </Panel>
    </div>
  );
};

export default Training;
