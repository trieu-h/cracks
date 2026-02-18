import React, { useState, useEffect } from 'react';
import { Play, Square, RotateCcw, Activity } from 'lucide-react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { LED } from '../components/ui/LED';
import { useWebSocket } from '../hooks/useWebSocket';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { 
  startTraining, 
  stopTraining, 
  getDatasets, 
  getTrainingSessions,
  getTrainingMetrics
} from '../api';

const Training: React.FC = () => {
  const [config, setConfig] = useState({
    model_type: 'yolo',
    model: 'yolo11n-seg.pt',
    dataset_yaml: '',
    epochs: 100,
    imgsz: 640,
    batch: 16,
    device: '0',
    lr0: 0.01,
    grad_accum_steps: 4,
  });
  
  const [datasets, setDatasets] = useState<any[]>([]);
  const [sessions, setSessions] = useState<any[]>([]);
  const [activeSession, setActiveSession] = useState<any>(null);
  const [metrics, setMetrics] = useState<any[]>([]);

  useEffect(() => {
    getDatasets().then(res => setDatasets(res.data));
    loadSessions();
  }, []);

  // Load existing metrics when active session changes
  useEffect(() => {
    if (activeSession?.id) {
      getTrainingMetrics(activeSession.id).then(res => {
        if (res.data && Array.isArray(res.data)) {
          setMetrics(res.data);
        }
      });
    } else {
      setMetrics([]);
    }
  }, [activeSession?.id]);

  const loadSessions = () => {
    getTrainingSessions().then(res => {
      setSessions(res.data);
      const active = res.data.find((s: any) => s.status === 'running');
      setActiveSession(active);
    });
  };

  // WebSocket for live training updates
  useWebSocket(
    activeSession ? `/ws/training/${activeSession.id}` : null,
    (data) => {
      console.log('WebSocket message received:', data);
      if (data.type === 'epoch') {
        console.log('Epoch update:', data.epoch, data.metrics);
        setMetrics((prev: any[]) => [...prev, data.metrics]);
        // Update activeSession current_epoch in real-time
        setActiveSession((prev: any) => prev ? {
          ...prev,
          current_epoch: data.epoch
        } : null);
      } else if (data.type === 'complete') {
        console.log('Training complete');
        loadSessions();
      }
    }
  );

  const handleStart = async () => {
    const res = await startTraining(config);
    if (res.data.success) {
      setMetrics([]);
      loadSessions();
    }
  };

  const handleStop = async () => {
    if (activeSession) {
      await stopTraining(activeSession.id);
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

            {/* Model Selection */}
            {config.model_type === 'yolo' ? (
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
            ) : (
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-stone-500 mb-2 block">RF-DETR Model (Segmentation)</label>
                  <select
                    value={config.model}
                    onChange={(e) => setConfig({...config, model: e.target.value})}
                    className="input-clean w-full"
                  >
                    <option value="RFDETRSegNano">RF-DETR Seg Nano</option>
                    <option value="RFDETRSegSmall">RF-DETR Seg Small</option>
                    <option value="RFDETRSegMedium">RF-DETR Seg Medium</option>
                    <option value="RFDETRSegLarge">RF-DETR Seg Large</option>
                    <option value="RFDETRSegXLarge">RF-DETR Seg XLarge</option>
                    <option value="RFDETRSeg2XLarge">RF-DETR Seg 2XLarge</option>
                  </select>
                  <p className="text-xs text-stone-500 mt-1">
                    All RF-DETR models are configured for instance segmentation
                  </p>
                </div>
              </div>
            )}

            {/* RF-DETR Info Panel */}
            {config.model_type === 'rfdetr' && (
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
                    {ds.name} ({ds.train_images} images)
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
              </div>
              <div>
                <label className="text-sm text-stone-500 mb-2 block">Batch Size</label>
                <input
                  type="number"
                  value={config.batch}
                  onChange={(e) => setConfig({...config, batch: parseInt(e.target.value)})}
                  className="input-clean w-full"
                />
              </div>
              {config.model_type === 'yolo' && (
                <div>
                  <label className="text-sm text-stone-500 mb-2 block">Image Size</label>
                  <input
                    type="number"
                    value={config.imgsz}
                    onChange={(e) => setConfig({...config, imgsz: parseInt(e.target.value)})}
                    className="input-clean w-full"
                  />
                </div>
              )}
              <div>
                <label className="text-sm text-stone-500 mb-2 block">Learning Rate</label>
                <input
                  type="number"
                  step="0.001"
                  value={config.lr0}
                  onChange={(e) => setConfig({...config, lr0: parseFloat(e.target.value)})}
                  className="input-clean w-full"
                />
              </div>
              {config.model_type === 'rfdetr' && (
                <div>
                  <label className="text-sm text-stone-500 mb-2 block">Gradient Accumulation</label>
                  <input
                    type="number"
                    value={config.grad_accum_steps}
                    onChange={(e) => setConfig({...config, grad_accum_steps: parseInt(e.target.value)})}
                    className="input-clean w-full"
                  />
                </div>
              )}
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

        {/* Live Monitor */}
        <Panel title="Live Monitor">
          {activeSession ? (
            <div className="space-y-4">
                <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <LED color="orange" pulse />
                  <div>
                    <span className="font-mono text-green-400">
                      Session #{activeSession.id}
                    </span>
                    <span className="ml-2 text-xs text-amber-400 animate-pulse">
                      TRAINING
                    </span>
                  </div>
                </div>
                <span className="text-sm text-stone-500">
                  {activeSession.current_epoch}/{activeSession.total_epochs} epochs
                </span>
              </div>

              {/* Progress Bar */}
              <div className="progress-clean">
                <div 
                  className="progress-clean-bar"
                  style={{ 
                    width: `${(activeSession.current_epoch / activeSession.total_epochs) * 100}%` 
                  }}
                />
              </div>

              {/* Loss Chart */}
              {metrics.length > 0 && (
                <div className="mt-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Activity size={16} className="text-green-400" />
                    <span className="text-sm font-medium text-stone-300">Loss Curve</span>
                    <span className="text-xs text-stone-500">
                      ({metrics.length} epochs recorded)
                    </span>
                  </div>
                  <div className="h-48 bg-stone-900/50 rounded-lg border border-stone-800 p-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={metrics} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#292524" />
                        <XAxis 
                          dataKey="epoch" 
                          stroke="#57534e" 
                          fontSize={10}
                          tickLine={false}
                        />
                        <YAxis 
                          stroke="#57534e" 
                          fontSize={10}
                          tickLine={false}
                          domain={['auto', 'auto']}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#1c1917', 
                            border: '1px solid #292524',
                            borderRadius: '8px',
                            fontSize: '12px'
                          }}
                          itemStyle={{ color: '#e7e5e4' }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="box_loss" 
                          stroke="#22c55e" 
                          strokeWidth={2}
                          dot={false}
                          name="Box Loss"
                        />
                        {(metrics[0]?.cls_loss !== undefined || metrics[0]?.class_loss !== undefined) && (
                          <Line 
                            type="monotone" 
                            dataKey={metrics[0]?.cls_loss !== undefined ? "cls_loss" : "class_loss"}
                            stroke="#3b82f6" 
                            strokeWidth={2}
                            dot={false}
                            name="Cls Loss"
                          />
                        )}
                        {metrics[0]?.dfl_loss > 0 && (
                          <Line 
                            type="monotone" 
                            dataKey="dfl_loss" 
                            stroke="#f59e0b" 
                            strokeWidth={2}
                            dot={false}
                            name="DFL Loss"
                          />
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  
                  {/* Legend */}
                  <div className="flex gap-4 mt-2 text-xs">
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-0.5 bg-green-500 rounded" />
                      <span className="text-stone-400">Box Loss</span>
                    </div>
                    {(metrics[0]?.cls_loss !== undefined || metrics[0]?.class_loss !== undefined) && (
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-0.5 bg-blue-500 rounded" />
                        <span className="text-stone-400">Cls Loss</span>
                      </div>
                    )}
                    {metrics[0]?.dfl_loss > 0 && (
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-0.5 bg-amber-500 rounded" />
                        <span className="text-stone-400">DFL Loss</span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Current Metrics */}
              {metrics.length > 0 && (
                <div className="grid grid-cols-3 gap-2 mt-4 p-3 bg-stone-800/30 rounded-lg">
                  <div className="text-center">
                    <div className="text-xs text-stone-500 mb-1">Box Loss</div>
                    <div className="font-mono text-sm text-green-400">
                      {(metrics[metrics.length - 1].box_loss ?? metrics[metrics.length - 1].loss)?.toFixed(4)}
                    </div>
                  </div>
                  {(metrics[metrics.length - 1].cls_loss !== undefined || metrics[metrics.length - 1].class_loss !== undefined) && (
                    <div className="text-center">
                      <div className="text-xs text-stone-500 mb-1">Cls Loss</div>
                      <div className="font-mono text-sm text-blue-400">
                        {(metrics[metrics.length - 1].cls_loss ?? metrics[metrics.length - 1].class_loss)?.toFixed(4)}
                      </div>
                    </div>
                  )}
                  {metrics[metrics.length - 1].dfl_loss > 0 && (
                    <div className="text-center">
                      <div className="text-xs text-stone-500 mb-1">DFL Loss</div>
                      <div className="font-mono text-sm text-amber-400">
                        {metrics[metrics.length - 1].dfl_loss?.toFixed(4)}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-stone-600">
              <div className="relative">
                <RotateCcw size={64} className="mb-4 opacity-20" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-2 h-2 bg-stone-700 rounded-full" />
                </div>
              </div>
              <p className="text-lg font-medium text-stone-500">Ready to Train</p>
              <p className="text-sm mt-2 text-stone-600 text-center max-w-xs">
                Configure your model and dataset on the left, then click "Start Training" to begin
              </p>
              <div className="mt-6 flex gap-2 text-xs text-stone-700">
                <span className="flex items-center gap-1">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full" />
                  Live metrics
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full" />
                  Loss curves
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-1.5 h-1.5 bg-amber-500 rounded-full" />
                  Real-time updates
                </span>
              </div>
            </div>
          )}
        </Panel>
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
                  />
                  <div>
                    <div className="font-mono text-sm">Session #{session.id}</div>
                    <div className="text-xs text-stone-600">
                      {session.model_type.toUpperCase()} • {session.current_epoch}/{session.total_epochs} epochs
                    </div>
                  </div>
                </div>
                <span className={`text-xs px-2 py-1 rounded ${
                  session.status === 'running' ? 'bg-amber-900/20 text-amber-400' :
                  session.status === 'completed' ? 'bg-green-900/20 text-green-400' :
                  'bg-red-900/20 text-red-400'
                }`}>
                  {session.status.toUpperCase()}
                </span>
              </div>
            ))
          )}
        </div>
      </Panel>
    </div>
  );
};

export default Training;
