import React, { useEffect, useState } from 'react';
import { 
  Activity, 
  Cpu, 
  HardDrive, 
  Play, 
  TrendingUp, 
  Clock, 
  Layers,
  Database,
  Info
} from 'lucide-react';
import { useWebSocket } from '../hooks/useWebSocket';
import { getGPUStats, getTrainingSessions } from '../api';
// Charts temporarily hidden - recharts imports removed

const Dashboard: React.FC = () => {
  const [gpuStats, setGpuStats] = useState<any>(null);
  const [sessions, setSessions] = useState<any[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [showRunInfo, setShowRunInfo] = useState(false);
  const [currentTime, setCurrentTime] = useState<number>(Date.now());

  // Update current time every minute to refresh training time display
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(Date.now());
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  useWebSocket('/ws/system', (data) => {
    if (data.type === 'gpu_stats') {
      setGpuStats(data.data);
    }
  });

  const fetchSessions = async () => {
    try {
      const res = await getTrainingSessions();
      setSessions(res.data);
      if (res.data.length > 0 && !selectedSessionId) {
        setSelectedSessionId(res.data[0].id);
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  useEffect(() => {
    getGPUStats().then(res => setGpuStats(res.data));
    fetchSessions();
  }, []);

  const activeSession = sessions.find(s => s.status === 'running');
  const selectedSession = sessions.find(s => s.id === selectedSessionId);

  // Calculate training time for active session
  const getTrainingTime = () => {
    if (!activeSession?.start_time) return '--';
    const elapsedMs = currentTime - (activeSession.start_time * 1000);
    const elapsedHours = elapsedMs / (1000 * 60 * 60);
    if (elapsedHours < 1) {
      const elapsedMinutes = Math.round(elapsedMs / (1000 * 60));
      return `${elapsedMinutes}m`;
    }
    return `${elapsedHours.toFixed(1)}h`;
  };

  // WebSocket for live training updates
  useWebSocket(
    activeSession ? `/ws/training/${activeSession.id}` : null,
    (data) => {
      if (data.type === 'epoch') {
        setSessions(prevSessions => 
          prevSessions.map(session => 
            session.id === activeSession?.id 
              ? { 
                  ...session, 
                  current_epoch: data.epoch,
                  latest_metrics: data.metrics,
                  all_metrics: [...(session.all_metrics || []), data.metrics]
                }
              : session
          )
        );
      } else if (data.type === 'complete') {
        fetchSessions();
      }
    }
  );

  const runningCount = sessions.filter(s => s.status === 'running').length;
  const completedCount = sessions.filter(s => s.status === 'completed').length;
  const stats = selectedSession?.latest_metrics || {};
  // const metrics = selectedSession?.all_metrics || []; // Temporarily unused - charts hidden

  const formatPercent = (val: number | undefined) => {
    if (val === undefined) return '--%';
    return `${(val * 100).toFixed(1)}%`;
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-12">
      {/* Welcome & Global Stats */}
      <div className="flex items-end justify-between">
        <div>
          <p className="text-stone-500 mb-1">Welcome back</p>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-stone-900 border border-stone-800 rounded-xl flex items-center justify-center p-2 shadow-lg relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 via-pink-500/10 to-blue-500/10 group-hover:opacity-100 transition-opacity" />
              <div className="flex items-end gap-[1px] h-full w-full relative z-10">
                <div className="flex-1 bg-green-500/60 rounded-t-[1px]" style={{ height: '40%' }} />
                <div className="flex-1 bg-pink-500/60 rounded-t-[1px]" style={{ height: '80%' }} />
                <div className="flex-1 bg-blue-500/60 rounded-t-[1px]" style={{ height: '55%' }} />
              </div>
            </div>
            <h1 className="font-serif text-4xl text-stone-100">Dashboard</h1>
          </div>
        </div>
        <button 
          onClick={() => window.location.href = '/training'}
          className="btn-clean btn-primary px-6"
        >
          <Play size={16} />
          New Training
        </button>
      </div>

      {/* Top row metrics (restored) */}
      <div className="grid grid-cols-4 gap-4">
        <div className="metric-clean">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 rounded-lg bg-stone-800">
              <Layers className="w-5 h-5 text-stone-400" />
            </div>
            <span className="badge-clean badge-neutral">Total</span>
          </div>
          <div className="text-3xl font-serif text-stone-100 mb-1">{sessions.length}</div>
          <div className="text-sm text-stone-500">Training Sessions</div>
        </div>

        <div className="metric-clean">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 rounded-lg bg-amber-900/20">
              <Activity className="w-5 h-5 text-amber-500" />
            </div>
            <span className="badge-clean badge-warning">Active</span>
          </div>
          <div className="text-3xl font-serif text-stone-100 mb-1">{runningCount}</div>
          <div className="text-sm text-stone-500">Currently Running</div>
        </div>

        <div className="metric-clean">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 rounded-lg bg-green-900/20">
              <TrendingUp className="w-5 h-5 text-green-500" />
            </div>
            <span className="badge-clean badge-success">Done</span>
          </div>
          <div className="text-3xl font-serif text-stone-100 mb-1">{completedCount}</div>
          <div className="text-sm text-stone-500">Completed</div>
        </div>

        <div className="metric-clean">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 rounded-lg bg-rose-900/20">
              <Clock className="w-5 h-5 text-rose-500" />
            </div>
            <span className="text-xs text-stone-500 font-mono">{activeSession ? 'Elapsed' : 'Avg'}</span>
          </div>
          <div className="text-3xl font-serif text-stone-100 mb-1">{getTrainingTime()}</div>
          <div className="text-sm text-stone-500">Training Time</div>
        </div>
      </div>

      {/* Main Grid: Segmentation Metrics & Run Info */}
      <div className="grid grid-cols-1 gap-6">
        {/* Run Selector & Info Accordion */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
             <h2 className="text-lg font-serif text-stone-300">Detailed Analytics</h2>
            <select 
              value={selectedSessionId || ''}
              onChange={(e) => setSelectedSessionId(e.target.value)}
              className="bg-stone-900 border border-stone-800 text-stone-100 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-stone-700 min-w-[200px]"
            >
              {sessions.map(s => (
                <option key={s.id} value={s.id}>
                  Run: {s.id} ({s.status.toUpperCase()})
                </option>
              ))}
            </select>
          </div>
          <button 
             onClick={() => setShowRunInfo(!showRunInfo)}
             className="text-sm text-stone-500 hover:text-stone-300 flex items-center gap-2 transition-colors"
          >
             <Info size={14} />
             {showRunInfo ? 'Hide Run Configuration' : 'Show Run Configuration'}
          </button>
        </div>

        {showRunInfo && (
           <div className="card-clean bg-stone-900/30 border-dashed">
              <div className="grid grid-cols-4 gap-6 text-sm">
                <div>
                  <span className="text-stone-500 block mb-1 uppercase text-[10px] tracking-wider">Model Type</span>
                  <span className="text-stone-200 capitalize">{selectedSession?.model_type || '--'}</span>
                </div>
                <div>
                  <span className="text-stone-500 block mb-1 uppercase text-[10px] tracking-wider">Total Epochs</span>
                  <span className="text-stone-200">{selectedSession?.total_epochs || '--'}</span>
                </div>
                <div>
                  <span className="text-stone-500 block mb-1 uppercase text-[10px] tracking-wider">Start Time</span>
                  <span className="text-stone-200">
                    {selectedSession?.start_time ? new Date(selectedSession.start_time * 1000).toLocaleString() : '--'}
                  </span>
                </div>
                <div>
                  <span className="text-stone-500 block mb-1 uppercase text-[10px] tracking-wider">Dataset Path</span>
                  <span className="text-stone-200 truncate block max-w-[200px]" title={selectedSession?.config?.dataset_yaml || selectedSession?.config?.data || '--'}>{selectedSession?.config?.dataset_yaml || selectedSession?.config?.data || '--'}</span>
                </div>
                {(selectedSession?.config?.imgsz || selectedSession?.config?.batch) && (
                  <>
                    <div>
                      <span className="text-stone-500 block mb-1 uppercase text-[10px] tracking-wider">Image Size</span>
                      <span className="text-stone-200">{selectedSession?.config?.imgsz || '--'}</span>
                    </div>
                    <div>
                      <span className="text-stone-500 block mb-1 uppercase text-[10px] tracking-wider">Batch Size</span>
                      <span className="text-stone-200">{selectedSession?.config?.batch || '--'}</span>
                    </div>
                    <div>
                      <span className="text-stone-500 block mb-1 uppercase text-[10px] tracking-wider">Learning Rate / Opt</span>
                      <span className="text-stone-200">{selectedSession?.config?.lr0 || '--'} / {selectedSession?.config?.optimizer || '--'}</span>
                    </div>
                  </>
                )}
              </div>
           </div>
        )}

        {/* Segmentation Metrics Cards (from redesigned dashboard) */}
        <div className="grid grid-cols-5 gap-4">
          <div className="metric-clean py-4 before:bg-purple-500">
            <span className="text-stone-500 text-[10px] uppercase tracking-wider mb-1 block">F1 Score</span>
            <div className="text-2xl font-serif text-stone-100">{formatPercent(stats.f1)}</div>
          </div>
          <div className="metric-clean py-4 before:bg-blue-500">
            <span className="text-stone-500 text-[10px] uppercase tracking-wider mb-1 block">Precision</span>
            <div className="text-2xl font-serif text-stone-100">{formatPercent(stats.precision)}</div>
          </div>
          <div className="metric-clean py-4 before:bg-pink-500">
            <span className="text-stone-500 text-[10px] uppercase tracking-wider mb-1 block">Recall</span>
            <div className="text-2xl font-serif text-stone-100">{formatPercent(stats.recall)}</div>
          </div>
          <div className="metric-clean py-4 before:bg-green-500">
            <span className="text-stone-500 text-[10px] uppercase tracking-wider mb-1 block">mAP50</span>
            <div className="text-2xl font-serif text-stone-100">{formatPercent(stats.mAP50)}</div>
          </div>
          <div className="metric-clean py-4 before:bg-amber-500">
            <span className="text-stone-500 text-[10px] uppercase tracking-wider mb-1 block">mAP50-95</span>
            <div className="text-2xl font-serif text-stone-100">{formatPercent(stats.mAP50_95)}</div>
          </div>
        </div>

        {/* Charts Grid - Hidden for now */}

        {/* System Status (Very small as requested) */}
        <div className="flex items-center gap-6 pt-6 border-t border-stone-800">
          <div className="flex items-center gap-3 bg-stone-900/50 px-4 py-2 rounded-lg border border-stone-800">
             <Cpu size={14} className="text-stone-500" />
             <span className="text-xs text-stone-400 font-mono">GPU: {gpuStats?.temperature || '--'}°C</span>
             <div className="w-20 h-1.5 bg-stone-800 rounded-full overflow-hidden">
                <div className="h-full bg-green-500" style={{ width: `${gpuStats?.utilization || 0}%` }} />
             </div>
             <span className="text-[10px] text-stone-500 uppercase">{gpuStats?.utilization || 0}% UTIL</span>
          </div>

          <div className="flex items-center gap-3 bg-stone-900/50 px-4 py-2 rounded-lg border border-stone-800">
             <HardDrive size={14} className="text-stone-500" />
             <span className="text-xs text-stone-400 font-mono">SSD: HEALTHY</span>
          </div>

          <div className="flex items-center gap-3 bg-stone-900/50 px-4 py-2 rounded-lg border border-stone-800 ml-auto">
             <Activity size={14} className="text-stone-500" />
             <span className="text-xs text-green-500 uppercase font-medium">System Online</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
