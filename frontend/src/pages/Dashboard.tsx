import React, { useEffect, useState } from 'react';
import { 
  Activity, 
  Cpu, 
  HardDrive, 
  Play, 
  TrendingUp, 
  Clock, 
  Layers,
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

  const formatPercent = (val: number | undefined) => {
    if (val === undefined) return '--%';
    return `${(val * 100).toFixed(1)}%`;
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-12">
      {/* Welcome & Global Stats */}
      <div className="flex items-end justify-between">
        <div>
          <p style={{ color: 'var(--text-muted)' }} className="mb-1">Welcome back</p>
          <div className="flex items-center gap-3">
            <div 
              className="w-10 h-10 rounded-xl flex items-center justify-center p-2 relative overflow-hidden group"
              style={{ 
                background: 'var(--bg-secondary)', 
                border: '1px solid var(--border-primary)'
              }}
            >
              <div 
                className="absolute inset-0 opacity-50"
                style={{
                  background: 'linear-gradient(135deg, rgba(225,29,72,0.1), rgba(244,63,94,0.1))'
                }}
              />
              <div className="flex items-end gap-[1px] h-full w-full relative z-10">
                <div 
                  className="flex-1 rounded-t-[1px]"
                  style={{ 
                    height: '40%', 
                    background: 'var(--accent-primary)',
                    opacity: 0.6 
                  }}
                />
                <div 
                  className="flex-1 rounded-t-[1px]"
                  style={{ 
                    height: '80%', 
                    background: 'var(--accent-secondary)',
                    opacity: 0.6 
                  }}
                />
                <div 
                  className="flex-1 rounded-t-[1px]"
                  style={{ 
                    height: '55%', 
                    background: 'var(--accent-primary)',
                    opacity: 0.4 
                  }}
                />
              </div>
            </div>
            <h1 className="font-serif text-4xl" style={{ color: 'var(--text-primary)' }}>Dashboard</h1>
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

      {/* Top row metrics */}
      <div className="grid grid-cols-4 gap-4">
        <div className="metric-clean">
          <div className="flex items-center justify-between mb-4">
            <div 
              className="p-2 rounded-lg"
              style={{ background: 'var(--bg-tertiary)' }}
            >
              <Layers className="w-5 h-5" style={{ color: 'var(--text-muted)' }} />
            </div>
            <span className="badge-clean badge-neutral">Total</span>
          </div>
          <div className="text-3xl font-serif mb-1" style={{ color: 'var(--text-primary)' }}>{sessions.length}</div>
          <div style={{ color: 'var(--text-muted)' }} className="text-sm">Training Sessions</div>
        </div>

        <div className="metric-clean">
          <div className="flex items-center justify-between mb-4">
            <div 
              className="p-2 rounded-lg"
              style={{ background: 'var(--warning-bg)' }}
            >
              <Activity className="w-5 h-5" style={{ color: 'var(--warning-text)' }} />
            </div>
            <span className="badge-clean badge-warning">Active</span>
          </div>
          <div className="text-3xl font-serif mb-1" style={{ color: 'var(--text-primary)' }}>{runningCount}</div>
          <div style={{ color: 'var(--text-muted)' }} className="text-sm">Currently Running</div>
        </div>

        <div className="metric-clean">
          <div className="flex items-center justify-between mb-4">
            <div 
              className="p-2 rounded-lg"
              style={{ background: 'var(--success-bg)' }}
            >
              <TrendingUp className="w-5 h-5" style={{ color: 'var(--success-text)' }} />
            </div>
            <span className="badge-clean badge-success">Done</span>
          </div>
          <div className="text-3xl font-serif mb-1" style={{ color: 'var(--text-primary)' }}>{completedCount}</div>
          <div style={{ color: 'var(--text-muted)' }} className="text-sm">Completed</div>
        </div>

        <div className="metric-clean">
          <div className="flex items-center justify-between mb-4">
            <div 
              className="p-2 rounded-lg"
              style={{ background: 'rgba(225, 29, 72, 0.1)' }}
            >
              <Clock className="w-5 h-5" style={{ color: 'var(--accent-primary)' }} />
            </div>
            <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{activeSession ? 'Elapsed' : 'Avg'}</span>
          </div>
          <div className="text-3xl font-serif mb-1" style={{ color: 'var(--text-primary)' }}>{getTrainingTime()}</div>
          <div style={{ color: 'var(--text-muted)' }} className="text-sm">Training Time</div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 gap-6">
        {/* Run Selector & Info Accordion */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h2 className="text-lg font-serif" style={{ color: 'var(--text-secondary)' }}>Detailed Analytics</h2>
            <select 
              value={selectedSessionId || ''}
              onChange={(e) => setSelectedSessionId(e.target.value)}
              className="input-clean min-w-[200px]"
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
            className="btn-clean"
          >
            <Info size={14} />
            {showRunInfo ? 'Hide Run Configuration' : 'Show Run Configuration'}
          </button>
        </div>

        {showRunInfo && (
          <div 
            className="card-clean border-dashed"
            style={{ background: 'var(--bg-secondary)' }}
          >
            <div className="grid grid-cols-4 gap-6 text-sm">
              <div>
                <span style={{ color: 'var(--text-muted)' }} className="block mb-1 uppercase text-[10px] tracking-wider">Model Type</span>
                <span style={{ color: 'var(--text-secondary)' }} className="capitalize">{selectedSession?.model_type || '--'}</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }} className="block mb-1 uppercase text-[10px] tracking-wider">Total Epochs</span>
                <span style={{ color: 'var(--text-secondary)' }}>{selectedSession?.total_epochs || '--'}</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }} className="block mb-1 uppercase text-[10px] tracking-wider">Start Time</span>
                <span style={{ color: 'var(--text-secondary)' }}>
                  {selectedSession?.start_time ? new Date(selectedSession.start_time * 1000).toLocaleString() : '--'}
                </span>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }} className="block mb-1 uppercase text-[10px] tracking-wider">Dataset Path</span>
                <span style={{ color: 'var(--text-secondary)' }} className="truncate block max-w-[200px]" title={selectedSession?.config?.dataset_yaml || selectedSession?.config?.data || '--'}>{selectedSession?.config?.dataset_yaml || selectedSession?.config?.data || '--'}</span>
              </div>
              {(selectedSession?.config?.imgsz || selectedSession?.config?.batch) && (
                <>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }} className="block mb-1 uppercase text-[10px] tracking-wider">Image Size</span>
                    <span style={{ color: 'var(--text-secondary)' }}>{selectedSession?.config?.imgsz || '--'}</span>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }} className="block mb-1 uppercase text-[10px] tracking-wider">Batch Size</span>
                    <span style={{ color: 'var(--text-secondary)' }}>{selectedSession?.config?.batch || '--'}</span>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }} className="block mb-1 uppercase text-[10px] tracking-wider">Learning Rate / Opt</span>
                    <span style={{ color: 'var(--text-secondary)' }}>{selectedSession?.config?.lr0 || '--'} / {selectedSession?.config?.optimizer || '--'}</span>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* Segmentation Metrics Cards */}
        <div className="grid grid-cols-5 gap-4">
          <div className="metric-clean py-4" style={{ '--accent-top': '#8b5cf6' } as React.CSSProperties}>
            <span style={{ color: 'var(--text-muted)' }} className="text-[10px] uppercase tracking-wider mb-1 block">F1 Score</span>
            <div className="text-2xl font-serif" style={{ color: 'var(--text-primary)' }}>{formatPercent(stats.f1)}</div>
          </div>
          <div className="metric-clean py-4" style={{ '--accent-top': '#3b82f6' } as React.CSSProperties}>
            <span style={{ color: 'var(--text-muted)' }} className="text-[10px] uppercase tracking-wider mb-1 block">Precision</span>
            <div className="text-2xl font-serif" style={{ color: 'var(--text-primary)' }}>{formatPercent(stats.precision)}</div>
          </div>
          <div className="metric-clean py-4" style={{ '--accent-top': '#ec4899' } as React.CSSProperties}>
            <span style={{ color: 'var(--text-muted)' }} className="text-[10px] uppercase tracking-wider mb-1 block">Recall</span>
            <div className="text-2xl font-serif" style={{ color: 'var(--text-primary)' }}>{formatPercent(stats.recall)}</div>
          </div>
          <div className="metric-clean py-4">
            <span style={{ color: 'var(--text-muted)' }} className="text-[10px] uppercase tracking-wider mb-1 block">mAP50</span>
            <div className="text-2xl font-serif" style={{ color: 'var(--text-primary)' }}>{formatPercent(stats.mAP50)}</div>
          </div>
          <div className="metric-clean py-4" style={{ '--accent-top': '#f59e0b' } as React.CSSProperties}>
            <span style={{ color: 'var(--text-muted)' }} className="text-[10px] uppercase tracking-wider mb-1 block">mAP50-95</span>
            <div className="text-2xl font-serif" style={{ color: 'var(--text-primary)' }}>{formatPercent(stats.mAP50_95)}</div>
          </div>
        </div>

        {/* System Status - Fixed Contrast */}
        <div 
          className="flex items-center gap-6 pt-6"
          style={{ borderTop: '1px solid var(--border-primary)' }}
        >
          {/* GPU Status */}
          <div 
            className="flex items-center gap-3 px-4 py-2 rounded-lg"
            style={{ 
              background: 'var(--bg-secondary)', 
              border: '1px solid var(--border-primary)'
            }}
          >
            <Cpu size={14} style={{ color: 'var(--text-muted)' }} />
            <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>GPU: {gpuStats?.temperature || '--'}°C</span>
            <div 
              className="w-20 h-1.5 rounded-full overflow-hidden"
              style={{ background: 'var(--bg-tertiary)' }}
            >
              <div 
                className="h-full rounded-full"
                style={{ 
                  width: `${gpuStats?.utilization || 0}%`,
                  background: 'var(--accent-primary)'
                }} 
              />
            </div>
            <span className="text-[10px] uppercase" style={{ color: 'var(--text-muted)' }}>{gpuStats?.utilization || 0}% UTIL</span>
          </div>

          {/* SSD Status */}
          <div 
            className="flex items-center gap-3 px-4 py-2 rounded-lg"
            style={{ 
              background: 'var(--bg-secondary)', 
              border: '1px solid var(--border-primary)'
            }}
          >
            <HardDrive size={14} style={{ color: 'var(--text-muted)' }} />
            <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>SSD: HEALTHY</span>
          </div>

          {/* System Online */}
          <div 
            className="flex items-center gap-3 px-4 py-2 rounded-lg ml-auto"
            style={{ 
              background: 'var(--success-bg)', 
              border: '1px solid var(--border-primary)'
            }}
          >
            <Activity size={14} style={{ color: 'var(--success-text)' }} />
            <span className="text-xs uppercase font-medium" style={{ color: 'var(--success-text)' }}>System Online</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
