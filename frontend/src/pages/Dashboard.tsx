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
  Box,
  Sparkles,
  ChevronRight
} from 'lucide-react';
import { useWebSocket } from '../hooks/useWebSocket';
import { getGPUStats, getTrainingSessions } from '../api';

const Dashboard: React.FC = () => {
  const [gpuStats, setGpuStats] = useState<any>(null);
  const [sessions, setSessions] = useState<any[]>([]);

  useWebSocket('/ws/system', (data) => {
    if (data.type === 'gpu_stats') {
      setGpuStats(data.data);
    }
  });

  useEffect(() => {
    getGPUStats().then(res => setGpuStats(res.data));
    getTrainingSessions().then(res => setSessions(res.data));
  }, []);

  const activeSession = sessions.find(s => s.status === 'running');
  const runningCount = sessions.filter(s => s.status === 'running').length;
  const completedCount = sessions.filter(s => s.status === 'completed').length;

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Welcome */}
      <div className="flex items-end justify-between">
        <div>
          <p className="text-stone-500 mb-1">Welcome back</p>
          <h1 className="font-serif text-4xl text-stone-100">
            Dashboard
          </h1>
        </div>
        <button 
          onClick={() => window.location.href = '/training'}
          className="btn-clean btn-primary"
        >
          <Play size={16} />
          New Training
        </button>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <div className="metric-clean">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 rounded-lg bg-stone-800">
              <Layers className="w-5 h-5 text-stone-400" />
            </div>
            <span className="badge-clean badge-neutral">Total</span>
          </div>
          <div className="text-3xl font-serif text-stone-100 mb-1">
            {sessions.length}
          </div>
          <div className="text-sm text-stone-500">Training Sessions</div>
        </div>

        <div className="metric-clean">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 rounded-lg bg-amber-900/20">
              <Activity className="w-5 h-5 text-amber-500" />
            </div>
            <span className="badge-clean badge-warning">Active</span>
          </div>
          <div className="text-3xl font-serif text-stone-100 mb-1">
            {runningCount}
          </div>
          <div className="text-sm text-stone-500">Currently Running</div>
        </div>

        <div className="metric-clean">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 rounded-lg bg-green-900/20">
              <TrendingUp className="w-5 h-5 text-green-500" />
            </div>
            <span className="badge-clean badge-success">Done</span>
          </div>
          <div className="text-3xl font-serif text-stone-100 mb-1">
            {completedCount}
          </div>
          <div className="text-sm text-stone-500">Completed</div>
        </div>

        <div className="metric-clean">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2 rounded-lg bg-rose-900/20">
              <Clock className="w-5 h-5 text-rose-500" />
            </div>
            <span className="text-xs text-stone-500 font-mono">Avg</span>
          </div>
          <div className="text-3xl font-serif text-stone-100 mb-1">
            2.4h
          </div>
          <div className="text-sm text-stone-500">Training Time</div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-3 gap-6">
        {/* GPU Status */}
        <div className="card-clean">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-stone-800">
              <Cpu className="w-5 h-5 text-stone-400" />
            </div>
            <div>
              <h3 className="font-medium text-stone-100">GPU Status</h3>
              <p className="text-xs text-stone-500">NVIDIA RTX 4090</p>
            </div>
          </div>

          <div className="space-y-5">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-stone-500">Temperature</span>
                <span className="text-lg font-mono text-stone-100">
                  {gpuStats?.temperature || '--'}°C
                </span>
              </div>
              <div className="progress-clean">
                <div 
                  className="progress-clean-bar"
                  style={{ width: `${Math.min((gpuStats?.temperature || 0) / 90 * 100, 100)}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-stone-500">Utilization</span>
                <span className="text-lg font-mono text-green-400">
                  {gpuStats?.utilization || '--'}%
                </span>
              </div>
              <div className="progress-clean">
                <div 
                  className="progress-clean-bar"
                  style={{ width: `${gpuStats?.utilization || 0}%` }}
                />
              </div>
            </div>

            <div className="pt-3 border-t border-stone-800">
              <div className="flex items-center justify-between">
                <span className="text-sm text-stone-500">Memory</span>
                <span className="text-sm font-mono text-stone-300">
                  {gpuStats?.memory_used || '--'} / {gpuStats?.memory_total || '--'} MB
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* System Info */}
        <div className="card-clean">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-stone-800">
              <HardDrive className="w-5 h-5 text-stone-400" />
            </div>
            <div>
              <h3 className="font-medium text-stone-100">System</h3>
              <p className="text-xs text-stone-500">Infrastructure Status</p>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 rounded-lg bg-stone-800/50">
              <div className="flex items-center gap-3">
                <Activity size={16} className="text-stone-500" />
                <span className="text-sm text-stone-300">Backend API</span>
              </div>
              <span className="badge-clean badge-success">Online</span>
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg bg-stone-800/50">
              <div className="flex items-center gap-3">
                <HardDrive size={16} className="text-stone-500" />
                <span className="text-sm text-stone-300">Storage</span>
              </div>
              <span className="badge-clean badge-success">Healthy</span>
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg bg-stone-800/50">
              <div className="flex items-center gap-3">
                <Cpu size={16} className="text-stone-500" />
                <span className="text-sm text-stone-300">GPU Compute</span>
              </div>
              <span className={`badge-clean ${gpuStats?.available ? 'badge-success' : 'badge-error'}`}>
                {gpuStats?.available ? 'Ready' : 'Offline'}
              </span>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="card-clean">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-stone-800">
              <Sparkles className="w-5 h-5 text-stone-400" />
            </div>
            <div>
              <h3 className="font-medium text-stone-100">Quick Actions</h3>
              <p className="text-xs text-stone-500">Common workflows</p>
            </div>
          </div>

          <div className="space-y-2">
            <button 
              onClick={() => window.location.href = '/training'}
              className="w-full flex items-center justify-between p-3 rounded-lg hover:bg-stone-800/50 transition-colors group"
            >
              <div className="flex items-center gap-3">
                <Play size={18} className="text-stone-500 group-hover:text-green-400" />
                <span className="text-sm text-stone-300">Start Training</span>
              </div>
              <ChevronRight size={16} className="text-stone-600 group-hover:text-stone-400" />
            </button>

            <button 
              onClick={() => window.location.href = '/datasets'}
              className="w-full flex items-center justify-between p-3 rounded-lg hover:bg-stone-800/50 transition-colors group"
            >
              <div className="flex items-center gap-3">
                <Database size={18} className="text-stone-500 group-hover:text-green-400" />
                <span className="text-sm text-stone-300">Upload Dataset</span>
              </div>
              <ChevronRight size={16} className="text-stone-600 group-hover:text-stone-400" />
            </button>

            <button 
              onClick={() => window.location.href = '/models'}
              className="w-full flex items-center justify-between p-3 rounded-lg hover:bg-stone-800/50 transition-colors group"
            >
              <div className="flex items-center gap-3">
                <Box size={18} className="text-stone-500 group-hover:text-green-400" />
                <span className="text-sm text-stone-300">Deploy Model</span>
              </div>
              <ChevronRight size={16} className="text-stone-600 group-hover:text-stone-400" />
            </button>
          </div>
        </div>
      </div>

      {/* Active Training */}
      {activeSession && (
        <div className="card-clean border-l-4 border-l-green-500">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-green-900/20 flex items-center justify-center">
                <Activity className="w-6 h-6 text-green-500" />
              </div>
              <div>
                <div className="flex items-center gap-3">
                  <h3 className="font-medium text-stone-100">
                    Active Training Session
                  </h3>
                  <span className="badge-clean badge-warning">In Progress</span>
                </div>
                <div className="flex items-center gap-3 mt-1 text-sm text-stone-500">
                  <span className="font-mono text-green-400">#{activeSession.id}</span>
                  <span>•</span>
                  <span className="uppercase">{activeSession.model_type}</span>
                  <span>•</span>
                  <span>Epoch {activeSession.current_epoch}/{activeSession.total_epochs}</span>
                </div>
              </div>
            </div>
            
            <div className="w-64">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-stone-500">Progress</span>
                <span className="text-sm font-mono text-green-400">
                  {Math.round((activeSession.current_epoch / activeSession.total_epochs) * 100)}%
                </span>
              </div>
              <div className="progress-clean">
                <div 
                  className="progress-clean-bar"
                  style={{ 
                    width: `${(activeSession.current_epoch / activeSession.total_epochs) * 100}%` 
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
