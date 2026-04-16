import React, { useEffect, useState } from 'react';
import { getTrainingSessions } from '../api';
import { Activity, ChevronRight, BarChart3, X, Check } from 'lucide-react';

const Evaluation: React.FC = () => {
  const [sessions, setSessions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'past' | 'compare'>('past');
  const [selectedForCompare, setSelectedForCompare] = useState<Set<string>>(new Set());

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      const res = await getTrainingSessions();
      setSessions(res.data);
    } catch (error) {
      console.error('Failed to load training sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleSelection = (id: string) => {
    setSelectedForCompare(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else if (newSet.size < 4) {
        newSet.add(id);
      }
      return newSet;
    });
  };

  const clearSelection = () => {
    setSelectedForCompare(new Set());
  };

  const selectedSessions = sessions.filter(s => selectedForCompare.has(s.id));

  // Calculate max values for bar chart normalization
  const getMaxValue = (key: string) => {
    if (selectedSessions.length === 0) return 1;
    const values = selectedSessions.map(s => s.latest_metrics?.[key] || 0);
    return Math.max(...values, 0.0001);
  };

  const maxF1 = getMaxValue('f1');
  const maxPrecision = getMaxValue('precision');
  const maxRecall = getMaxValue('recall');
  const maxmAP50 = getMaxValue('mAP50');
  const maxmAP50_95 = getMaxValue('mAP50_95');

  const metricColors = [
    'var(--accent-primary)',
    'var(--success-text)', 
    'var(--warning-text)',
    '#8b5cf6' // purple for 4th
  ];

  return (
    <div className="space-y-6 h-full overflow-auto pb-12">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 relative">
            <div 
              className="absolute inset-0 rounded-lg rotate-[-6deg] opacity-80"
              style={{ background: 'var(--accent-secondary)' }}
            />
            <div 
              className="absolute inset-0 rounded-lg rotate-[-3deg] opacity-90"
              style={{ background: 'var(--accent-primary)' }}
            />
            <div 
              className="absolute inset-0 rounded-lg shadow-lg flex items-center justify-center"
              style={{ background: 'var(--accent-primary)' }}
            >
              <Activity size={20} className="text-white" />
            </div>
          </div>
          <h1 className="font-serif text-3xl" style={{ color: 'var(--text-primary)' }}>Training History</h1>
        </div>
      </div>

      {/* Tabs */}
      <div 
        className="flex"
        style={{ borderBottom: '1px solid var(--border-primary)' }}
      >
        <button 
          onClick={() => setActiveTab('past')}
          className="px-6 py-3 text-sm font-medium transition-colors relative"
          style={
            activeTab === 'past'
              ? { color: 'var(--accent-primary)' }
              : { color: 'var(--text-muted)' }
          }
          onMouseEnter={(e) => {
            if (activeTab !== 'past') {
              e.currentTarget.style.color = 'var(--text-secondary)';
            }
          }}
          onMouseLeave={(e) => {
            if (activeTab !== 'past') {
              e.currentTarget.style.color = 'var(--text-muted)';
            }
          }}
        >
          Past Runs
          {activeTab === 'past' && (
            <div 
              className="absolute bottom-0 left-0 right-0 h-0.5"
              style={{ background: 'var(--accent-primary)' }}
            />
          )}
        </button>
        <button 
          onClick={() => {
            if (selectedForCompare.size > 0) {
              setActiveTab('compare');
            }
          }}
          className="px-6 py-3 text-sm font-medium transition-colors relative"
          style={
            activeTab === 'compare'
              ? { color: 'var(--accent-primary)' }
              : { 
                  color: selectedForCompare.size > 0 ? 'var(--text-secondary)' : 'var(--text-muted)',
                  opacity: selectedForCompare.size > 0 ? 1 : 0.5
                }
          }
          onMouseEnter={(e) => {
            if (activeTab !== 'compare' && selectedForCompare.size > 0) {
              e.currentTarget.style.color = 'var(--text-secondary)';
            }
          }}
          onMouseLeave={(e) => {
            if (activeTab !== 'compare') {
              e.currentTarget.style.color = selectedForCompare.size > 0 ? 'var(--text-secondary)' : 'var(--text-muted)';
            }
          }}
        >
          Compare Models
          {selectedForCompare.size > 0 && (
            <span 
              className="ml-2 px-2 py-0.5 text-xs rounded-full"
              style={{ background: 'var(--accent-primary)', color: 'white' }}
            >
              {selectedForCompare.size}
            </span>
          )}
          {activeTab === 'compare' && (
            <div 
              className="absolute bottom-0 left-0 right-0 h-0.5"
              style={{ background: 'var(--accent-primary)' }}
            />
          )}
        </button>
      </div>

      {/* Selection hint bar */}
      {activeTab === 'past' && (
        <div 
          className="flex items-center justify-between p-4 rounded-lg"
          style={{ background: 'var(--bg-secondary)' }}
        >
          <div className="flex items-center gap-3">
            <BarChart3 size={20} style={{ color: 'var(--accent-primary)' }} />
            <span style={{ color: 'var(--text-secondary)' }}>
              Select up to 4 runs to compare
            </span>
            {selectedForCompare.size > 0 && (
              <span style={{ color: 'var(--text-muted)' }}>
                ({selectedForCompare.size} selected)
              </span>
            )}
          </div>
          {selectedForCompare.size > 0 && (
            <div className="flex items-center gap-3">
              <button
                onClick={() => setActiveTab('compare')}
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--accent-primary)', color: 'white' }}
              >
                View Comparison
              </button>
              <button
                onClick={clearSelection}
                className="p-2 rounded-lg"
                style={{ color: 'var(--text-muted)' }}
                title="Clear selection"
              >
                <X size={18} />
              </button>
            </div>
          )}
        </div>
      )}

      {activeTab === 'past' ? (
        <div className="card-clean overflow-hidden p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr 
                  className="text-[11px] uppercase tracking-wider"
                  style={{ 
                    background: 'var(--bg-secondary)', 
                    color: 'var(--text-muted)',
                    borderBottom: '1px solid var(--border-primary)'
                  }}
                >
                  <th className="py-4 px-4 font-medium">Select</th>
                  <th className="py-4 px-4 font-medium">run_id</th>
                  <th className="py-4 px-4 font-medium">timestamp</th>
                  <th className="py-4 px-4 font-medium">data_yml</th>
                  <th className="py-4 px-4 font-medium">epochs</th>
                  <th className="py-4 px-4 font-medium">learning_rate</th>
                  <th className="py-4 px-4 font-medium">batch_size</th>
                  <th className="py-4 px-4 font-medium">optimizer</th>
                  <th className="py-4 px-4 font-medium">image_size</th>
                  <th className="py-4 px-4 font-medium">patience</th>
                  <th className="py-4 px-4 font-medium">weight_decay</th>
                  <th className="py-4 px-4 font-medium">momentum</th>
                  <th className="py-4 px-4 font-medium">final_loss</th>
                  <th className="py-4 px-4 font-medium">F1 Score</th>
                  <th className="py-4 px-4 font-medium">Precision</th>
                  <th className="py-4 px-4 font-medium">Recall</th>
                  <th className="py-4 px-4 font-medium">mAP50</th>
                  <th className="py-4 px-4 font-medium">mAP50-95</th>
                </tr>
              </thead>
              <tbody 
                className="text-sm"
                style={{ 
                  color: 'var(--text-secondary)',
                  borderTop: '1px solid var(--border-primary)'
                }}
              >
                {sessions.length === 0 ? (
                  <tr>
                    <td 
                      colSpan={18} 
                      className="py-20 text-center"
                      style={{ color: 'var(--text-disabled)' }}
                    >
                      {loading ? 'Fetching history...' : 'No training sessions found'}
                    </td>
                  </tr>
                ) : (
                  sessions.map((session) => (
                    <tr 
                      key={session.id} 
                      className="transition-colors cursor-pointer"
                      style={{ borderBottom: '1px solid var(--border-primary)' }}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLTableRowElement).style.background = 'var(--bg-hover)';
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLTableRowElement).style.background = selectedForCompare.has(session.id) ? 'var(--accent-glow)' : 'transparent';
                      }}
                      onClick={() => toggleSelection(session.id)}
                    >
                      <td className="py-4 px-4">
                        <div 
                          className="w-5 h-5 rounded border flex items-center justify-center transition-all"
                          style={{
                            borderColor: selectedForCompare.has(session.id) ? 'var(--accent-primary)' : 'var(--border-secondary)',
                            background: selectedForCompare.has(session.id) ? 'var(--accent-primary)' : 'transparent'
                          }}
                        >
                          {selectedForCompare.has(session.id) && <Check size={14} className="text-white" />}
                        </div>
                      </td>
                      <td className="py-4 px-4 font-mono" style={{ color: 'var(--text-secondary)' }}>{session.id}</td>
                      <td className="py-4 px-4 whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
                        {session.start_time ? new Date(session.start_time * 1000).toLocaleString() : '--'}
                      </td>
                      <td className="py-4 px-4" style={{ color: 'var(--text-muted)' }}>
                        {session.config?.dataset_yaml?.split(/[\\/]/).pop() || 'data.yaml'}
                      </td>
                      <td className="py-4 px-4 text-center" style={{ color: 'var(--text-secondary)' }}>{session.total_epochs || session.config?.epochs || '--'}</td>
                      <td className="py-4 px-4 text-center" style={{ color: 'var(--text-secondary)' }}>{session.config?.lr0 || '--'}</td>
                      <td className="py-4 px-4 text-center" style={{ color: 'var(--text-secondary)' }}>{session.config?.batch || '--'}</td>
                      <td className="py-4 px-4 text-center" style={{ color: 'var(--text-secondary)' }}>{session.config?.optimizer || 'Adam'}</td>
                      <td className="py-4 px-4 text-center" style={{ color: 'var(--text-secondary)' }}>{session.config?.imgsz || '--'}</td>
                      <td className="py-4 px-4 text-center" style={{ color: 'var(--text-secondary)' }}>{session.config?.patience || '100'}</td>
                      <td className="py-4 px-4 text-center" style={{ color: 'var(--text-secondary)' }}>{session.config?.weight_decay || '0.0005'}</td>
                      <td className="py-4 px-4 text-center" style={{ color: 'var(--text-secondary)' }}>{session.config?.momentum || '0.937'}</td>
                      <td className="py-4 px-4 text-center font-mono" style={{ color: 'var(--text-secondary)' }}>
                        {session.latest_metrics?.box_loss?.toFixed(4) || '--'}
                      </td>
                      <td className="py-4 px-4 text-center font-mono" style={{ color: 'var(--text-secondary)' }}>
                        {session.latest_metrics?.f1 !== undefined ? `${(session.latest_metrics.f1 * 100).toFixed(1)}%` : '--'}
                      </td>
                      <td className="py-4 px-4 text-center font-mono" style={{ color: 'var(--text-secondary)' }}>
                        {session.latest_metrics?.precision !== undefined ? `${(session.latest_metrics.precision * 100).toFixed(1)}%` : '--'}
                      </td>
                      <td className="py-4 px-4 text-center font-mono" style={{ color: 'var(--text-secondary)' }}>
                        {session.latest_metrics?.recall !== undefined ? `${(session.latest_metrics.recall * 100).toFixed(1)}%` : '--'}
                      </td>
                      <td className="py-4 px-4 text-center font-mono" style={{ color: 'var(--text-secondary)' }}>
                        {session.latest_metrics?.mAP50 !== undefined ? `${(session.latest_metrics.mAP50 * 100).toFixed(1)}%` : '--'}
                      </td>
                      <td className="py-4 px-4 text-center font-mono" style={{ color: 'var(--text-secondary)' }}>
                        {session.latest_metrics?.mAP50_95 !== undefined ? `${(session.latest_metrics.mAP50_95 * 100).toFixed(1)}%` : '--'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      ) : selectedSessions.length === 0 ? (
        <div 
          className="flex flex-col items-center justify-center py-20 rounded-xl border border-dashed"
          style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-primary)' }}
        >
          <ChevronRight size={48} className="mb-4" style={{ color: 'var(--border-secondary)' }} />
          <h3 className="font-medium" style={{ color: 'var(--text-muted)' }}>No Models Selected</h3>
          <p className="text-sm mt-1" style={{ color: 'var(--text-disabled)' }}>Go to 'Past Runs' and select runs to compare.</p>
          <button
            onClick={() => setActiveTab('past')}
            className="mt-4 px-4 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
          >
            Go to Past Runs
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Comparison Controls */}
          <div 
            className="flex items-center justify-between p-4 rounded-lg"
            style={{ background: 'var(--bg-secondary)' }}
          >
            <div className="flex items-center gap-3">
              <BarChart3 size={20} style={{ color: 'var(--accent-primary)' }} />
              <span style={{ color: 'var(--text-secondary)' }}>
                Comparing {selectedSessions.length} model{selectedSessions.length > 1 ? 's' : ''}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setActiveTab('past')}
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
              >
                Change Selection
              </button>
              <button
                onClick={clearSelection}
                className="p-2 rounded-lg"
                style={{ color: 'var(--text-muted)' }}
                title="Clear all"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {/* Visual Bar Chart Comparison */}
          <div className="card-clean">
            <h3 className="text-lg font-serif mb-6" style={{ color: 'var(--text-secondary)' }}>Performance Metrics Comparison</h3>
            
            {['F1 Score', 'Precision', 'Recall', 'mAP50', 'mAP50-95'].map((metricName) => {
              const key = metricName.toLowerCase().replace('-', '_');
              const maxVal = metricName === 'F1 Score' ? maxF1 : 
                           metricName === 'Precision' ? maxPrecision :
                           metricName === 'Recall' ? maxRecall :
                           metricName === 'mAP50' ? maxmAP50 : maxmAP50_95;
              
              return (
                <div key={metricName} className="mb-6">
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>{metricName}</span>
                  </div>
                  <div className="space-y-2">
                    {selectedSessions.map((session, idx) => {
                      const value = session.latest_metrics?.[key] || 0;
                      const percentage = (value / maxVal) * 100;
                      
                      return (
                        <div key={session.id} className="flex items-center gap-3">
                          <div className="w-24 text-xs truncate" style={{ color: 'var(--text-muted)' }}>
                            Run {session.id}
                          </div>
                          <div className="flex-1 h-8 rounded-lg overflow-hidden" style={{ background: 'var(--bg-tertiary)' }}>
                            <div
                              className="h-full flex items-center px-3 text-xs font-medium transition-all duration-500"
                              style={{ 
                                width: `${percentage}%`,
                                background: metricColors[idx % metricColors.length],
                                color: 'white',
                                minWidth: percentage > 10 ? 'auto' : '30px'
                              }}
                            >
                              {value !== undefined ? `${(value * 100).toFixed(1)}%` : '--'}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Side-by-Side Comparison Table */}
          <div className="card-clean overflow-hidden p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr 
                    className="text-[11px] uppercase tracking-wider"
                    style={{ 
                      background: 'var(--bg-secondary)', 
                      color: 'var(--text-muted)',
                      borderBottom: '1px solid var(--border-primary)'
                    }}
                  >
                    <th className="py-4 px-4 font-medium">Metric</th>
                    {selectedSessions.map((session, idx) => (
                      <th 
                        key={session.id} 
                        className="py-4 px-4 font-medium text-center"
                        style={{ color: metricColors[idx % metricColors.length] }}
                      >
                        Run {session.id}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody 
                  className="text-sm"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  {[
                    { label: 'Model Type', key: 'model_type', getValue: (s: any) => s.model_type || '--' },
                    { label: 'Dataset', key: 'dataset', getValue: (s: any) => s.config?.dataset_yaml?.split(/[\\/]/).pop() || 'data.yaml' },
                    { label: 'Epochs', key: 'epochs', getValue: (s: any) => s.total_epochs || s.config?.epochs || '--' },
                    { label: 'Image Size', key: 'imgsz', getValue: (s: any) => s.config?.imgsz || '--' },
                    { label: 'Batch Size', key: 'batch', getValue: (s: any) => s.config?.batch || '--' },
                    { label: 'Learning Rate', key: 'lr0', getValue: (s: any) => s.config?.lr0 || '--' },
                    { label: 'Optimizer', key: 'optimizer', getValue: (s: any) => s.config?.optimizer || 'Adam' },
                    { label: 'F1 Score', key: 'f1', getValue: (s: any) => s.latest_metrics?.f1 !== undefined ? `${(s.latest_metrics.f1 * 100).toFixed(1)}%` : '--' },
                    { label: 'Precision', key: 'precision', getValue: (s: any) => s.latest_metrics?.precision !== undefined ? `${(s.latest_metrics.precision * 100).toFixed(1)}%` : '--' },
                    { label: 'Recall', key: 'recall', getValue: (s: any) => s.latest_metrics?.recall !== undefined ? `${(s.latest_metrics.recall * 100).toFixed(1)}%` : '--' },
                    { label: 'mAP50', key: 'mAP50', getValue: (s: any) => s.latest_metrics?.mAP50 !== undefined ? `${(s.latest_metrics.mAP50 * 100).toFixed(1)}%` : '--' },
                    { label: 'mAP50-95', key: 'mAP50_95', getValue: (s: any) => s.latest_metrics?.mAP50_95 !== undefined ? `${(s.latest_metrics.mAP50_95 * 100).toFixed(1)}%` : '--' },
                    { label: 'Final Loss', key: 'box_loss', getValue: (s: any) => s.latest_metrics?.box_loss?.toFixed(4) || '--' },
                  ].map((row, rowIdx) => (
                    <tr 
                      key={row.key}
                      style={{ 
                        borderBottom: rowIdx < 12 ? '1px solid var(--border-primary)' : 'none',
                        background: rowIdx % 2 === 0 ? 'transparent' : 'var(--bg-secondary)'
                      }}
                    >
                      <td className="py-3 px-4 font-medium" style={{ color: 'var(--text-muted)' }}>{row.label}</td>
                      {selectedSessions.map((session) => (
                        <td key={session.id} className="py-3 px-4 text-center font-mono">
                          {row.getValue(session)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Model Details Cards */}
          <div className="grid grid-cols-2 gap-4">
            {selectedSessions.map((session, idx) => (
              <div 
                key={session.id}
                className="card-clean"
                style={{ borderLeft: `3px solid ${metricColors[idx % metricColors.length]}` }}
              >
                <div className="flex items-center justify-between mb-4">
                  <h4 className="font-medium" style={{ color: 'var(--text-secondary)' }}>
                    Run {session.id}
                  </h4>
                  <span 
                    className="text-xs px-2 py-1 rounded"
                    style={{ 
                      background: 'var(--bg-tertiary)', 
                      color: metricColors[idx % metricColors.length]
                    }}
                  >
                    {session.model_type}
                  </span>
                </div>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-muted)' }}>Created:</span>
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {session.start_time ? new Date(session.start_time * 1000).toLocaleDateString() : '--'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-muted)' }}>Epochs:</span>
                    <span style={{ color: 'var(--text-secondary)' }}>{session.total_epochs || session.config?.epochs || '--'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-muted)' }}>Dataset:</span>
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {session.config?.dataset_yaml?.split(/[\\/]/).pop() || 'data.yaml'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Evaluation;
