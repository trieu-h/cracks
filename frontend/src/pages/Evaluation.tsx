import React, { useEffect, useState } from 'react';
import { getTrainingSessions } from '../api';
import { Activity, ChevronRight } from 'lucide-react';

const Evaluation: React.FC = () => {
  const [sessions, setSessions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'past' | 'compare'>('past');

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
          onClick={() => setActiveTab('compare')}
          className="px-6 py-3 text-sm font-medium transition-colors relative"
          style={
            activeTab === 'compare'
              ? { color: 'var(--accent-primary)' }
              : { color: 'var(--text-muted)' }
          }
          onMouseEnter={(e) => {
            if (activeTab !== 'compare') {
              e.currentTarget.style.color = 'var(--text-secondary)';
            }
          }}
          onMouseLeave={(e) => {
            if (activeTab !== 'compare') {
              e.currentTarget.style.color = 'var(--text-muted)';
            }
          }}
        >
          Compare Models
          {activeTab === 'compare' && (
            <div 
              className="absolute bottom-0 left-0 right-0 h-0.5"
              style={{ background: 'var(--accent-primary)' }}
            />
          )}
        </button>
      </div>

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
                      colSpan={17} 
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
                      className="transition-colors"
                      style={{ borderBottom: '1px solid var(--border-primary)' }}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLTableRowElement).style.background = 'var(--bg-hover)';
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLTableRowElement).style.background = 'transparent';
                      }}
                    >
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
      ) : (
        <div 
          className="flex flex-col items-center justify-center py-20 rounded-xl border border-dashed"
          style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border-primary)' }}
        >
          <ChevronRight size={48} className="mb-4" style={{ color: 'var(--border-secondary)' }} />
          <h3 className="font-medium" style={{ color: 'var(--text-muted)' }}>Model Comparison</h3>
          <p className="text-sm mt-1" style={{ color: 'var(--text-disabled)' }}>Select multiple runs from 'Past Runs' to compare side-by-side.</p>
        </div>
      )}
    </div>
  );
};

export default Evaluation;
