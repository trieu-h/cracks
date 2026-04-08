import React, { useEffect, useState } from 'react';
import { Panel } from '../components/ui/Panel';
import { getTrainingSessions, BASE_URL } from '../api';
import { Clock, Database, Settings, Activity, ChevronRight, Search, Filter } from 'lucide-react';

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
            <div className="absolute inset-0 bg-green-500 rounded-lg rotate-[-6deg] opacity-80" />
            <div className="absolute inset-0 bg-pink-500 rounded-lg rotate-[-3deg] opacity-90" />
            <div className="absolute inset-0 bg-blue-500 rounded-lg shadow-lg flex items-center justify-center">
               <Activity size={20} className="text-white" />
            </div>
          </div>
          <h1 className="font-serif text-3xl text-stone-100">Training History</h1>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-stone-800">
        <button 
          onClick={() => setActiveTab('past')}
          className={`px-6 py-3 text-sm font-medium transition-colors relative ${
            activeTab === 'past' ? 'text-red-500' : 'text-stone-500 hover:text-stone-300'
          }`}
        >
          Past Runs
          {activeTab === 'past' && (
            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-red-500" />
          )}
        </button>
        <button 
          onClick={() => setActiveTab('compare')}
          className={`px-6 py-3 text-sm font-medium transition-colors relative ${
            activeTab === 'compare' ? 'text-red-500' : 'text-stone-500 hover:text-stone-300'
          }`}
        >
          Compare Models
          {activeTab === 'compare' && (
            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-red-500" />
          )}
        </button>
      </div>

      {activeTab === 'past' ? (
        <div className="card-clean overflow-hidden p-0 border-stone-800/50">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-stone-900/30 text-stone-500 text-[11px] uppercase tracking-wider border-b border-stone-800">
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
              <tbody className="text-sm divide-y divide-stone-800/50">
                {sessions.length === 0 ? (
                  <tr>
                      <td colSpan={17} className="py-20 text-center text-stone-600">
                      {loading ? 'Fetching history...' : 'No training sessions found'}
                    </td>
                  </tr>
                ) : (
                  sessions.map((session) => (
                    <tr key={session.id} className="hover:bg-stone-800/20 transition-colors">
                      <td className="py-4 px-4 font-mono text-stone-300">{session.id}</td>
                      <td className="py-4 px-4 text-stone-400 whitespace-nowrap">
                        {session.start_time ? new Date(session.start_time * 1000).toLocaleString() : '--'}
                      </td>
                      <td className="py-4 px-4 text-stone-400">
                        {session.config?.dataset_yaml?.split(/[\\/]/).pop() || 'data.yaml'}
                      </td>
                      <td className="py-4 px-4 text-stone-300 text-center">{session.total_epochs || session.config?.epochs || '--'}</td>
                      <td className="py-4 px-4 text-stone-300 text-center">{session.config?.lr0 || '--'}</td>
                      <td className="py-4 px-4 text-stone-300 text-center">{session.config?.batch || '--'}</td>
                      <td className="py-4 px-4 text-stone-300 text-center">{session.config?.optimizer || 'Adam'}</td>
                      <td className="py-4 px-4 text-stone-300 text-center">{session.config?.imgsz || '--'}</td>
                      <td className="py-4 px-4 text-stone-300 text-center">{session.config?.patience || '100'}</td>
                      <td className="py-4 px-4 text-stone-300 text-center">{session.config?.weight_decay || '0.0005'}</td>
                      <td className="py-4 px-4 text-stone-300 text-center">{session.config?.momentum || '0.937'}</td>
                      <td className="py-4 px-4 text-stone-300 text-center font-mono">
                        {session.latest_metrics?.box_loss?.toFixed(4) || '--'}
                      </td>
                      <td className="py-4 px-4 text-stone-300 text-center font-mono">
                        {session.latest_metrics?.f1 !== undefined ? `${(session.latest_metrics.f1 * 100).toFixed(1)}%` : '--'}
                      </td>
                      <td className="py-4 px-4 text-stone-300 text-center font-mono">
                        {session.latest_metrics?.precision !== undefined ? `${(session.latest_metrics.precision * 100).toFixed(1)}%` : '--'}
                      </td>
                      <td className="py-4 px-4 text-stone-300 text-center font-mono">
                        {session.latest_metrics?.recall !== undefined ? `${(session.latest_metrics.recall * 100).toFixed(1)}%` : '--'}
                      </td>
                      <td className="py-4 px-4 text-stone-300 text-center font-mono">
                        {session.latest_metrics?.mAP50 !== undefined ? `${(session.latest_metrics.mAP50 * 100).toFixed(1)}%` : '--'}
                      </td>
                      <td className="py-4 px-4 text-stone-300 text-center font-mono">
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
        <div className="flex flex-col items-center justify-center py-20 bg-stone-900/20 rounded-xl border border-dashed border-stone-800">
          <ChevronRight size={48} className="text-stone-800 mb-4" />
          <h3 className="text-stone-400 font-medium">Model Comparison</h3>
          <p className="text-stone-600 text-sm mt-1">Select multiple runs from 'Past Runs' to compare side-by-side.</p>
        </div>
      )}
    </div>
  );
};

export default Evaluation;
