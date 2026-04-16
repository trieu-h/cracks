import { useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import { BarChart3, TrendingUp, Award, Target, Zap, Info } from 'lucide-react';
import RadarChart from '../../../components/charts/RadarChart';
import './NodeStyles.css';

const MetricsNode = ({ data }) => {
  const [showRadar, setShowRadar] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState(null);

  // Mock data for visualization
  const sessions = [
    {
      id: '1',
      latest_metrics: {
        f1: 0.92,
        precision: 0.89,
        recall: 0.95,
        mAP50: 0.91,
        mAP50_95: 0.78
      }
    }
  ];

  const metrics = [
    { key: 'f1', label: 'F1 Score', max: 1.0 },
    { key: 'precision', label: 'Precision', max: 1.0 },
    { key: 'recall', label: 'Recall', max: 1.0 },
    { key: 'mAP50', label: 'mAP50', max: 1.0 },
    { key: 'mAP50_95', label: 'mAP50-95', max: 1.0 }
  ];

  const currentMetrics = sessions[0].latest_metrics;

  const metricCards = [
    { 
      key: 'f1', 
      label: 'F1 Score', 
      value: currentMetrics.f1, 
      icon: Award,
      color: 'var(--accent-primary)',
      desc: 'Harmonic mean of precision and recall'
    },
    { 
      key: 'precision', 
      label: 'Precision', 
      value: currentMetrics.precision, 
      icon: Target,
      color: 'var(--success-text)',
      desc: 'Accuracy of positive predictions'
    },
    { 
      key: 'recall', 
      label: 'Recall', 
      value: currentMetrics.recall, 
      icon: Zap,
      color: 'var(--warning-text)',
      desc: 'Coverage of actual cracks found'
    },
    { 
      key: 'mAP50', 
      label: 'mAP50', 
      value: currentMetrics.mAP50, 
      icon: BarChart3,
      color: '#8b5cf6',
      desc: 'Mean Average Precision at IoU=0.5'
    },
    { 
      key: 'mAP50_95', 
      label: 'mAP50-95', 
      value: currentMetrics.mAP50_95, 
      icon: TrendingUp,
      color: 'var(--accent-secondary)',
      desc: 'mAP across IoU thresholds 0.5-0.95'
    }
  ];

  return (
    <div className="node-card metrics-node">
      <Handle type="target" position={Position.Left} className="node-handle" />
      
      <div className="node-header" onClick={() => data.openConceptDialog?.('metrics')}>
        <BarChart3 className="node-icon" size={18} />
        <span className="node-title">Metrics</span>
        <Info className="info-trigger" size={14} />
      </div>

      <div className="node-content">
        {/* View Toggle */}
        <div className="view-toggle">
          <button 
            className={`toggle-btn ${showRadar ? 'active' : ''}`}
            onClick={() => setShowRadar(true)}
          >
            Radar View
          </button>
          <button 
            className={`toggle-btn ${!showRadar ? 'active' : ''}`}
            onClick={() => setShowRadar(false)}
          >
            Cards View
          </button>
        </div>

        {/* Radar Chart View */}
        {showRadar ? (
          <div className="radar-section">
            <RadarChart sessions={sessions} metrics={metrics} />
          </div>
        ) : (
          /* Cards View */
          <div className="metrics-cards">
            {metricCards.map(metric => (
              <div 
                key={metric.key}
                className={`metric-card ${selectedMetric === metric.key ? 'expanded' : ''}`}
                onClick={() => setSelectedMetric(selectedMetric === metric.key ? null : metric.key)}
              >
                <div 
                  className="card-accent"
                  style={{ background: metric.color }}
                />
                <div className="card-content">
                  <div className="card-header">
                    <metric.icon size={16} style={{ color: metric.color }} />
                    <span className="card-label">{metric.label}</span>
                  </div>
                  <div className="card-value">
                    {(metric.value * 100).toFixed(1)}%
                  </div>
                  {selectedMetric === metric.key && (
                    <div className="card-desc">
                      {metric.desc}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}


      </div>

      <Handle type="source" position={Position.Right} className="node-handle" />
    </div>
  );
};

export default MetricsNode;
