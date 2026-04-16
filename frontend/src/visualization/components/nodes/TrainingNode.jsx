import { useState, useEffect, useRef } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Play, Pause, Square, Activity, TrendingDown, Clock, RotateCcw, Info } from 'lucide-react';
import './NodeStyles.css';

// Simple chart component using canvas
const MiniChart = ({ data, color, width = 200, height = 60 }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length < 2) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 2;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear with light background
    ctx.fillStyle = '#1c1917';  // Dark but visible background
    ctx.fillRect(0, 0, width, height);

    // Draw subtle grid lines
    ctx.strokeStyle = '#44403c';  // Visible gray
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let i = 1; i < 4; i++) {
      const y = (height / 4) * i;
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
    }
    ctx.stroke();

    // Draw chart line with bright color
    const maxVal = Math.max(...data, 1);
    const minVal = Math.min(...data, 0);
    const range = maxVal - minVal || 1;

    ctx.beginPath();
    ctx.strokeStyle = '#4ade80';  // Bright green, always visible
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    data.forEach((val, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((val - minVal) / range) * (height - 10) - 5;
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw current point with glow effect
    const lastY = height - ((data[data.length - 1] - minVal) / range) * (height - 10) - 5;
    
    // Glow
    ctx.beginPath();
    ctx.arc(width - 4, lastY, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(74, 222, 128, 0.3)';
    ctx.fill();
    
    // Point
    ctx.beginPath();
    ctx.arc(width - 4, lastY, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#4ade80';
    ctx.fill();
  }, [data, color, width, height]);

  return <canvas ref={canvasRef} className="mini-chart" style={{ width, height }} />;
};

const TrainingNode = ({ data }) => {
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs] = useState(100);
  const [lossHistory, setLossHistory] = useState([]);
  const [mapHistory, setMapHistory] = useState([]);
  const [currentLoss, setCurrentLoss] = useState(2.5);
  const [currentMap, setCurrentMap] = useState(0);
  const [trainingTime, setTrainingTime] = useState(0);
  const intervalRef = useRef(null);
  const startTimeRef = useRef(null);

  // Simulate training
  useEffect(() => {
    if (isTraining) {
      if (!startTimeRef.current) startTimeRef.current = Date.now();
      
      intervalRef.current = setInterval(() => {
        setCurrentEpoch(prev => {
          const next = prev + 1;
          if (next >= totalEpochs) {
            setIsTraining(false);
            return totalEpochs;
          }
          
          // Simulate metrics
          const newLoss = 2.5 * Math.exp(-next / 30) + 0.1 + Math.random() * 0.05;
          const newMap = Math.min(0.95, 0.3 + (next / totalEpochs) * 0.65 + Math.random() * 0.05);
          
          setCurrentLoss(newLoss);
          setCurrentMap(newMap);
          setLossHistory(h => [...h.slice(-50), newLoss]);
          setMapHistory(h => [...h.slice(-50), newMap]);
          
          // Update time
          const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
          setTrainingTime(elapsed);
          
          return next;
        });
      }, 200); // Fast simulation
    } else {
      clearInterval(intervalRef.current);
    }

    return () => clearInterval(intervalRef.current);
  }, [isTraining, totalEpochs]);

  const startTraining = () => {
    if (currentEpoch >= totalEpochs) {
      // Reset
      setCurrentEpoch(0);
      setLossHistory([]);
      setMapHistory([]);
      setCurrentLoss(2.5);
      setCurrentMap(0);
      startTimeRef.current = null;
      setTrainingTime(0);
    }
    setIsTraining(true);
  };

  const stopTraining = () => {
    setIsTraining(false);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const progress = (currentEpoch / totalEpochs) * 100;

  return (
    <div className="node-card training-node">
      <Handle type="target" position={Position.Left} className="node-handle" />
      
      <div className="node-header" onClick={() => data.openConceptDialog?.('training')}>
        <Activity className="node-icon" size={18} />
        <span className="node-title">Training Monitor</span>
        <Info className="info-trigger" size={14} />
      </div>

      <div className="node-content">
        {/* Progress Bar */}
        <div className="progress-section">
          <div className="progress-header">
            <span className="epoch-counter">Epoch {currentEpoch}/{totalEpochs}</span>
            <span className="time-counter">
              <Clock size={12} />
              {formatTime(trainingTime)}
            </span>
          </div>
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Charts */}
        <div className="charts-grid">
          <div className="chart-box">
            <div className="chart-header">
              <TrendingDown size={14} />
              <span>Loss: {currentLoss.toFixed(4)}</span>
            </div>
            <MiniChart 
              data={lossHistory.length > 0 ? lossHistory : [2.5, 2.4, 2.3, 2.2]} 
              color="var(--accent-primary)" 
            />
          </div>
          
          <div className="chart-box">
            <div className="chart-header">
              <Activity size={14} />
              <span>mAP50: {(currentMap * 100).toFixed(1)}%</span>
            </div>
            <MiniChart 
              data={mapHistory.length > 0 ? mapHistory.map(m => m * 100) : [30, 35, 42, 48]} 
              color="var(--success-text)" 
            />
          </div>
        </div>

        {/* Live Preview */}
        <div className="preview-section">
          <span className="preview-label">Recent Predictions</span>
          <div className="preview-thumbs">
            {[1, 2, 3].map(i => (
              <div key={i} className="preview-thumb">
                <div className="thumb-placeholder">
                  <span>Epoch {(Math.max(1, currentEpoch - 3 + i))}</span>
                </div>
                {isTraining && (
                  <div className="thumb-overlay">
                    <Activity size={12} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div className="training-controls">
          {!isTraining ? (
            <button className="control-btn primary" onClick={startTraining}>
              <Play size={16} />
              {currentEpoch >= totalEpochs ? 'Retrain' : 'Start Training'}
            </button>
          ) : (
            <>
              <button className="control-btn secondary" onClick={stopTraining}>
                <Pause size={16} />
                Pause
              </button>
              <button className="control-btn danger" onClick={stopTraining}>
                <Square size={16} />
                Stop
              </button>
            </>
          )}
          
          <button 
            className="control-btn ghost"
            onClick={() => {
              setCurrentEpoch(0);
              setLossHistory([]);
              setMapHistory([]);
              setCurrentLoss(2.5);
              setCurrentMap(0);
              setTrainingTime(0);
              startTimeRef.current = null;
              setIsTraining(false);
            }}
            disabled={isTraining || currentEpoch === 0}
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      <Handle type="source" position={Position.Right} className="node-handle" />
    </div>
  );
};

export default TrainingNode;
