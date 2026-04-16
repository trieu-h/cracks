import { Handle, NodeResizer, Position, useReactFlow } from '@xyflow/react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useNeuralNetworkStore } from '../../stores/neuralNetworkStore';
import { resolveCollisions } from '../../utils/collisionDetection';
import './NodeStyles.css';

// Preset datasets with recommended hidden layer configurations
const PRESET_DATASETS = {
  custom: {
    name: 'Custom',
    x: [[1], [2], [3], [4], [5]],
    y: [[2], [4], [6], [8], [10]],
    hiddenLayers: [1],
    predictionDefault: [7],
    description: 'Custom dataset'
  },
  twice: {
    name: 'Twice (f(x) = 2x)',
    x: [[1], [2], [3], [4], [5], [6], [7], [8]],
    y: [[2], [4], [6], [8], [10], [12], [14], [16]],
    hiddenLayers: [1],
    predictionDefault: [10],
    description: 'Linear function - multiply by 2'
  },
  xor: {
    name: 'XOR Gate',
    x: [[0, 0], [0, 1], [1, 0], [1, 1]],
    y: [[0], [1], [1], [0]],
    hiddenLayers: [2],
    predictionDefault: [1, 0],
    description: 'Non-linear XOR logic gate'
  },
  and: {
    name: 'AND Gate',
    x: [[0, 0], [0, 1], [1, 0], [1, 1]],
    y: [[0], [0], [0], [1]],
    hiddenLayers: [1],
    predictionDefault: [1, 1],
    description: 'AND logic gate'
  },
  or: {
    name: 'OR Gate',
    x: [[0, 0], [0, 1], [1, 0], [1, 1]],
    y: [[0], [1], [1], [1]],
    hiddenLayers: [1],
    predictionDefault: [0, 0],
    description: 'OR logic gate'
  },
  quadratic: {
    name: 'Quadratic (f(x) = x²)',
    x: [[0], [1], [2], [3], [4], [5]],
    y: [[0], [1], [4], [9], [16], [25]],
    hiddenLayers: [4],
    predictionDefault: [6],
    description: 'Quadratic function'
  },
  sine: {
    name: 'Sine Wave',
    x: [[0], [0.5], [1], [1.5], [2], [2.5], [3]],
    y: [[0], [0.479], [0.841], [0.997], [0.909], [0.599], [0.141]],
    hiddenLayers: [6],
    predictionDefault: [3.5],
    description: 'Sine wave approximation'
  },
  circle: {
    name: 'Circle Classification',
    x: [
      [0, 0], [0, 1], [1, 0], [1, 1],
      [0.5, 0.5], [0.8, 0.8], [0.2, 0.2],
      [0.9, 0.1], [0.1, 0.9]
    ],
    y: [[0], [0], [0], [0], [1], [0], [1], [0], [0]],
    hiddenLayers: [4],
    predictionDefault: [0.5, 0.5],
    description: 'Classify points inside/outside circle'
  }
};

export default function TrainingDataNode({ data, selected }) {
  const { x: initialX = [], y: initialY = [], openConceptDialog } = data;
  const [x, setX] = useState(initialX);
  const [y, setY] = useState(initialY);
  const [selectedPreset, setSelectedPreset] = useState('twice');
  const [hasError, setHasError] = useState(false);
  const [isDataCollapsed, setIsDataCollapsed] = useState(true);
  const updateTrainingData = useNeuralNetworkStore(state => state.updateTrainingData);
  const updateConfig = useNeuralNetworkStore(state => state.updateConfig);
  const { setNodes } = useReactFlow();
  const saveTimeoutRef = useRef(null);

  const handleResizeEnd = () => {
    setNodes((nds) =>
      resolveCollisions(nds, {
        maxIterations: 50,
        overlapThreshold: 0.5,
        margin: 15,
      })
    );
  };

  const parseValue = (value) => {
    if (typeof value === 'string') {
      const parsed = value.split(',').map(v => {
        const num = parseFloat(v.trim());
        return isNaN(num) ? 0 : num;
      }).filter(n => !isNaN(n));
      return parsed.length > 0 ? parsed : [0];
    }
    if (Array.isArray(value)) {
      return value;
    }
    return [value];
  };

  const saveData = useCallback(() => {
    // Parse and validate data
    const parsedX = x.map(parseValue).filter(arr => arr.length > 0);
    const parsedY = y.map(parseValue).filter(arr => arr.length > 0);

    console.log('[TrainingDataNode] Auto-save - raw x:', x);
    console.log('[TrainingDataNode] Auto-save - raw y:', y);
    console.log('[TrainingDataNode] Auto-save - parsed x:', parsedX);
    console.log('[TrainingDataNode] Auto-save - parsed y:', parsedY);

    if (parsedX.length === 0 || parsedY.length === 0) {
      setHasError(true);
      return;
    }

    if (parsedX.length !== parsedY.length) {
      setHasError(true);
      return;
    }

    // Clear any previous errors
    setHasError(false);

    // Save to Zustand store
    console.log('[TrainingDataNode] Saving to store:', { x: parsedX, y: parsedY });
    updateTrainingData({ x: parsedX, y: parsedY });

    // Update hidden layers and prediction input if using a preset
    if (selectedPreset !== 'custom' && PRESET_DATASETS[selectedPreset]) {
      const dataset = PRESET_DATASETS[selectedPreset];
      console.log('[TrainingDataNode] Updating config with hiddenLayers:', dataset.hiddenLayers);
      updateConfig({ hiddenLayers: dataset.hiddenLayers });
      // Update prediction input to match the dataset's default
      if (dataset.predictionDefault) {
        console.log('[TrainingDataNode] Updating prediction input:', dataset.predictionDefault);
        useNeuralNetworkStore.getState().updatePredictionInput(dataset.predictionDefault);
      }
    }
  }, [x, y, selectedPreset, updateTrainingData, updateConfig]);

  // Auto-save with debouncing (500ms delay)
  useEffect(() => {
    // Clear any pending save
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    // Schedule new save
    saveTimeoutRef.current = setTimeout(() => {
      saveData();
    }, 500);

    // Cleanup on unmount
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [x, y, saveData]);

  const handlePresetChange = (e) => {
    const preset = e.target.value;
    console.log('[TrainingDataNode] Preset changed to:', preset);
    setSelectedPreset(preset);
    setHasError(false);

    if (preset !== 'custom' && PRESET_DATASETS[preset]) {
      const dataset = PRESET_DATASETS[preset];
      console.log('[TrainingDataNode] Loading preset data:', {
        x: dataset.x,
        y: dataset.y,
        predictionDefault: dataset.predictionDefault
      });
      setX(dataset.x);
      setY(dataset.y);
      // Also update prediction input immediately
      if (dataset.predictionDefault) {
        console.log('[TrainingDataNode] Setting prediction default:', dataset.predictionDefault);
        useNeuralNetworkStore.getState().updatePredictionInput(dataset.predictionDefault);
      }
    }
  };

  const addSample = () => {
    setX([...x, [0]]);
    setY([...y, [0]]);
  };

  const removeSample = (index) => {
    setX(x.filter((_, i) => i !== index));
    setY(y.filter((_, i) => i !== index));
  };

  return (
    <div className="custom-node training-data-node">
      <NodeResizer
        color="#4ade80"
        isVisible={selected}
        minWidth={200}
        minHeight={150}
        onResizeEnd={handleResizeEnd}
      />
      <div
        className="node-header clickable-header"
        onClick={(e) => {
          if (openConceptDialog) {
            e.stopPropagation();
            openConceptDialog('training-data');
          }
        }}
        title="Click to learn about Training Data"
        style={{ cursor: openConceptDialog ? 'pointer' : 'default' }}
      >
        2. Training Data
      </div>
      <div className="node-content">
        <div className="data-section" style={{ marginBottom: '10px' }}>
          <div className="section-title">Dataset Preset</div>
          <select
            value={selectedPreset}
            onChange={handlePresetChange}
            onMouseDown={(e) => e.stopPropagation()}
            className="nodrag"
            style={{
              width: '100%',
              padding: '6px',
              fontSize: '14px',
              borderRadius: '4px',
              border: '1px solid #444',
              backgroundColor: '#2a2a2a',
              color: '#fff',
              cursor: 'pointer'
            }}
          >
            {Object.entries(PRESET_DATASETS).map(([key, dataset]) => (
              <option key={key} value={key}>
                {dataset.name}
              </option>
            ))}
          </select>
          {selectedPreset !== 'custom' && PRESET_DATASETS[selectedPreset] && (
            <div style={{
              fontSize: '12px',
              color: '#888',
              marginTop: '4px',
              fontStyle: 'italic'
            }}>
              {PRESET_DATASETS[selectedPreset].description}
              {' '} (Hidden: [{PRESET_DATASETS[selectedPreset].hiddenLayers.join(', ')}])
            </div>
          )}
        </div>
        <div className="data-section" style={{ marginTop: '10px' }}>
          <div
            className="section-title"
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              cursor: 'pointer',
              userSelect: 'none',
              padding: '4px 0'
            }}
            onClick={(e) => {
              e.stopPropagation();
              setIsDataCollapsed(!isDataCollapsed);
            }}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <span>Data</span>
            <span style={{
              fontSize: '12px',
              color: '#aaa',
              transition: 'transform 0.2s',
              transform: isDataCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)'
            }}>
              ▼
            </span>
          </div>
          {!isDataCollapsed && (
            <>
              <div className="data-section" style={{ marginTop: '8px' }}>
                <div className="section-title">Input x[]</div>
                {x.map((sample, idx) => (
                  <div key={idx} className="sample-row">
                    <input
                      type="text"
                      value={Array.isArray(sample) ? sample.join(', ') : sample}
                      onChange={(e) => {
                        const newX = [...x];
                        newX[idx] = e.target.value;
                        setX(newX);
                        setHasError(false);
                      }}
                      onMouseDown={(e) => e.stopPropagation()}
                      onPointerDown={(e) => e.stopPropagation()}
                      className="data-input nodrag"
                      placeholder="1"
                      style={hasError ? { border: '2px solid #ef4444' } : {}}
                    />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeSample(idx);
                      }}
                      onMouseDown={(e) => e.stopPropagation()}
                      className="remove-btn nodrag"
                    >×</button>
                  </div>
                ))}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    addSample();
                  }}
                  onMouseDown={(e) => e.stopPropagation()}
                  className="add-btn nodrag"
                >+ Add Sample</button>
              </div>
              <div className="data-section">
                <div className="section-title">Expected Output y[]</div>
                {y.map((sample, idx) => (
                  <div key={idx} className="sample-row">
                    <input
                      type="text"
                      value={Array.isArray(sample) ? sample.join(', ') : sample}
                      onChange={(e) => {
                        const newY = [...y];
                        newY[idx] = e.target.value;
                        setY(newY);
                        setHasError(false);
                      }}
                      onMouseDown={(e) => e.stopPropagation()}
                      onPointerDown={(e) => e.stopPropagation()}
                      className="data-input nodrag"
                      placeholder="0"
                      style={hasError ? { border: '2px solid #ef4444' } : {}}
                    />
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
      <Handle type="source" position={Position.Right} id="output" />
    </div>
  );
}
