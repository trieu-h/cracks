import { Handle, NodeResizer, Position, useReactFlow } from '@xyflow/react';
import { Minus, Play, Save, Square } from 'lucide-react';
import { useEffect, useState } from 'react';
import { resolveCollisions } from '../../utils/collisionDetection';
import './NodeStyles.css';

// Custom group node component for neural network group
export default function GroupNode({ data, selected }) {
  const {
    onToggle,
    onResize,
    minGroupSize = { width: 1000, height: 220 },
    label = 'Neural Network',
    trainingConfig = {},
    config = {},
    onTrain,
    onCancelTraining,
    onNetworkConfigChange,
    onTrainingConfigChange,
    isTraining = false,
    isTrained = false,
    openConceptDialog
  } = data;

  const [isRunning, setIsRunning] = useState(false);
  const [, setLocalStep] = useState(0);
  const [, setLocalLoss] = useState(null);
  const { setNodes } = useReactFlow();

  const handleResizeEnd = (event, params) => {
    onResize?.({ width: params.width, height: params.height });
    setNodes((nds) =>
      resolveCollisions(nds, {
        maxIterations: 50,
        overlapThreshold: 0.5,
        margin: 15,
      })
    );
  };

  // Extract config values with defaults
  const hiddenLayers = config.hiddenLayers || [4];
  const activation = config.activation || 'relu';
  const costFunction = config.costFunction || 'mse';
  const steps = trainingConfig.steps || 100;
  const learningRate = trainingConfig.learningRate || 0.01;
  const method = trainingConfig.method || 'backpropagation';

  // Local state for pending configuration changes
  const [pendingHiddenLayers, setPendingHiddenLayers] = useState(hiddenLayers.join(', '));
  const [pendingActivation, setPendingActivation] = useState(activation);
  const [pendingCostFunction, setPendingCostFunction] = useState(costFunction);
  const [pendingSteps, setPendingSteps] = useState(steps);
  const [pendingLearningRate, setPendingLearningRate] = useState(learningRate);
  const [pendingMethod, setPendingMethod] = useState(method);

  // Sync pending hidden layers when config changes
  useEffect(() => {
    setPendingHiddenLayers(hiddenLayers.join(', '));
  }, [hiddenLayers.join(', ')]);

  // Configuration change handlers (now just update local state)
  const handleSaveConfig = (e) => {
    e.stopPropagation();

    const newHiddenLayers = pendingHiddenLayers.split(',').map(v => parseInt(v.trim()) || 1).filter(v => v > 0);
    onNetworkConfigChange?.({ hiddenLayers: newHiddenLayers, activation: pendingActivation, costFunction: pendingCostFunction });
    onTrainingConfigChange?.({ steps: pendingSteps, learningRate: pendingLearningRate, method: pendingMethod });
  };

  const handleTrain = async (e) => {
    e.stopPropagation();

    if (!trainingConfig.steps || trainingConfig.steps < 1) {
      alert('Training steps must be provided and greater than 0');
      return;
    }

    setIsRunning(true);
    setLocalStep(0);
    setLocalLoss(null);

    try {
      await onTrain?.(async (step, loss, parameters) => {
        setLocalStep(step);
        setLocalLoss(loss);
      });
    } catch (error) {
      console.error('Training error:', error);
      alert('Training failed: ' + error.message);
    } finally {
      setIsRunning(false);
    }
  };

  const handleCancel = (e) => {
    e.stopPropagation();
    onCancelTraining?.();
  };

  return (
    <>
      <NodeResizer
        color="#646cff"
        isVisible={selected}
        minWidth={minGroupSize.width}
        minHeight={minGroupSize.height}
        onResizeEnd={handleResizeEnd}
      />
      <Handle type="target" position={Position.Left} id="input" />
      <Handle type="source" position={Position.Right} id="output" />
      <Handle type="source" position={Position.Bottom} id="bottom-left" style={{ left: '33%' }} />
      <Handle type="source" position={Position.Bottom} id="bottom-right" style={{ left: '66%' }} />
      <div className="neural-network-group-header">
        <div
          className="group-title clickable-header"
          onClick={(e) => {
            if (openConceptDialog) {
              e.stopPropagation();
              openConceptDialog('neural-network');
            }
          }}
          title="Click to learn about Neural Networks"
        >
          1. {label}
        </div>
        {isTrained && (
          <div className="config-warning">⚠️ Changes will invalidate training</div>
        )}
        <div className="config-panel">
          <div className="config-row">
            <label className="config-label">
              <span
                className="label-text clickable-label"
                onClick={(e) => {
                  if (openConceptDialog) {
                    e.stopPropagation();
                    openConceptDialog('hidden-layer');
                  }
                }}
                title="Click to learn about Hidden Layers"
              >
                Hidden Layers:
              </span>
              <input
                type="text"
                value={pendingHiddenLayers}
                onChange={(e) => setPendingHiddenLayers(e.target.value)}
                onMouseDown={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
                className="config-input nodrag"
                placeholder="4, 4"
                title="Input/output sizes auto-detected from training data"
              />
            </label>
            <label className="config-label">
              <span
                className="label-text clickable-label"
                onClick={(e) => {
                  if (openConceptDialog) {
                    e.stopPropagation();
                    openConceptDialog('activation');
                  }
                }}
                title="Click to learn about Activation Functions"
              >
                Activation:
              </span>
              <select
                value={pendingActivation}
                onChange={(e) => setPendingActivation(e.target.value)}
                onMouseDown={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
                className="config-select nodrag"
              >
                <option value="relu">ReLU</option>
                <option value="sigmoid">Sigmoid</option>
              </select>
            </label>
            <label className="config-label">
              <span
                className="label-text clickable-label"
                onClick={(e) => {
                  if (openConceptDialog) {
                    e.stopPropagation();
                    openConceptDialog('learning-rate');
                  }
                }}
                title="Click to learn about Learning Rate"
              >
                LR:
              </span>
              <input
                type="number"
                value={pendingLearningRate}
                onChange={(e) => setPendingLearningRate(parseFloat(e.target.value) || 0)}
                onMouseDown={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
                className="config-input nodrag"
                placeholder="0.01"
                min="0.0001"
                step="0.001"
              />
            </label>
            <label className="config-label">
              <span
                className="label-text clickable-label"
                onClick={(e) => {
                  if (openConceptDialog) {
                    e.stopPropagation();
                    openConceptDialog('step');
                  }
                }}
                title="Click to learn about Training Steps"
              >
                Steps:
              </span>
              <input
                type="number"
                value={pendingSteps}
                onChange={(e) => setPendingSteps(parseInt(e.target.value) || 0)}
                onMouseDown={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
                className="config-input nodrag"
                placeholder="100"
                min="1"
              />
            </label>
            <label className="config-label">
              <span
                className="label-text clickable-label"
                onClick={(e) => {
                  if (openConceptDialog) {
                    e.stopPropagation();
                    openConceptDialog('cost');
                  }
                }}
                title="Click to learn about Loss Function"
              >
                Loss:
              </span>
              <select
                value={pendingCostFunction}
                onChange={(e) => setPendingCostFunction(e.target.value)}
                onMouseDown={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
                className="config-select nodrag"
              >
                <option value="mse">MSE</option>
              </select>
            </label>
            <label className="config-label">
              <span
                className="label-text clickable-label"
                onClick={(e) => {
                  if (openConceptDialog) {
                    e.stopPropagation();
                    openConceptDialog('method');
                  }
                }}
                title="Click to learn about Training Methods"
              >
                Method:
              </span>
              <select
                value={pendingMethod}
                onChange={(e) => setPendingMethod(e.target.value)}
                onMouseDown={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
                className="config-select nodrag"
              >
                <option value="backpropagation">Backpropagation</option>
                <option value="finite-difference">Finite Difference</option>
              </select>
            </label>
            <button
              onClick={handleSaveConfig}
              className="icon-btn save-btn"
              title="Save configuration"
            >
              <Save size={18} />
            </button>
          </div>
        </div>
        <div className="group-header-controls">
          <button
            onClick={handleTrain}
            disabled={isRunning || isTraining}
            className={`icon-btn train-btn ${isRunning || isTraining ? 'running' : ''}`}
            title={isRunning || isTraining ? 'Training...' : 'Start Training'}
          >
            <Play size={18} fill="currentColor" />
          </button>
          {(isRunning || isTraining) && (
            <button
              onClick={handleCancel}
              className="icon-btn cancel-btn"
              title="Cancel training"
            >
              <Square size={18} fill="currentColor" />
            </button>
          )}
          <button onClick={onToggle} className="icon-btn collapse-btn" title="Collapse"><Minus size={18} /></button>
        </div>
      </div>
    </>
  );
}
