import { Handle, NodeResizer, Position, useReactFlow } from '@xyflow/react';
import { Play, Square } from 'lucide-react';
import { useState } from 'react';
import {
  selectCurrentStep,
  selectIsTrained,
  selectIsTraining,
  selectTrainingData,
  selectTrainingHistory,
  useNeuralNetworkStore,
} from '../../stores/neuralNetworkStore';
import { resolveCollisions } from '../../utils/collisionDetection';
import Cube3D from '../Cube3D';
import './NodeStyles.css';

export default function NeuralNetworkNode({ data, selected }) {
  // Use Zustand selectors - only re-renders when specific values change
  const trainingData = useNeuralNetworkStore(selectTrainingData);
  const isTraining = useNeuralNetworkStore(selectIsTraining);
  const currentStep = useNeuralNetworkStore(selectCurrentStep);
  const trainingHistory = useNeuralNetworkStore(selectTrainingHistory);
  const isTrained = useNeuralNetworkStore(selectIsTrained);

  const {
    isExpanded = false,
    onToggle,
    config = {},
    trainingConfig = {},
    onTrain,
    onCancelTraining,
    openConceptDialog,
  } = data;

  const [isRunning, setIsRunning] = useState(false);
  const [localStep, setLocalStep] = useState(0);
  const [localLoss, setLocalLoss] = useState(null);
  const { setNodes } = useReactFlow();

  const handleResizeEnd = () => {
    setNodes((nds) =>
      resolveCollisions(nds, {
        maxIterations: 50,
        overlapThreshold: 0.5,
        margin: 15,
      })
    );
  };

  // Use context state (updates live during training)
  const displayStep = isRunning ? localStep : currentStep;
  const displayLoss = isRunning
    ? localLoss
    : (trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1]?.loss : null);

  const handleTrain = async (e) => {
    e.stopPropagation(); // Prevent expand/collapse when clicking train

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

  // Note: When expanded, this node becomes type 'group' and React Flow handles it differently
  // The group header is rendered by GroupNode component
  if (isExpanded) {
    return null; // Group nodes are handled by React Flow's built-in group rendering
  }

  // When collapsed, render normal node with cube and training button
  return (
    <div className={`custom-node neural-network-node`}>
      <NodeResizer
        color="#646cff"
        isVisible={selected}
        minWidth={250}
        minHeight={200}
        onResizeEnd={handleResizeEnd}
      />
      <Handle type="target" position={Position.Left} />
      <div
        className="node-header clickable-header"
        onClick={(e) => {
          if (openConceptDialog) {
            e.stopPropagation();
            openConceptDialog('neural-network');
          }
        }}
        title="Click to learn about Neural Networks"
      >
        1. Neural Network
      </div>
      <div className="node-content">
        <div className="cube-container" onClick={onToggle}>
          <Cube3D onClick={onToggle} isExpanded={isExpanded} />
        </div>
        <div className="train-section-collapsed">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <button
                onClick={handleTrain}
                disabled={isRunning || isTraining}
                className="icon-btn train-btn"
                title={isRunning || isTraining ? 'Training...' : 'Start Training'}
                style={{ width: '32px', height: '32px' }}
              >
                <Play size={18} />
              </button>
              {(isRunning || isTraining) && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onCancelTraining?.();
                  }}
                  className="icon-btn cancel-btn"
                  title="Stop Training"
                  style={{ width: '32px', height: '32px' }}
                >
                  <Square size={18} />
                </button>
              )}
            </div>
            {(isRunning || isTraining || isTrained) && (displayStep > 0 || isTrained) && (
              <div className="training-status" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '2px' }}>
                <div className="section-title" style={{ fontSize: '11px', margin: 0 }}>Step: {displayStep + 1} / {trainingConfig.steps}</div>
                {displayLoss !== null && (
                  <div className="section-title" style={{ fontSize: '11px', margin: 0 }}>Loss: {displayLoss.toFixed(6)}</div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} id="output" />
      <Handle type="source" position={Position.Bottom} id="bottom-left" style={{ left: '33%' }} />
      <Handle type="source" position={Position.Bottom} id="bottom-right" style={{ left: '66%' }} />
    </div>
  );
}
