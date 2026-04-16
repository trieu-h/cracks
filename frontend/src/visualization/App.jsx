import { useCallback, useMemo, useState } from 'react';
import './App.css';
import FlowCanvas from './components/FlowCanvas';
import DatasetNode from './components/nodes/DatasetNode';
import ModelNode from './components/nodes/ModelNode';
import TrainingNode from './components/nodes/TrainingNode';
import DetectionNode from './components/nodes/DetectionNode';
import MetricsNode from './components/nodes/MetricsNode';
import { useCrackDetectionStore } from './stores/crackDetectionStore';

const nodeTypes = {
  dataset: DatasetNode,
  model: ModelNode,
  training: TrainingNode,
  detection: DetectionNode,
  metrics: MetricsNode,
};

const edgeTypes = {
  default: 'smoothstep',
  smoothstep: 'smoothstep',
};

function App() {
  // Get state from crack detection store
  const datasetConfig = useCrackDetectionStore(state => state.datasetConfig);
  const modelConfig = useCrackDetectionStore(state => state.modelConfig);
  const trainingState = useCrackDetectionStore(state => state.trainingState);
  const detectionResults = useCrackDetectionStore(state => state.detectionResults);
  const metrics = useCrackDetectionStore(state => state.metrics);
  
  const updateDatasetConfig = useCrackDetectionStore(state => state.updateDatasetConfig);
  const updateModelConfig = useCrackDetectionStore(state => state.updateModelConfig);
  const startTraining = useCrackDetectionStore(state => state.startTraining);
  const stopTraining = useCrackDetectionStore(state => state.stopTraining);
  const runDetection = useCrackDetectionStore(state => state.runDetection);
  const resetMetrics = useCrackDetectionStore(state => state.resetMetrics);

  // Define nodes for the crack detection workflow
  const nodes = useMemo(() => [
    {
      id: 'dataset',
      type: 'dataset',
      position: { x: 50, y: 100 },
      data: {
        config: datasetConfig,
        onConfigChange: updateDatasetConfig,
      }
    },
    {
      id: 'model',
      type: 'model',
      position: { x: 450, y: 100 },
      data: {
        config: modelConfig,
        onConfigChange: updateModelConfig,
      }
    },
    {
      id: 'training',
      type: 'training',
      position: { x: 850, y: 100 },
      data: {
        state: trainingState,
        onStart: startTraining,
        onStop: stopTraining,
      }
    },
    {
      id: 'detection',
      type: 'detection',
      position: { x: 450, y: 450 },
      data: {
        results: detectionResults,
        onRunDetection: runDetection,
      }
    },
    {
      id: 'metrics',
      type: 'metrics',
      position: { x: 850, y: 450 },
      data: {
        metrics: metrics,
        onReset: resetMetrics,
      }
    }
  ], [
    datasetConfig, modelConfig, trainingState, detectionResults, metrics,
    updateDatasetConfig, updateModelConfig, startTraining, stopTraining, 
    runDetection, resetMetrics
  ]);

  // Define edges connecting the workflow - use hardcoded colors for SVG
  const edges = useMemo(() => [
    {
      id: 'e-dataset-model',
      source: 'dataset',
      target: 'model',
      type: 'smoothstep',
      animated: true,
      label: 'Data',
      style: { stroke: '#22c55e', strokeWidth: 2 },
      labelStyle: { fill: '#22c55e', fontSize: 12, fontWeight: 500 }
    },
    {
      id: 'e-model-training',
      source: 'model',
      target: 'training',
      type: 'smoothstep',
      animated: true,
      label: 'Architecture',
      style: { stroke: '#4ade80', strokeWidth: 2 },
      labelStyle: { fill: '#4ade80', fontSize: 12, fontWeight: 500 }
    },
    {
      id: 'e-training-detection',
      source: 'training',
      target: 'detection',
      type: 'smoothstep',
      animated: true,
      label: 'Model Weights',
      style: { stroke: '#22c55e', strokeWidth: 2 },
      labelStyle: { fill: '#22c55e', fontSize: 12, fontWeight: 500 }
    },
    {
      id: 'e-training-metrics',
      source: 'training',
      target: 'metrics',
      type: 'smoothstep',
      animated: true,
      label: 'Validation',
      style: { stroke: '#3b82f6', strokeWidth: 2 },
      labelStyle: { fill: '#3b82f6', fontSize: 12, fontWeight: 500 }
    },
    {
      id: 'e-detection-metrics',
      source: 'detection',
      target: 'metrics',
      type: 'smoothstep',
      animated: false,
      label: 'Predictions',
      style: { stroke: '#f59e0b', strokeWidth: 2 },
      labelStyle: { fill: '#f59e0b', fontSize: 12, fontWeight: 500 }
    }
  ], []);

  return (
    <div className="app">
      <FlowCanvas
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
      />
    </div>
  );
}

export default App;
