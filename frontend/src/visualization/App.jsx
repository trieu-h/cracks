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
  // Flow: Dataset → Model Architecture → Training Monitor → [Detection ↑ Metrics Dashboard]
  const nodes = useMemo(() => [
    {
      id: 'dataset',
      type: 'dataset',
      position: { x: 0, y: 455 },
      data: {
        config: datasetConfig,
        onConfigChange: updateDatasetConfig,
      }
    },
    {
      id: 'model',
      type: 'model',
      position: { x: 500, y: 400 },
      data: {
        config: modelConfig,
        onConfigChange: updateModelConfig,
      }
    },
    {
      id: 'training',
      type: 'training',
      position: { x: 950, y: 400 },
      data: {
        state: trainingState,
        onStart: startTraining,
        onStop: stopTraining,
      }
    },
    {
      id: 'detection',
      type: 'detection',
      position: { x: 1500, y: 150 },
      data: {
        results: detectionResults,
        onRunDetection: runDetection,
      }
    },
    {
      id: 'metrics',
      type: 'metrics',
      position: { x: 1500, y: 650 },
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

  // Define edges connecting the workflow - white dotted lines with flowing animation
  // Flow: Dataset -> Model Architecture -> Training Monitor -> Detection -> Metrics Dashboard
  const edges = useMemo(() => [
    {
      id: 'e-dataset-model',
      source: 'dataset',
      target: 'model',
      type: 'straight',
      animated: true,  // Enable dashdraw animation
      label: 'Data',
      style: { 
        stroke: '#ffffff', 
        strokeWidth: 3,
        strokeDasharray: '6,4'
      },
      labelStyle: { 
        fill: '#ffffff', 
        fontSize: 12, 
        fontWeight: 700
      },
      labelBgStyle: {
        fill: '#0c0a09',
        fillOpacity: 1,
        stroke: '#ffffff',
        strokeWidth: 1
      }
    },
    {
      id: 'e-model-training',
      source: 'model',
      target: 'training',
      type: 'straight',
      animated: true,
      label: 'Architecture',
      style: { 
        stroke: '#ffffff', 
        strokeWidth: 3,
        strokeDasharray: '6,4'
      },
      labelStyle: { 
        fill: '#ffffff', 
        fontSize: 12, 
        fontWeight: 700
      },
      labelBgStyle: {
        fill: '#0c0a09',
        fillOpacity: 1,
        stroke: '#ffffff',
        strokeWidth: 1
      }
    },
    {
      id: 'e-training-detection',
      source: 'training',
      target: 'detection',
      type: 'straight',
      animated: true,
      label: 'Model Weights',
      style: { 
        stroke: '#ffffff', 
        strokeWidth: 3,
        strokeDasharray: '6,4'
      },
      labelStyle: { 
        fill: '#ffffff', 
        fontSize: 12, 
        fontWeight: 700
      },
      labelBgStyle: {
        fill: '#0c0a09',
        fillOpacity: 1,
        stroke: '#ffffff',
        strokeWidth: 1
      }
    },
    {
      id: 'e-training-metrics',
      source: 'training',
      target: 'metrics',
      type: 'straight',
      animated: true,
      label: 'Validation Metrics',
      style: { 
        stroke: '#ffffff', 
        strokeWidth: 3,
        strokeDasharray: '6,4'
      },
      labelStyle: { 
        fill: '#ffffff', 
        fontSize: 12, 
        fontWeight: 700
      },
      labelBgStyle: {
        fill: '#0c0a09',
        fillOpacity: 1,
        stroke: '#ffffff',
        strokeWidth: 1
      }
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
