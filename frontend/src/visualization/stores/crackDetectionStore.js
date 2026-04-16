import { create } from 'zustand';

/**
 * Zustand store for crack detection workflow state management
 * Manages dataset, model, training, detection, and metrics state
 */
export const useCrackDetectionStore = create((set, get) => ({
  // Dataset configuration
  datasetConfig: {
    yamlPath: 'data/crack_dataset.yaml',
    trainPath: 'data/images/train',
    valPath: 'data/images/val',
    testPath: 'data/images/test',
    numClasses: 1,
    classNames: ['crack'],
    uploadedImages: [],
  },
  
  // Model configuration
  modelConfig: {
    type: 'yolov11',
    size: 'small',
    inputSize: 640,
    batchSize: 16,
    epochs: 100,
    learningRate: 0.001,
    optimizer: 'AdamW',
    augmentations: {
      flip: true,
      rotate: true,
      mosaic: true,
      mixup: false,
    }
  },
  
  // Training state
  trainingState: {
    isTraining: false,
    isPaused: false,
    currentEpoch: 0,
    totalEpochs: 100,
    currentBatch: 0,
    totalBatches: 0,
    loss: {
      boxLoss: 0,
      clsLoss: 0,
      dflLoss: 0,
      totalLoss: 0,
    },
    history: {
      loss: [],
      map50: [],
      map50_95: [],
      precision: [],
      recall: [],
    }
  },
  
  // Detection results
  detectionResults: {
    inputImage: null,
    outputImage: null,
    detections: [],
    processingTime: 0,
    confidenceThreshold: 0.25,
    nmsThreshold: 0.45,
  },
  
  // Metrics
  metrics: {
    f1Score: 0,
    precision: 0,
    recall: 0,
    map50: 0,
    map50_95: 0,
    inferenceTime: 0,
    totalImages: 0,
    totalDetections: 0,
  },
  
  // Actions - Dataset
  updateDatasetConfig: (config) => set((state) => ({
    datasetConfig: { ...state.datasetConfig, ...config }
  })),
  
  addUploadedImage: (image) => set((state) => ({
    datasetConfig: {
      ...state.datasetConfig,
      uploadedImages: [...state.datasetConfig.uploadedImages, image].slice(0, 6)
    }
  })),
  
  removeUploadedImage: (id) => set((state) => ({
    datasetConfig: {
      ...state.datasetConfig,
      uploadedImages: state.datasetConfig.uploadedImages.filter(img => img.id !== id)
    }
  })),
  
  // Actions - Model
  updateModelConfig: (config) => set((state) => ({
    modelConfig: { ...state.modelConfig, ...config }
  })),
  
  setModelType: (type) => set((state) => ({
    modelConfig: { ...state.modelConfig, type }
  })),
  
  setModelSize: (size) => set((state) => ({
    modelConfig: { ...state.modelConfig, size }
  })),
  
  toggleAugmentation: (key) => set((state) => ({
    modelConfig: {
      ...state.modelConfig,
      augmentations: {
        ...state.modelConfig.augmentations,
        [key]: !state.modelConfig.augmentations[key]
      }
    }
  })),
  
  // Actions - Training
  startTraining: () => set((state) => ({
    trainingState: {
      ...state.trainingState,
      isTraining: true,
      isPaused: false,
    }
  })),
  
  stopTraining: () => set((state) => ({
    trainingState: {
      ...state.trainingState,
      isTraining: false,
      isPaused: false,
    }
  })),
  
  pauseTraining: () => set((state) => ({
    trainingState: {
      ...state.trainingState,
      isPaused: true,
    }
  })),
  
  resumeTraining: () => set((state) => ({
    trainingState: {
      ...state.trainingState,
      isPaused: false,
    }
  })),
  
  updateTrainingProgress: (progress) => set((state) => ({
    trainingState: {
      ...state.trainingState,
      ...progress,
    }
  })),
  
  addTrainingHistory: (epoch, data) => set((state) => ({
    trainingState: {
      ...state.trainingState,
      history: {
        loss: [...state.trainingState.history.loss, data.loss].slice(-50),
        map50: [...state.trainingState.history.map50, data.map50].slice(-50),
        map50_95: [...state.trainingState.history.map50_95, data.map50_95].slice(-50),
        precision: [...state.trainingState.history.precision, data.precision].slice(-50),
        recall: [...state.trainingState.history.recall, data.recall].slice(-50),
      }
    }
  })),
  
  resetTraining: () => set((state) => ({
    trainingState: {
      ...state.trainingState,
      isTraining: false,
      isPaused: false,
      currentEpoch: 0,
      currentBatch: 0,
      loss: { boxLoss: 0, clsLoss: 0, dflLoss: 0, totalLoss: 0 },
      history: { loss: [], map50: [], map50_95: [], precision: [], recall: [] },
    }
  })),
  
  // Actions - Detection
  runDetection: (image) => set((state) => ({
    detectionResults: {
      ...state.detectionResults,
      inputImage: image,
      processingTime: Math.random() * 100 + 50, // Simulated processing time
    }
  })),
  
  updateDetectionResults: (results) => set((state) => ({
    detectionResults: { ...state.detectionResults, ...results }
  })),
  
  setConfidenceThreshold: (threshold) => set((state) => ({
    detectionResults: {
      ...state.detectionResults,
      confidenceThreshold: threshold
    }
  })),
  
  setNmsThreshold: (threshold) => set((state) => ({
    detectionResults: {
      ...state.detectionResults,
      nmsThreshold: threshold
    }
  })),
  
  // Actions - Metrics
  updateMetrics: (newMetrics) => set((state) => ({
    metrics: { ...state.metrics, ...newMetrics }
  })),
  
  resetMetrics: () => set(() => ({
    metrics: {
      f1Score: 0,
      precision: 0,
      recall: 0,
      map50: 0,
      map50_95: 0,
      inferenceTime: 0,
      totalImages: 0,
      totalDetections: 0,
    }
  })),
  
  // Simulate training step (for demo purposes)
  simulateTrainingStep: () => set((state) => {
    if (!state.trainingState.isTraining || state.trainingState.isPaused) {
      return state;
    }
    
    const newEpoch = state.trainingState.currentEpoch + 1;
    const totalEpochs = state.trainingState.totalEpochs;
    
    // Simulate loss decreasing
    const baseLoss = 2.5;
    const progress = newEpoch / totalEpochs;
    const loss = baseLoss * Math.exp(-3 * progress) + 0.1 + Math.random() * 0.05;
    
    // Simulate metrics improving
    const map50 = Math.min(0.95, 0.3 + progress * 0.65 + Math.random() * 0.05);
    const map50_95 = Math.min(0.75, 0.2 + progress * 0.55 + Math.random() * 0.05);
    const precision = Math.min(0.95, 0.4 + progress * 0.55 + Math.random() * 0.05);
    const recall = Math.min(0.95, 0.35 + progress * 0.6 + Math.random() * 0.05);
    
    return {
      trainingState: {
        ...state.trainingState,
        currentEpoch: newEpoch,
        loss: {
          boxLoss: loss * 0.6,
          clsLoss: loss * 0.3,
          dflLoss: loss * 0.1,
          totalLoss: loss,
        },
        history: {
          loss: [...state.trainingState.history.loss, loss].slice(-50),
          map50: [...state.trainingState.history.map50, map50].slice(-50),
          map50_95: [...state.trainingState.history.map50_95, map50_95].slice(-50),
          precision: [...state.trainingState.history.precision, precision].slice(-50),
          recall: [...state.trainingState.history.recall, recall].slice(-50),
        }
      }
    };
  }),
}));

// Selectors for optimal performance
export const selectDatasetConfig = (state) => state.datasetConfig;
export const selectModelConfig = (state) => state.modelConfig;
export const selectTrainingState = (state) => state.trainingState;
export const selectDetectionResults = (state) => state.detectionResults;
export const selectMetrics = (state) => state.metrics;

// Action selectors
export const selectUpdateDatasetConfig = (state) => state.updateDatasetConfig;
export const selectUpdateModelConfig = (state) => state.updateModelConfig;
export const selectStartTraining = (state) => state.startTraining;
export const selectStopTraining = (state) => state.stopTraining;
export const selectRunDetection = (state) => state.runDetection;
export const selectUpdateMetrics = (state) => state.updateMetrics;
export const selectResetMetrics = (state) => state.resetMetrics;
