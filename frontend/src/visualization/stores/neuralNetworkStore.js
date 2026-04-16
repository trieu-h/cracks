import { create } from 'zustand';

/**
 * Production-grade Zustand store for neural network state
 * Prevents unnecessary re-renders through selective subscriptions
 * Following React Flow team's recommended pattern
 */
export const useNeuralNetworkStore = create((set, get) => ({
  // Training state
  isTraining: false,
  abortTraining: false,
  isTrained: false,
  currentStep: 0,
  trainingHistory: [],
  
  // Network configuration
  config: {
    hiddenLayers: [1],  // Only hidden layers, input/output derived from data
    activation: 'relu',
    costFunction: 'mse',
  },
  
  // Training configuration
  trainingConfig: {
    steps: 30,
    learningRate: 0.01,
    method: 'backpropagation',
  },
  
  // Training data
  trainingData: {
    x: [[1], [2], [3], [4], [5], [6], [7], [8]],
    y: [[2], [4], [6], [8], [10], [12], [14], [16]],
  },
  
  // Prediction state
  predictionInput: [10],
  predictionOutput: null,
  
  // Network parameters (weights, biases)
  parameters: null,
  
  // Network reference (stored separately to avoid serialization)
  networkRef: null,
  
  // Actions
  setIsTraining: (isTraining) => set({ isTraining }),
  setAbortTraining: (abortTraining) => set({ abortTraining }),
  setIsTrained: (isTrained) => set({ isTrained }),
  setCurrentStep: (currentStep) => set({ currentStep }),
  
  addTrainingHistory: (step, loss) => set((state) => ({
    trainingHistory: [...state.trainingHistory.slice(-99), { step, loss }],
  })),
  
  resetTrainingHistory: () => set({ trainingHistory: [] }),
  
  updateConfig: (config) => set((state) => ({
    config: { ...state.config, ...config },
    isTrained: false,
    trainingHistory: [],
  })),
  
  updateTrainingConfig: (trainingConfig) => set({ trainingConfig }),
  
  updateTrainingData: (trainingData) => set({ trainingData }),
  
  updatePredictionInput: (predictionInput) => set({ predictionInput }),
  
  updatePredictionOutput: (predictionOutput) => set({ predictionOutput }),
  
  updateParameters: (parameters) => set({ parameters }),
  
  setNetworkRef: (networkRef) => set({ networkRef }),
}));

// Selectors for optimal performance - only re-render when specific data changes
export const selectIsTraining = (state) => state.isTraining;
export const selectAbortTraining = (state) => state.abortTraining;
export const selectIsTrained = (state) => state.isTrained;
export const selectCurrentStep = (state) => state.currentStep;
export const selectTrainingHistory = (state) => state.trainingHistory;
export const selectConfig = (state) => state.config;
export const selectTrainingConfig = (state) => state.trainingConfig;
export const selectTrainingData = (state) => state.trainingData;
export const selectPredictionInput = (state) => state.predictionInput;
export const selectPredictionOutput = (state) => state.predictionOutput;
export const selectParameters = (state) => state.parameters;
export const selectNetworkRef = (state) => state.networkRef;

// Action selectors
export const selectSetIsTraining = (state) => state.setIsTraining;
export const selectSetAbortTraining = (state) => state.setAbortTraining;
export const selectSetIsTrained = (state) => state.setIsTrained;
export const selectSetCurrentStep = (state) => state.setCurrentStep;
export const selectAddTrainingHistory = (state) => state.addTrainingHistory;
export const selectResetTrainingHistory = (state) => state.resetTrainingHistory;
export const selectUpdateConfig = (state) => state.updateConfig;
export const selectUpdateTrainingConfig = (state) => state.updateTrainingConfig;
export const selectUpdateTrainingData = (state) => state.updateTrainingData;
export const selectUpdatePredictionInput = (state) => state.updatePredictionInput;
export const selectUpdatePredictionOutput = (state) => state.updatePredictionOutput;
export const selectUpdateParameters = (state) => state.updateParameters;
export const selectSetNetworkRef = (state) => state.setNetworkRef;
