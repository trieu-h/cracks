import { useCallback, useEffect, useRef } from 'react';
import {
  selectAddTrainingHistory,
  selectConfig,
  selectResetTrainingHistory,
  selectSetCurrentStep,
  selectSetIsTrained,
  selectSetIsTraining,
  selectSetAbortTraining,
  selectAbortTraining,
  selectSetNetworkRef,
  selectTrainingConfig,
  selectUpdateConfig,
  selectUpdateParameters,
  selectUpdateTrainingConfig,
  useNeuralNetworkStore,
} from '../stores/neuralNetworkStore';
import { NeuralNetwork } from '../utils/neuralNetwork';

export function useNeuralNetwork() {
  // Use Zustand selectors - only subscribes to what we need
  const config = useNeuralNetworkStore(selectConfig);
  const trainingConfig = useNeuralNetworkStore(selectTrainingConfig);
  const setIsTraining = useNeuralNetworkStore(selectSetIsTraining);
  const setAbortTraining = useNeuralNetworkStore(selectSetAbortTraining);
  const abortTraining = useNeuralNetworkStore(selectAbortTraining);
  const setIsTrained = useNeuralNetworkStore(selectSetIsTrained);
  const setCurrentStep = useNeuralNetworkStore(selectSetCurrentStep);
  const addTrainingHistory = useNeuralNetworkStore(selectAddTrainingHistory);
  const resetTrainingHistory = useNeuralNetworkStore(selectResetTrainingHistory);
  const updateConfig = useNeuralNetworkStore(selectUpdateConfig);
  const updateTrainingConfig = useNeuralNetworkStore(selectUpdateTrainingConfig);
  const updateParameters = useNeuralNetworkStore(selectUpdateParameters);
  const setNetworkRef = useNeuralNetworkStore(selectSetNetworkRef);
  
  const networkRef = useRef(null);

  // Train the network
  const train = useCallback(async (x, y, onStepCallback, fullLayers) => {
    console.log('[train] Training started, will reinitialize network');
    console.log('[train] Full layers architecture:', fullLayers);
    // Always reinitialize network to reset weights to random values
    const nn = new NeuralNetwork(
      fullLayers,
      config.activation,
      config.costFunction
    );
    networkRef.current = nn;
    setNetworkRef(nn);
    setIsTrained(false);
    
    setIsTraining(true);
    setAbortTraining(false); // Reset abort flag
    setCurrentStep(0);
    resetTrainingHistory();

    // Use requestAnimationFrame to batch updates and prevent infinite loops
    let lastUpdateTime = 0;
    const updateInterval = 100; // Update UI every 100ms for smoother performance

    try {
      const history = await networkRef.current.train(
        x,
        y,
        trainingConfig.steps,
        trainingConfig.learningRate,
        trainingConfig.method,
        async (step, loss, parameters) => {
          // Check if training was aborted
          if (useNeuralNetworkStore.getState().abortTraining) {
            throw new Error('Training cancelled by user');
          }
          
          const now = Date.now();
          // Throttle UI updates to prevent excessive re-renders
          if (now - lastUpdateTime >= updateInterval || step === 0 || step === trainingConfig.steps - 1) {
            setCurrentStep(step);
            addTrainingHistory(step, loss);
            updateParameters(parameters);
            lastUpdateTime = now;
            
            if (onStepCallback) {
              // Use setTimeout to defer callback and prevent blocking
              await new Promise(resolve => setTimeout(resolve, 0));
              await onStepCallback(step, loss, parameters);
            }
          }
        }
      );

      // Final update
      setCurrentStep(trainingConfig.steps - 1);
      setIsTrained(true);
      updateParameters(networkRef.current.getParameters());
      return history;
    } catch (error) {
      if (error.message === 'Training cancelled by user') {
        console.log('Training was cancelled');
        return null;
      }
      console.error('Training error:', error);
      throw error;
    } finally {
      setIsTraining(false);
      setAbortTraining(false); // Reset abort flag
    }
  }, [trainingConfig.steps, trainingConfig.learningRate, trainingConfig.method, 
    config.activation, config.costFunction, setIsTraining, setAbortTraining, setCurrentStep, resetTrainingHistory, 
    addTrainingHistory, updateParameters, setIsTrained, setNetworkRef]);

  // Cancel training
  const cancelTraining = useCallback(() => {
    setAbortTraining(true);
  }, [setAbortTraining]);

  // Predict using trained network
  const predict = useCallback((x) => {
    if (!networkRef.current) {
      return null;
    }
    return networkRef.current.predict(x);
  }, []);

  // Get current parameters
  const getParameters = useCallback(() => {
    if (!networkRef.current) {
      return null;
    }
    return networkRef.current.getParameters();
  }, []);

  return {
    train,
    cancelTraining,
    predict,
    getParameters,
    updateConfig,
    updateTrainingConfig,
  };
}
