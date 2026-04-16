// Neural Network implementation with forward/backward pass and training

export class NeuralNetwork {
  constructor(layers, activation = 'relu', costFunction = 'mse') {
    this.layers = layers; // Array of layer sizes, e.g., [2, 4, 1]
    this.activation = activation;
    this.costFunction = costFunction;
    this.weights = [];
    this.biases = [];
    this.initialized = false;
    
    this.initialize();
  }

  initialize() {
    // Initialize weights and biases
    for (let i = 0; i < this.layers.length - 1; i++) {
      const rows = this.layers[i + 1];
      const cols = this.layers[i];
      
      // Xavier initialization
      const limit = Math.sqrt(6.0 / (rows + cols));
      this.weights[i] = Array(rows).fill(0).map(() =>
        Array(cols).fill(0).map(() => (Math.random() * 2 - 1) * limit)
      );
      
      this.biases[i] = Array(rows).fill(0).map(() => (Math.random() * 2 - 1) * 0.1);
    }
    this.initialized = true;
  }

  // Activation functions
  activate(x) {
    switch (this.activation) {
      case 'relu':
        return Math.max(0, x);
      case 'sigmoid':
      default:
        // Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
        return 1 / (1 + Math.exp(-x));
    }
  }

  activateDerivative(x) {
    switch (this.activation) {
      case 'relu':
        return x > 0 ? 1 : 0;
      case 'sigmoid':
      default:
        // Sigmoid derivative: σ'(x) = σ(x)(1 - σ(x))
        const s = 1 / (1 + Math.exp(-x));
        return s * (1 - s);
    }
  }

  // Cost functions (for a single output value)
  cost(y, yPred) {
    // Mean Squared Error: C = (y - ŷ)^2
    // Following tsoding/ml-notes approach
    return (y - yPred) ** 2;
  }

  // Cost for an entire sample (all outputs)
  sampleCost(y, yPred) {
    let totalCost = 0;
    for (let i = 0; i < y.length; i++) {
      totalCost += this.cost(y[i], yPred[i]);
    }
    return totalCost / y.length; // Return average cost across outputs
  }

  costDerivative(y, yPred) {
    // MSE derivative: ∂C/∂ŷ = 2(ŷ - y)
    // Following tsoding/ml-notes approach
    return 2 * (yPred - y);
  }

  // Forward pass
  forward(x) {
    if (!Array.isArray(x)) {
      x = [x];
    }
    
    let activations = [x];
    let zs = []; // Pre-activation values
    
    for (let i = 0; i < this.weights.length; i++) {
      const z = [];
      for (let j = 0; j < this.weights[i].length; j++) {
        let sum = this.biases[i][j];
        for (let k = 0; k < activations[i].length; k++) {
          sum += this.weights[i][j][k] * activations[i][k];
        }
        z.push(sum);
      }
      zs.push(z);
      
      // Use linear activation (no activation) for output layer, activation function for hidden layers
      const isOutputLayer = (i === this.weights.length - 1);
      activations.push(z.map(val => isOutputLayer ? val : this.activate(val)));
    }
    
    return { activations, zs, output: activations[activations.length - 1] };
  }

  // Backward pass (backpropagation)
  backward(x, y, yPred, activations, zs) {
    if (!Array.isArray(x)) {
      x = [x];
    }
    if (!Array.isArray(y)) {
      y = [y];
    }
    
    const gradients = {
      weights: this.weights.map(w => w.map(row => row.map(() => 0))),
      biases: this.biases.map(b => b.map(() => 0))
    };
    
    // Output layer error (use derivative of 1 for linear activation on output layer)
    let delta = [];
    for (let i = 0; i < yPred.length; i++) {
      const costDeriv = this.costDerivative(y[i] || y[0], yPred[i]);
      // Output layer uses linear activation, so derivative is 1
      delta.push(costDeriv * 1);
    }
    
    // Backpropagate through layers
    for (let layer = this.weights.length - 1; layer >= 0; layer--) {
      // Update gradients for this layer
      for (let i = 0; i < this.weights[layer].length; i++) {
        gradients.biases[layer][i] += delta[i];
        for (let j = 0; j < this.weights[layer][i].length; j++) {
          gradients.weights[layer][i][j] += delta[i] * activations[layer][j];
        }
      }
      
      // Calculate delta for previous layer
      if (layer > 0) {
        const newDelta = [];
        for (let j = 0; j < activations[layer].length; j++) {
          let sum = 0;
          for (let i = 0; i < delta.length; i++) {
            sum += delta[i] * this.weights[layer][i][j];
          }
          newDelta.push(sum * this.activateDerivative(zs[layer - 1][j]));
        }
        delta = newDelta;
      }
    }
    
    return gradients;
  }

  // Finite difference gradient approximation
  finiteDifferenceGradient(x, y, epsilon = 1e-5) {
    if (!Array.isArray(x)) {
      x = [x];
    }
    if (!Array.isArray(y)) {
      y = [y];
    }
    
    const gradients = {
      weights: this.weights.map(w => w.map(row => row.map(() => 0))),
      biases: this.biases.map(b => b.map(() => 0))
    };
    
    const { output: yPred } = this.forward(x);
    const baseCost = this.cost(y[0] || y[y.length - 1], yPred[0] || yPred[yPred.length - 1]);
    
    // Compute gradients for weights
    for (let layer = 0; layer < this.weights.length; layer++) {
      for (let i = 0; i < this.weights[layer].length; i++) {
        for (let j = 0; j < this.weights[layer][i].length; j++) {
          this.weights[layer][i][j] += epsilon;
          const { output: yPredPerturbed } = this.forward(x);
          const perturbedCost = this.cost(y[0] || y[y.length - 1], yPredPerturbed[0] || yPredPerturbed[yPredPerturbed.length - 1]);
          gradients.weights[layer][i][j] = (perturbedCost - baseCost) / epsilon;
          this.weights[layer][i][j] -= epsilon;
        }
      }
    }
    
    // Compute gradients for biases
    for (let layer = 0; layer < this.biases.length; layer++) {
      for (let i = 0; i < this.biases[layer].length; i++) {
        this.biases[layer][i] += epsilon;
        const { output: yPredPerturbed } = this.forward(x);
        const perturbedCost = this.cost(y[0] || y[y.length - 1], yPredPerturbed[0] || yPredPerturbed[yPredPerturbed.length - 1]);
        gradients.biases[layer][i] = (perturbedCost - baseCost) / epsilon;
        this.biases[layer][i] -= epsilon;
      }
    }
    
    return gradients;
  }

  // Update parameters
  updateParameters(gradients, learningRate) {
    for (let layer = 0; layer < this.weights.length; layer++) {
      for (let i = 0; i < this.weights[layer].length; i++) {
        this.biases[layer][i] -= learningRate * gradients.biases[layer][i];
        for (let j = 0; j < this.weights[layer][i].length; j++) {
          this.weights[layer][i][j] -= learningRate * gradients.weights[layer][i][j];
        }
      }
    }
  }

  // Training loop
  async train(x, y, steps, learningRate, method = 'backpropagation', onStep = null) {
    const history = [];
    
    for (let step = 0; step < steps; step++) {
      let totalLoss = 0;
      let gradients = {
        weights: this.weights.map(w => w.map(row => row.map(() => 0))),
        biases: this.biases.map(b => b.map(() => 0))
      };
      
      // Process all training samples
      for (let sample = 0; sample < x.length; sample++) {
        const xSample = Array.isArray(x[sample]) ? x[sample] : [x[sample]];
        const ySample = Array.isArray(y[sample]) ? y[sample] : [y[sample]];
        
        const { activations, zs, output: yPred } = this.forward(xSample);
        
        // Calculate loss for all outputs
        const loss = this.sampleCost(ySample, yPred);
        totalLoss += loss;
        
        // Compute gradients for this sample
        let sampleGradients;
        if (method === 'backpropagation') {
          sampleGradients = this.backward(xSample, ySample, yPred, activations, zs);
        } else if (method === 'finite-difference') {
          sampleGradients = this.finiteDifferenceGradient(xSample, ySample);
        } else {
          // Default to backpropagation if method is unrecognized
          sampleGradients = this.backward(xSample, ySample, yPred, activations, zs);
        }
        
        // Accumulate gradients
        for (let layer = 0; layer < gradients.weights.length; layer++) {
          for (let i = 0; i < gradients.weights[layer].length; i++) {
            gradients.biases[layer][i] += sampleGradients.biases[layer][i];
            for (let j = 0; j < gradients.weights[layer][i].length; j++) {
              gradients.weights[layer][i][j] += sampleGradients.weights[layer][i][j];
            }
          }
        }
      }
      
      // Average gradients over samples
      for (let layer = 0; layer < gradients.weights.length; layer++) {
        for (let i = 0; i < gradients.weights[layer].length; i++) {
          gradients.biases[layer][i] /= x.length;
          for (let j = 0; j < gradients.weights[layer][i].length; j++) {
            gradients.weights[layer][i][j] /= x.length;
          }
        }
      }
      
      // Update parameters
      this.updateParameters(gradients, learningRate);
      
      const avgLoss = totalLoss / x.length;
      history.push({ step, loss: avgLoss });
      
      if (onStep) {
        // Use requestAnimationFrame for smoother updates
        await new Promise(resolve => {
          requestAnimationFrame(() => {
            setTimeout(resolve, 0);
          });
        });
        await onStep(step, avgLoss, this.getParameters());
      }
    }
    
    return history;
  }

  // Get current parameters
  getParameters() {
    return {
      weights: JSON.parse(JSON.stringify(this.weights)),
      biases: JSON.parse(JSON.stringify(this.biases))
    };
  }

  // Set parameters
  setParameters(parameters) {
    this.weights = JSON.parse(JSON.stringify(parameters.weights));
    this.biases = JSON.parse(JSON.stringify(parameters.biases));
  }

  // Predict (forward pass only)
  predict(x) {
    const { output } = this.forward(x);
    return output;
  }
}
