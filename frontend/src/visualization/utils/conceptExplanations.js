export const conceptExplanations = {
  'neural-network': {
    title: 'Neural Network (Overview)',
    explanation: `
      <p>
        A neural network is like an <strong>assembly line for decisions</strong>.
        Data goes in, gets refined step by step, and a result comes out.
      </p>
      <p>
        Each step adjusts the data slightly to get closer to the right answer.
        Think of it as multiple reviewers improving a document, where each layer
        learns a slightly better version of the data.
      </p>
      <p>
        At its core, it's just: <code>output = input × weight + bias</code>
      </p>
    `,
    codeSnippet: `// Training data: input → output
const train = { 
  x: [[0], [1], [2], [3], [4]],
  y: [[0], [2], [4], [6], [8]],
};

let w = Math.random() * 10;
let learningRate = 1e-1;
let epsilon = 1e-3;

// Training loop: adjust weight to minimize loss
for (let step = 0; step < 50; step++) {
  // Finite difference (approximate derivative or gradient)
  let gradient = (loss(w + epsilon) - loss(w)) / epsilon; 
  w -= learningRate * gradient;
}
expect(5 * w).toBeCloseTo(10); // weight should approach 2`,
    fileReference: 'src/utils/neuralNetwork.js',
  },

  'hidden-layer': {
    title: 'Hidden Layers',
    explanation: `
    <p>
      Hidden layers are made of small units (neurons) that transform input data into new representations. Each layer combines inputs with learned importance (weights) and a small adjustment (bias), then passes the result forward until the final output is produced.
    </p>
    <p>
      It's like writing a program by refining logic step by step — first you handle the obvious cases, then add conditions, tweaks, and edge handling, until the final behavior works the way you want.
    </p>
    <div style="text-align: center; margin: 20px 0;">
      <img 
        src="matrix-multiplications.png" 
        alt="Matrix Multiplications"
        style="max-width: 100%; max-height: 60vh; display: block; margin: 0 auto;"
      />
    </div>
  `,
    fileReference: 'src/utils/neuralNetwork.js',
  },


  'activation': {
    title: 'Activation Functions',
    explanation: `
      <p>
        Activation functions decide <strong>what information matters</strong>.
        Without them, the network would behave too simply, just like a basic calculator.
      </p>
      <p>
        Think of it as a filter that blocks weak signals. The most common one is ReLU,
        which simply keeps positive values and removes negative ones: <code>ReLU(x) = max(0, x)</code>
      </p>
      <div style="text-align: center; margin: 20px 0;">
        <img 
          src="activation-functions.png" 
          alt="Activation Functions"
          style="max-width: 100%; max-height: 60vh; display: block; margin: 0 auto;"
        />
      </div>
    `,
    fileReference: 'src/utils/neuralNetwork.js',
  },

  'training-data': {
    title: 'Training Data',
    explanation: `
      <p>
        Training data is a list of <strong>examples with correct answers</strong>.
        The network learns by comparing its guess to the real answer, like practicing
        questions with answer keys.
      </p>
      <p>
        Each example shows what input should produce what output. The more examples
        you have, the better your network learns the pattern.
      </p>
    `,
    codeSnippet: `// Twice dataset (f(x) = 2x)
x: [[1], [2], [3], [4], [5], [6], [7], [8]]
y: [[2], [4], [6], [8], [10], [12], [14], [16]]`,
    fileReference: 'TrainingDataNode.jsx',
  },

  'cost': {
    title: 'Cost (Loss)',
    explanation: `
      <p>
        Cost tells us <strong>how wrong the prediction is</strong>. Lower cost means
        a better model, like measuring distance from a bullseye.
      </p>
      <p>
        During training, we try to make this number as small as possible. It's
        calculated by comparing predictions to actual answers: <code>error = prediction − answer</code>
      </p>
    `,
    codeSnippet: `// Mean Squared Error (simplified)
(predicted - actual) ** 2;`,
    fileReference: 'src/utils/neuralNetwork.js',
  },

  'learning-rate': {
    title: 'Learning Rate',
    explanation: `
      <p>
        Learning rate controls <strong>how big each correction is</strong> when the
        network learns from its mistakes.
      </p>
      <p>
        Too big and learning becomes unstable, like jerking a bike handle. Too small
        and learning is painfully slow. Finding the right balance is key.
      </p>
      <p>
        The formula is: <code>newWeight = oldWeight − learningRate × error</code>
      </p>
      <div style="text-align: center; margin: 20px 0;">
        <img 
          src="training-progress.png" 
          alt="Training Progress Visualization"
          style="max-width: 100%; max-height: 60vh; display: block; margin: 0 auto;"
        />
      </div>
    `,
    fileReference: 'src/utils/neuralNetwork.js',
  },

  'step': {
    title: 'Training Step',
    explanation: `
      <p>
        A training step is <strong>one full practice round</strong> using all your examples.
        After each step, the model gets slightly better at its task.
      </p>
      <p>
        Think of it as one full rehearsal before a performance. The more steps you run,
        the more refined your model becomes (up to a point).
      </p>
    `,
    codeSnippet: `for (let step = 0; step < steps; step++) {
  trainOnAllData();
}`,
    fileReference: 'src/utils/neuralNetwork.js',
  },

  'prediction': {
    title: 'Prediction (Inference)',
    explanation: `
      <p>
        Prediction is using the trained model to <strong>get answers for new data</strong>.
        No learning happens here just calculation, like taking the final exam after studying.
      </p>
      <p>
        You feed in new input, and the network runs through all its learned weights
        and biases to produce an output.
      </p>
    `,
    fileReference: 'src/utils/neuralNetwork.js',
  },

  'training-progress': {
    title: 'Training Progress',
    explanation: `
      <p>
        Training progress shows <strong>whether learning is working</strong> by tracking
        the loss over time.
      </p>
      <ul>
        <li>Loss going down → learning is working well</li>
        <li>Loss flat → model has finished learning or is stuck</li>
        <li>Loss going up → learning rate is too high</li>
      </ul>
      <p>
        It's like finding the right balance in life, pushing too hard leads to burnout, 
        pushing too little leads to stagnation. You want to apply just the right amount 
        of effort to see steady growth.
      </p>
      <div style="text-align: center; margin: 20px 0;">
        <img 
          src="training-progress.png" 
          alt="Training Progress Visualization"
          style="max-width: 100%; max-height: 60vh; display: block; margin: 0 auto;"
        />
      </div>
    `,
    fileReference: 'TrainingProgressNode.jsx',
  },

  'method': {
    title: 'Training Method',
    explanation: `
      <p>
        Training method is <strong>how the network calculates what to fix</strong>.
      </p>
      <div style="margin: 30px 0;">
        <p>
          <strong>Finite Difference</strong> is slower but simpler to understand. It's like
          trying every possible turn to see which one works best.
        </p>
        <div style="text-align: center; margin: 20px 0;">
          <img 
            src="finite-difference.png" 
            alt="Finite Difference Method"
            style="max-width: 80%; max-height: 50vh; display: block; margin: 0 auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);"
          />
        </div>
      </div>
      <div style="margin: 30px 0;">
        <p>
          <strong>Backpropagation</strong> is fast and accurate like using GPS navigation
          to find the best route. It's what everyone uses in practice.
        </p>
        <div style="text-align: center; margin: 20px 0; position: relative; display: flex; align-items: center; justify-content: center;">
          <button 
            id="backprop-prev" 
            onclick="var img0 = document.getElementById('backprop-img-0'); var img1 = document.getElementById('backprop-img-1'); if (img0.style.display === 'none') { img0.style.display = 'block'; img1.style.display = 'none'; }"
            style="width: 40px; height: 40px; border-radius: 50%; background: rgba(68, 68, 68, 0.8); color: #fff; border: 1px solid #666; cursor: pointer; display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0; padding: 0;"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="m15 18-6-6 6-6"/>
            </svg>
          </button>
          <div id="backprop-carousel" style="position: relative; flex: 1; max-width: 80%;">
            <img 
              id="backprop-img-0"
              src="derivative.png" 
              alt="Backpropagation Method - Derivative"
              style="max-width: 100%; max-height: 50vh; display: block; margin: 0 auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);"
            />
            <img 
              id="backprop-img-1"
              src="backpropagation.png" 
              alt="Backpropagation Method"
              style="max-width: 100%; max-height: 50vh; display: none; margin: 0 auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);"
            />
          </div>
          <button 
            id="backprop-next" 
            onclick="var img0 = document.getElementById('backprop-img-0'); var img1 = document.getElementById('backprop-img-1'); if (img1.style.display === 'none') { img0.style.display = 'none'; img1.style.display = 'block'; }"
            style="width: 40px; height: 40px; border-radius: 50%; background: rgba(68, 68, 68, 0.8); color: #fff; border: 1px solid #666; cursor: pointer; display: flex; align-items: center; justify-content: center; margin-left: 15px; flex-shrink: 0; padding: 0;"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="m9 18 6-6-6-6"/>
            </svg>
          </button>
        </div>
      </div>
    `,
    fileReference: 'src/utils/neuralNetwork.js',
  },

  'learning-progress': {
    title: 'Learning Progress',
    explanation: `
      <div style="text-align: center; margin: 20px 0;">
        <img 
          src="machine-learning-xkcd.png" 
          alt="Machine Learning - XKCD"
          style="max-width: 100%; max-height: 70vh; display: block; margin: 0 auto;"
        />
      </div>
    `,
    fileReference: 'LearningProgressNode.jsx',
  },
};
