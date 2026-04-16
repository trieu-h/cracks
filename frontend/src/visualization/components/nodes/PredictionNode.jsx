import { Handle, NodeResizer, Position, useReactFlow } from '@xyflow/react';
import { useEffect, useRef } from 'react';
import { resolveCollisions } from '../../utils/collisionDetection';
import './NodeStyles.css';

export default function PredictionNode({ data, selected }) {
  const { input = [], output = null, onUpdateInput, onPredict, isTrained = false, openConceptDialog } = data;
  const prevInputRef = useRef(input);
  const isPredictingRef = useRef(false);
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

  const handleInputChange = (e) => {
    onUpdateInput?.(e.target.value);
  };

  // Auto-predict when trained and input changes
  useEffect(() => {
    // Prevent concurrent predictions
    if (isPredictingRef.current) {
      return;
    }

    const inputChanged = JSON.stringify(prevInputRef.current) !== JSON.stringify(input);

    if (isTrained && onPredict && inputChanged) {
      prevInputRef.current = input;
      isPredictingRef.current = true;

      // Call predict and reset flag after
      try {
        onPredict();
      } finally {
        // Reset flag after a short delay to allow state updates
        setTimeout(() => {
          isPredictingRef.current = false;
        }, 100);
      }
    }
  }, [isTrained, input, onPredict, output]);

  return (
    <div className={`custom-node prediction-node ${!isTrained ? 'not-trained' : ''}`}>
      <NodeResizer
        color="#f59e0b"
        isVisible={selected}
        minWidth={200}
        minHeight={150}
        onResizeEnd={handleResizeEnd}
      />
      <Handle type="target" position={Position.Left} id="input" />
      <div
        className="node-header clickable-header"
        onClick={(e) => {
          if (openConceptDialog) {
            e.stopPropagation();
            openConceptDialog('prediction');
          }
        }}
        title="Click to learn about Prediction"
        style={{ cursor: openConceptDialog ? 'pointer' : 'default' }}
      >
        4. Prediction
      </div>
      <div className="node-content">
        {!isTrained && (
          <div className="warning-message">
            ⚠️ Model must be trained before prediction
          </div>
        )}
        <div className="data-section">
          <div className="section-title">Input x[]</div>
          <input
            type="text"
            value={Array.isArray(input) ? input.join(', ') : input}
            onChange={handleInputChange}
            onMouseDown={(e) => e.stopPropagation()}
            onPointerDown={(e) => e.stopPropagation()}
            className="data-input nodrag"
            placeholder="3"
            disabled={!isTrained}
          />
        </div>
        <div className="data-section">
          <div className="section-title">Predicted ŷ[]</div>
          <div className="output-display">
            {output ? (
              Array.isArray(output) ? output.map((val, idx) => (
                <span key={idx} className="output-value">{val.toFixed(4)}</span>
              )) : (
                <span className="output-value">{output.toFixed(4)}</span>
              )
            ) : (
              <span className="output-placeholder">
                {isTrained ? 'No prediction yet' : 'Train model first'}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
