import { Handle, NodeResizer, Position, useReactFlow } from '@xyflow/react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { selectIsTrained, selectNetworkRef, selectParameters, selectTrainingData, useNeuralNetworkStore } from '../../stores/neuralNetworkStore';
import { resolveCollisions } from '../../utils/collisionDetection';
import './NodeStyles.css';

export default function LearningProgressNode({ data, selected }) {
    // Use Zustand selectors - only re-renders when these specific values change
    const trainingData = useNeuralNetworkStore(selectTrainingData);
    const isTrained = useNeuralNetworkStore(selectIsTrained);
    const networkRef = useNeuralNetworkStore(selectNetworkRef);
    const parameters = useNeuralNetworkStore(selectParameters); // Subscribe to parameter updates during training
    const { setNodes } = useReactFlow();
    const onHeaderClick = data?.onHeaderClick;

    const handleResizeEnd = () => {
        setNodes((nds) =>
            resolveCollisions(nds, {
                maxIterations: 50,
                overlapThreshold: 0.5,
                margin: 15,
            })
        );
    };

    // Compute predictions for training data to show actual vs predicted
    let predictedValues = [];
    // Show predictions when we have a network (trained or training)
    if (networkRef && trainingData.x && trainingData.x.length > 0) {
        predictedValues = trainingData.x.map(input => {
            const output = networkRef.predict?.(input);
            // Extract single value if array output
            return Array.isArray(output) && output.length === 1 ? output[0] : output;
        });
    }

    const trainingInputs = trainingData.x || [];
    const actualValues = (trainingData.y || []).map(y =>
        Array.isArray(y) && y.length === 1 ? y[0] : y
    );

    // Combine the data for charting
    const chartData = trainingInputs.map((input, idx) => ({
        index: idx,
        input: Array.isArray(input) ? input.join(', ') : input, // Format array as comma-separated string
        actual: actualValues[idx],
        predicted: predictedValues[idx] !== undefined ? predictedValues[idx] : null
    }));

    return (
        <div className="custom-node learning-progress-node">
            <NodeResizer
                color="#f59e0b"
                isVisible={selected}
                minWidth={380}
                minHeight={380}
                onResizeEnd={handleResizeEnd}
            />
            <Handle type="target" position={Position.Top} id="input" />
            <div
                className="node-header clickable-header"
                onClick={(e) => {
                    e.stopPropagation();
                    onHeaderClick?.();
                }}
                style={{ cursor: 'pointer' }}
            >
                3.b Learning Progress
            </div>
            <div className="node-content">
                <div className="chart-container">
                    {chartData.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                                <XAxis
                                    dataKey="input"
                                    stroke="#888"
                                    style={{ fontSize: '10px' }}
                                    label={{ value: 'Input', position: 'insideBottom', offset: -5, fill: '#888', fontSize: 10 }}
                                />
                                <YAxis
                                    stroke="#888"
                                    style={{ fontSize: '10px' }}
                                    label={{ value: 'Output', angle: -90, position: 'insideLeft', fill: '#888', fontSize: 10 }}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#2a2a2a',
                                        border: '1px solid #444',
                                        borderRadius: '4px',
                                        color: '#fff'
                                    }}
                                />
                                <Legend
                                    wrapperStyle={{ fontSize: '11px' }}
                                    iconType="line"
                                />
                                <Line
                                    type="monotone"
                                    dataKey="actual"
                                    stroke="#3b82f6"
                                    strokeWidth={2}
                                    dot={{ r: 3 }}
                                    name="Actual"
                                    isAnimationActive={false}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="predicted"
                                    stroke="#f59e0b"
                                    strokeWidth={2}
                                    dot={{ r: 3 }}
                                    name="Predicted"
                                    connectNulls
                                    isAnimationActive={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="no-data">No training data yet</div>
                    )}
                </div>
            </div>
        </div>
    );
}
