import React from 'react';
import { ReactFlowProvider } from '@xyflow/react';
// @ts-ignore
import NnfsApp from '../visualization/App';
import '@xyflow/react/dist/style.css';
import '../visualization/App.css';
import '../visualization/components/nodes/NodeStyles.css';

const Visualization: React.FC = () => {
  return (
    <div className="visualization-page" style={{ height: 'calc(100vh - 100px)', width: '100%' }}>
      <ReactFlowProvider>
        <NnfsApp />
      </ReactFlowProvider>
    </div>
  );
};

export default Visualization;
