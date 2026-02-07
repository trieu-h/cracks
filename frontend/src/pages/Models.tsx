import React, { useState, useEffect } from 'react';
import { Box, Download, Calendar, HardDrive } from 'lucide-react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { LED } from '../components/ui/LED';
import { getModels } from '../api';

const Models: React.FC = () => {
  const [models, setModels] = useState<any[]>([]);

  useEffect(() => {
    getModels().then(res => setModels(res.data));
  }, []);

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleDateString();
  };

  const formatSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  return (
    <div className="space-y-6">
      <Panel title="Trained Models">
        <div className="space-y-3">
          {models.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <Box size={48} className="mx-auto mb-4 opacity-30" />
              <p>No trained models yet</p>
              <p className="text-sm mt-2">Train a model to see it here</p>
            </div>
          ) : (
            models.map(model => (
              <div 
                key={model.id}
                className="flex items-center justify-between py-4 px-4 bg-gray-900/50 rounded-lg"
              >
                <div className="flex items-start gap-4">
                  <div className="mt-1">
                    <LED color="green" />
                  </div>
                  <div>
                    <div className="font-medium text-gray-200 flex items-center gap-2">
                      <Box size={16} className="text-vintage-orange" />
                      {model.name}
                    </div>
                    <div className="text-sm text-gray-500 font-mono mt-1">
                      ID: {model.id}
                    </div>
                    <div className="flex gap-4 mt-2 text-xs text-gray-400">
                      <span className="flex items-center gap-1">
                        <Calendar size={12} />
                        {formatDate(model.created)}
                      </span>
                      <span className="flex items-center gap-1">
                        <HardDrive size={12} />
                        {formatSize(model.size)}
                      </span>
                    </div>
                  </div>
                </div>
                
                <Button
                  onClick={() => {
                    // Trigger download
                    window.open(`/api/models/${model.id}/download`, '_blank');
                  }}
                >
                  <Download size={18} className="mr-2" />
                  Download
                </Button>
              </div>
            ))
          )}
        </div>
      </Panel>

      {/* Model Info */}
      <div className="grid grid-cols-3 gap-6">
        <Panel title="Total Models">
          <div className="text-center py-4">
            <div className="text-4xl font-mono text-scope-green">
              {models.length}
            </div>
            <div className="text-sm text-gray-500 mt-2">Trained models</div>
          </div>
        </Panel>

        <Panel title="Storage Used">
          <div className="text-center py-4">
            <div className="text-4xl font-mono text-vintage-orange">
              {formatSize(models.reduce((acc, m) => acc + m.size, 0))}
            </div>
            <div className="text-sm text-gray-500 mt-2">Total size</div>
          </div>
        </Panel>

        <Panel title="Latest Model">
          <div className="text-center py-4">
            <div className="text-lg font-mono text-scope-blue">
              {models.length > 0 
                ? formatDate(models[0].created)
                : '--'
              }
            </div>
            <div className="text-sm text-gray-500 mt-2">Created on</div>
          </div>
        </Panel>
      </div>
    </div>
  );
};

export default Models;
