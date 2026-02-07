import React, { useState, useEffect } from 'react';
import { Folder, Trash2, RefreshCw, FileText } from 'lucide-react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { LED } from '../components/ui/LED';
import { getDatasets, importDataset, deleteDataset } from '../api';

const Datasets: React.FC = () => {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [importPath, setImportPath] = useState('');
  const [loading, setLoading] = useState(false);

  const loadDatasets = () => {
    getDatasets().then(res => setDatasets(res.data));
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  const handleImport = async () => {
    if (!importPath) return;
    setLoading(true);
    await importDataset(importPath);
    setImportPath('');
    loadDatasets();
    setLoading(false);
  };

  const handleDelete = async (id: string) => {
    await deleteDataset(id);
    loadDatasets();
  };

  return (
    <div className="space-y-6">
      {/* Import Panel */}
      <Panel title="Import Dataset">
        <div className="flex gap-4">
          <input
            type="text"
            value={importPath}
            onChange={(e) => setImportPath(e.target.value)}
            placeholder="/path/to/dataset/folder"
            className="input-clean flex-1"
          />
          <Button 
            primary 
            onClick={handleImport}
            disabled={loading || !importPath}
          >
            {loading ? (
              <RefreshCw size={18} className="inline mr-2 animate-spin" />
            ) : (
              <Folder size={18} className="inline mr-2" />
            )}
            Import
          </Button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Dataset folder must contain YOLO format: images/, labels/, and data.yaml
        </p>
      </Panel>

      {/* Datasets List */}
      <Panel title="Available Datasets">
        <div className="space-y-3">
          {datasets.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <Folder size={48} className="mx-auto mb-4 opacity-30" />
              <p>No datasets imported</p>
              <p className="text-sm mt-2">Import a dataset to get started</p>
            </div>
          ) : (
            datasets.map(dataset => (
              <div 
                key={dataset.id}
                className="flex items-center justify-between py-4 px-4 bg-gray-900/50 rounded-lg"
              >
                <div className="flex items-start gap-4">
                  <div className="mt-1">
                    <LED color="green" />
                  </div>
                  <div>
                    <div className="font-medium text-gray-200 flex items-center gap-2">
                      <FileText size={16} className="text-vintage-orange" />
                      {dataset.name}
                    </div>
                    <div className="text-sm text-gray-500 font-mono mt-1">
                      {dataset.path}
                    </div>
                    <div className="flex gap-4 mt-2 text-xs">
                      <span className="text-scope-green">
                        Train: {dataset.train_images}
                      </span>
                      <span className="text-scope-blue">
                        Val: {dataset.val_images}
                      </span>
                      <span className="text-vintage-yellow">
                        Test: {dataset.test_images}
                      </span>
                      <span className="text-gray-400">
                        Classes: {dataset.num_classes}
                      </span>
                    </div>
                  </div>
                </div>
                
                <button
                  onClick={() => handleDelete(dataset.id)}
                  className="p-2 rounded-lg transition-colors"
                  style={{ 
                    color: 'var(--error)',
                    backgroundColor: 'transparent'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'rgba(239, 68, 68, 0.1)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                  title="Delete dataset"
                >
                  <Trash2 size={18} />
                </button>
              </div>
            ))
          )}
        </div>
      </Panel>
    </div>
  );
};

export default Datasets;
