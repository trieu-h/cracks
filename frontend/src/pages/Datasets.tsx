import React, { useState, useEffect } from 'react';
import { Folder, Trash2, RefreshCw, FileText } from 'lucide-react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { getDatasets, importDataset, deleteDataset } from '../api';

const Datasets: React.FC = () => {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [importPath, setImportPath] = useState('');
  const [loading, setLoading] = useState(false);
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());

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
    if (!confirm('Are you sure you want to delete this dataset?')) return;
    
    // Start delete animation
    setDeletingIds(prev => new Set(prev).add(id));
    
    // Wait for animation to complete
    setTimeout(async () => {
      try {
        await deleteDataset(id);
        loadDatasets();
      } catch (error) {
        alert('Error deleting dataset');
        setDeletingIds(prev => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      }
    }, 300);
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
        <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
          Dataset folder must contain YOLO format: images/, labels/, and data.yaml
        </p>
      </Panel>

      {/* Datasets List */}
      <Panel title="Available Datasets">
        <div className="space-y-0">
          {datasets.length === 0 ? (
            <div className="text-center py-12" style={{ color: 'var(--text-muted)' }}>
              <Folder size={48} className="mx-auto mb-4" style={{ opacity: 0.3 }} />
              <p>No datasets imported</p>
              <p className="text-sm mt-2" style={{ color: 'var(--text-disabled)' }}>Import a dataset to get started</p>
            </div>
          ) : (
            datasets.map((dataset, index) => (
              <div
                key={dataset.id}
                className={`flex items-center justify-between py-4 px-4 rounded-lg transition-all duration-300 ease-out ${
                  deletingIds.has(dataset.id)
                    ? 'opacity-0 scale-95 -translate-x-4'
                    : 'opacity-100 scale-100 translate-x-0'
                }`}
                style={{ 
                  background: 'var(--bg-secondary)',
                  borderBottom: index < datasets.length - 1 ? '1px solid var(--border-primary)' : 'none'
                }}
              >
                <div className="flex items-start gap-4">
                  <div>
                    <div className="font-medium flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
                      <FileText size={16} style={{ color: 'var(--warning-text)' }} />
                      {dataset.name}
                    </div>
                    <div className="text-sm font-mono mt-1" style={{ color: 'var(--text-muted)' }}>
                      {dataset.path}
                    </div>
                    <div className="flex gap-4 mt-2 text-xs">
                      <span style={{ color: 'var(--success-text)' }}>
                        Train: {dataset.train_images}
                      </span>
                      <span style={{ color: 'var(--accent-primary)' }}>
                        Val: {dataset.val_images}
                      </span>
                      <span style={{ color: 'var(--warning-text)' }}>
                        Test: {dataset.test_images}
                      </span>
                      <span style={{ color: 'var(--text-muted)' }}>
                        Classes: {dataset.num_classes}
                      </span>
                    </div>
                  </div>
                </div>
                
                <button
                  onClick={() => handleDelete(dataset.id)}
                  className="p-2 rounded-lg transition-colors"
                  style={{ 
                    color: 'var(--error-text)',
                    backgroundColor: 'transparent'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'var(--error-bg)';
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
