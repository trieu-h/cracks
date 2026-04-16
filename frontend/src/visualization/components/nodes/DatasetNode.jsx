import { useState, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Upload, Image, Folder, Database, FileText, Info, Trash2 } from 'lucide-react';
import './NodeStyles.css';

const DatasetNode = ({ data }) => {
  const [uploadedImages, setUploadedImages] = useState([]);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    const newImages = files.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      preview: URL.createObjectURL(file),
      size: file.size
    }));
    
    setUploadedImages(prev => [...prev, ...newImages].slice(0, 6));
  }, []);

  const handleFileInput = useCallback((e) => {
    const files = Array.from(e.target.files || []).filter(f => f.type.startsWith('image/'));
    const newImages = files.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      preview: URL.createObjectURL(file),
      size: file.size
    }));
    
    setUploadedImages(prev => [...prev, ...newImages].slice(0, 6));
  }, []);

  const removeImage = useCallback((id) => {
    setUploadedImages(prev => prev.filter(img => img.id !== id));
  }, []);

  return (
    <div className={`node-card dataset-node ${isDragging ? 'dragging' : ''}`}>
      <Handle type="target" position={Position.Left} className="node-handle" />
      
      <div className="node-header">
        <Database className="node-icon" size={18} />
        <span className="node-title">Dataset</span>
        <Info 
          className="info-trigger" 
          size={14} 
          onClick={(e) => {
            e.stopPropagation();
            data.openConceptDialog?.('dataset');
          }}
          style={{ cursor: 'pointer' }}
        />
      </div>

      <div className="node-content">
        {/* Upload Zone */}
        <div 
          className="upload-zone"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <Upload size={24} className="upload-icon" />
          <p className="upload-text">Drop crack images here</p>
          <p className="upload-hint">or click to browse</p>
          <input 
            type="file" 
            accept="image/*" 
            multiple 
            onChange={handleFileInput}
            className="file-input"
          />
        </div>

        {/* Preview Grid */}
        {uploadedImages.length > 0 && (
          <div className="preview-grid">
            {uploadedImages.map(img => (
              <div key={img.id} className="preview-item">
                <img src={img.preview} alt={img.name} />
                <button 
                  className="remove-btn"
                  onClick={() => removeImage(img.id)}
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Stats */}
        <div className="dataset-stats">
          <div className="stat-item">
            <Image size={14} />
            <span>{uploadedImages.length} images</span>
          </div>
          <div className="stat-item">
            <FileText size={14} />
            <span>1 class: crack</span>
          </div>
        </div>

        {/* YAML Preview */}
        <div className="yaml-preview">
          <pre>train: images/train
val: images/val
nc: 1
names: ['crack']</pre>
        </div>
      </div>

      <Handle type="source" position={Position.Right} className="node-handle" />
    </div>
  );
};

export default DatasetNode;
