import React from 'react';
import { X } from 'lucide-react';

interface PanelProps {
  children: React.ReactNode;
  title?: string;
  className?: string;
  icon?: React.ReactNode;
  onClose?: () => void;
}

export const Panel: React.FC<PanelProps> = ({ children, title, className = '', icon, onClose }) => (
  <div className={`card-clean ${className} relative`}>
    {title && (
      <div className="flex items-center justify-between mb-5 pb-4 border-b" 
           style={{ borderColor: 'var(--border-primary)' }}>
        <div className="flex items-center gap-3">
          {icon && (
            <div className="p-1.5 rounded-lg" style={{ backgroundColor: 'var(--bg-tertiary)' }}>
              {icon}
            </div>
          )}
          <h3 className="font-medium" style={{ color: 'var(--text-primary)' }}>{title}</h3>
        </div>
        {onClose && (
          <button 
            onClick={onClose}
            className="p-1 rounded-lg transition-colors"
            style={{ 
              color: 'var(--text-muted)',
              backgroundColor: 'transparent'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
              e.currentTarget.style.color = 'var(--text-secondary)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.color = 'var(--text-muted)';
            }}
          >
            <X size={16} />
          </button>
        )}
      </div>
    )}
    {children}
  </div>
);
