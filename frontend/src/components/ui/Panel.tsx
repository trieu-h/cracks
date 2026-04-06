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
      <div className="flex items-center justify-between mb-5 pb-4 border-b border-paper-200">
        <div className="flex items-center gap-3">
          {icon && (
            <div className="p-1.5 rounded-lg bg-paper-100">
              {icon}
            </div>
          )}
          <h3 className="font-medium text-ink-900">{title}</h3>
        </div>
        {onClose && (
          <button 
            onClick={onClose}
            className="p-1 hover:bg-stone-800 rounded-lg text-stone-500 hover:text-stone-300 transition-colors"
          >
            <X size={16} />
          </button>
        )}
      </div>
    )}
    {children}
  </div>
);
