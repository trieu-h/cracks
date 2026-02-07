import React from 'react';

interface PanelProps {
  children: React.ReactNode;
  title?: string;
  className?: string;
  icon?: React.ReactNode;
}

export const Panel: React.FC<PanelProps> = ({ children, title, className = '', icon }) => (
  <div className={`card-clean ${className}`}>
    {title && (
      <div className="flex items-center gap-3 mb-5 pb-4 border-b border-paper-200">
        {icon && (
          <div className="p-1.5 rounded-lg bg-paper-100">
            {icon}
          </div>
        )}
        <h3 className="font-medium text-ink-900">{title}</h3>
      </div>
    )}
    {children}
  </div>
);
