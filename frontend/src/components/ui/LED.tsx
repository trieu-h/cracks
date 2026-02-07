import React from 'react';

interface LEDProps {
  color: 'green' | 'orange' | 'red' | 'off' | 'gray';
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const LED: React.FC<LEDProps> = ({ 
  color, 
  label,
  size = 'md'
}) => {
  const sizeClasses = {
    sm: 'w-1.5 h-1.5',
    md: 'w-2 h-2',
    lg: 'w-2.5 h-2.5'
  };

  const colorClasses = {
    green: 'bg-primary-500',
    orange: 'bg-amber-500',
    red: 'bg-red-500',
    off: 'bg-paper-300',
    gray: 'bg-ink-400'
  };

  return (
    <div className="flex items-center gap-2">
      <div 
        className={`rounded-full ${sizeClasses[size]} ${colorClasses[color]}`} 
      />
      {label && <span className="text-xs text-ink-500 font-mono">{label}</span>}
    </div>
  );
};
