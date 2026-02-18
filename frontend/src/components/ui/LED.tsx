import React from 'react';

interface LEDProps {
  color: 'green' | 'orange' | 'red' | 'off' | 'gray';
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  pulse?: boolean;
}

export const LED: React.FC<LEDProps> = ({ 
  color, 
  label,
  size = 'md',
  pulse = false
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

  const glowClasses = {
    green: 'shadow-[0_0_8px_rgba(34,197,94,0.6)]',
    orange: 'shadow-[0_0_8px_rgba(245,158,11,0.6)]',
    red: 'shadow-[0_0_8px_rgba(239,68,68,0.6)]',
    off: '',
    gray: ''
  };

  return (
    <div className="flex items-center gap-2">
      <div 
        className={`rounded-full ${sizeClasses[size]} ${colorClasses[color]} ${
          pulse ? `animate-pulse ${glowClasses[color]}` : ''
        }`} 
      />
      {label && <span className="text-xs text-ink-500 font-mono">{label}</span>}
    </div>
  );
};
