import React from 'react';

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  primary?: boolean;
  disabled?: boolean;
  className?: string;
  icon?: React.ReactNode;
  variant?: 'default' | 'primary' | 'ghost' | 'danger';
}

export const Button: React.FC<ButtonProps> = ({ 
  children, 
  onClick, 
  primary = false,
  disabled = false,
  className = '',
  icon,
  variant = 'default'
}) => {
  const baseClasses = 'btn-clean';
  
  const getVariantClasses = () => {
    if (primary || variant === 'primary') {
      return 'btn-primary';
    }
    if (variant === 'danger') {
      return 'bg-red-900/20 border-red-800 text-red-400 hover:bg-red-900/30 hover:border-red-700';
    }
    if (variant === 'ghost') {
      return 'bg-transparent border-transparent text-stone-400 hover:text-stone-200 hover:bg-stone-800/50';
    }
    return ''; // default btn-clean styling
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${getVariantClasses()} ${className}`}
    >
      {icon && <span className="flex-shrink-0">{icon}</span>}
      {children}
    </button>
  );
};
