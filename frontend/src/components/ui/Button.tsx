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
  const baseClasses = 'inline-flex items-center gap-2 px-4 py-2 rounded-xl font-medium text-sm transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed';
  
  const variantClasses = {
    default: 'bg-white border border-paper-200 text-ink-700 hover:bg-paper-50 hover:border-paper-300',
    primary: 'bg-primary-500 border border-primary-500 text-white hover:bg-primary-600 hover:border-primary-600',
    ghost: 'text-ink-500 hover:text-ink-700 hover:bg-paper-50',
    danger: 'bg-red-50 border border-red-200 text-red-600 hover:bg-red-100'
  };

  const resolvedVariant = primary ? 'primary' : variant;

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${variantClasses[resolvedVariant]} ${className}`}
    >
      {icon && <span className="flex-shrink-0">{icon}</span>}
      {children}
    </button>
  );
};
