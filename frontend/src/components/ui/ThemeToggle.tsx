import React from 'react';
import { Sun, Moon } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

export const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className="btn-clean theme-toggle"
      aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
      title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
    >
      {theme === 'dark' ? (
        <>
          <Sun size={18} strokeWidth={2} />
          <span className="hidden sm:inline">Light</span>
        </>
      ) : (
        <>
          <Moon size={18} strokeWidth={2} />
          <span className="hidden sm:inline">Dark</span>
        </>
      )}
    </button>
  );
};
