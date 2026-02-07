import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Play, 
  Image, 
  Database, 
  Box, 
  Settings,
  Layers
} from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
}

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/training', label: 'Training', icon: Play },
  { path: '/prediction', label: 'Prediction', icon: Image },
  { path: '/datasets', label: 'Datasets', icon: Database },
  { path: '/models', label: 'Models', icon: Box },
  { path: '/settings', label: 'Settings', icon: Settings },
];

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();

  return (
    <div className="min-h-screen flex bg-stone-950">
      {/* Sidebar */}
      <aside className="w-64 sidebar-clean flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-stone-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500 to-green-600 flex items-center justify-center">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-serif text-xl text-stone-100">
                Crack Lab
              </h1>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-clean ${isActive ? 'active' : ''}`}
              >
                <Icon size={18} strokeWidth={2} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-stone-800">
          <div className="text-xs text-stone-600">
            Version 1.0.0
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="header-clean px-8 py-4 flex items-center justify-between">
          <div>
            <h2 className="font-serif text-2xl text-stone-100">
              {navItems.find(item => item.path === location.pathname)?.label || 'Dashboard'}
            </h2>
          </div>
        </header>

        {/* Page Content */}
        <div className="flex-1 overflow-auto p-8">
          {children}
        </div>
      </main>
    </div>
  );
};
