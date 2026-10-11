import React, { useEffect, useState } from 'react';
import { Bell, Gauge, LayoutDashboard, Menu, ShieldCheck, Shield, UserRoundSearch, X } from 'lucide-react';
import { api } from './api.js';
import ErrorBoundary from './ErrorBoundary.jsx';
import Overview from './pages/Overview.jsx';
import Alerts from './pages/Alerts.jsx';
import Inspector from './pages/Inspector.jsx';
import Results from './pages/Results.jsx';

import Limitations from './pages/Limitations.jsx';

const nav = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, component: Overview, sub: 'Monitor suspicious activity, volumes and key statistics from the dataset.' },
  { id: 'alerts', label: 'Alerts', icon: Bell, component: Alerts, sub: 'Prioritised high-risk transactions and accounts requiring AML review.' },
  { id: 'inspector', label: 'Investigation', icon: UserRoundSearch, component: Inspector, sub: 'Detailed transaction risk factors, feature explanations and account profiles.' },
  { id: 'results', label: 'Model performance', icon: Gauge, component: Results, sub: 'Evaluate M1 and M2 detection capabilities, ROC/PR curves and trade-offs.' },

  { id: 'limitations', label: 'Limitations & safeguards', icon: ShieldCheck, component: Limitations, sub: 'Responsible interpretation, data quality indicators and technical safeguards.' },
];

export default function App() {
  const [page, setPage] = useState('overview');
  const [params, setParams] = useState({});
  const [collapsed, setCollapsed] = useState(false);
  const [health, setHealth] = useState({ state: 'checking', message: 'Checking API…' });

  useEffect(() => {
    let live = true;
    api.health()
      .then(d => live && setHealth({ state: 'ok', message: d.message || 'Backend connected' }))
      .catch(() => live && setHealth({ state: 'down', message: 'API unavailable' }));
    return () => { live = false; };
  }, []);

  const navigate = (newPage, newParams = {}) => {
    setParams(newParams);
    setPage(newPage);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const active = nav.find(n => n.id === page) || nav[0];
  const Page = active.component;

  return (
    <div className={`app-shell ${collapsed ? 'collapsed' : ''}`}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Shield size={21} /></div>
          <div className="brand-copy"><b>PODS AML</b><small>Anti-Money Laundering System</small></div>
          <button className="icon-button mobile-close" onClick={() => setCollapsed(true)} aria-label="Close menu"><X size={18} /></button>
        </div>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {nav.map(({ id, label, icon: Icon }) => (
            <button key={id} className={`nav-item ${active.id === id ? 'active' : ''}`} onClick={() => navigate(id)}>
              <Icon size={18} /><span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className={`connection ${health.state}`}>
            <span className="status-dot" />
            <div>
              <b>{health.state === 'ok' ? 'Backend connected' : health.state === 'checking' ? 'Checking backend' : 'Backend unavailable'}</b>
              <small>{health.message}</small>
            </div>
          </div>
          <div className="sidebar-foot">PODS · AI & Data Science</div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button menu-toggle" onClick={() => setCollapsed(!collapsed)} aria-label="Toggle sidebar"><Menu size={20} /></button>
            <div className="breadcrumbs"><span>PODS AML</span><b>/</b><strong>{active.label}</strong></div>
          </div>
          <div className="topbar-right">
            <span className={`api-chip ${health.state}`}><span className="status-dot" />{health.state === 'ok' ? 'API online' : health.state === 'checking' ? 'Connecting' : 'API offline'}</span>
          </div>
        </header>
        <div className="content">
          <div className="page-title"><div>
            <div className="eyebrow">ANTI-MONEY LAUNDERING ANALYTICS</div>
            <h1>{active.label}</h1><p>{active.sub}</p>
          </div></div>
          <ErrorBoundary key={active.id}><Page onNavigate={navigate} params={params} /></ErrorBoundary>
          <footer>PODS AML · XGBoost Detection · Real Backend Data Only</footer>
        </div>
      </main>
    </div>
  );
}
