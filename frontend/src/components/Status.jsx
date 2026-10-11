import React from 'react';
import { CheckCircle2, AlertCircle, LoaderCircle, WifiOff } from 'lucide-react';
export function Status({kind='info', children}) {
  const Icon = kind === 'success' ? CheckCircle2 : kind === 'error' ? AlertCircle : kind === 'loading' ? LoaderCircle : WifiOff;
  return <div className={`notice ${kind}`}><Icon size={17}/><span>{children}</span></div>;
}
export function EmptyState({title='No data available', detail='This endpoint has not returned data yet.'}) {
  return <div className="empty"><div className="empty-icon"><WifiOff size={23}/></div><h3>{title}</h3><p>{detail}</p></div>;
}
export function Metric({label,value,sub,icon:Icon}) {
  return <article className="metric card"><div className="metric-top"><span>{label}</span>{Icon && <Icon size={17}/>}</div><div className="metric-value">{value ?? '—'}</div>{sub && <div className="metric-sub">{sub}</div>}</article>;
}
export function RiskPill({risk}) {
  const value = String(risk ?? 'Unknown');
  const key = value.toLowerCase();
  const cls = key.includes('high') || key.includes('suspicious') ? 'high' : key.includes('medium') ? 'medium' : key.includes('low') || key.includes('normal') ? 'low' : 'unknown';
  return <span className={`risk-pill ${cls}`}>{value}</span>;
}
export function Loading({label='Loading data…'}) { return <div className="loading"><LoaderCircle className="spin" size={19}/>{label}</div>; }