import React, {useEffect, useState} from 'react';
import {Activity, AlertTriangle, ArrowLeftRight, Database, Gauge, GitBranch, LayoutDashboard, Menu, ShieldCheck, Shield, UserRoundSearch, Wifi, WifiOff, X} from 'lucide-react';
import {api} from './api.js';
import Overview from './pages/Overview.jsx';
import Network from './pages/Network.jsx';
import Inspector from './pages/Inspector.jsx';
import Results from './pages/Results.jsx';
import DataSplit from './pages/DataSplit.jsx';
import Limitations from './pages/Limitations.jsx';

const nav=[
 {id:'overview',label:'Overview',icon:LayoutDashboard,component:Overview},
 {id:'network',label:'Graph investigation',icon:GitBranch,component:Network},
 {id:'inspector',label:'Transaction / account',icon:UserRoundSearch,component:Inspector},
 {id:'results',label:'Model performance',icon:Gauge,component:Results},
 {id:'split',label:'Data split',icon:Database,component:DataSplit},
 {id:'limitations',label:'Limitations & safeguards',icon:ShieldCheck,component:Limitations}
];
export default function App(){
 const [page,setPage]=useState('overview'),[collapsed,setCollapsed]=useState(false),[health,setHealth]=useState({state:'checking',message:'Checking API…'});
 useEffect(()=>{let live=true;api.health().then(d=>{if(live)setHealth({state:'ok',message:d.message||'Backend connected'})}).catch(()=>{if(live)setHealth({state:'down',message:'API unavailable'})});return()=>{live=false}},[]);
 const active=nav.find(n=>n.id===page)||nav[0],Page=active.component;
 return <div className={`app-shell ${collapsed?'collapsed':''}`}>
  <aside className="sidebar">
   <div className="brand"><div className="brand-mark"><Shield size={21}/></div><div className="brand-copy"><b>PODS AML</b><small>Graph-based anti-money laundering</small></div><button className="icon-button mobile-close" onClick={()=>setCollapsed(true)} aria-label="Close menu"><X size={18}/></button></div>
   <div className="nav-label">WORKSPACE</div><nav>{nav.map(({id,label,icon:Icon})=><button key={id} className={`nav-item ${page===id?'active':''}`} onClick={()=>setPage(id)}><Icon size={18}/><span>{label}</span></button>)}</nav>
   <div className="sidebar-bottom"><div className={`connection ${health.state}`}><span className="status-dot"/><div><b>{health.state==='ok'?'Backend connected':health.state==='checking'?'Checking backend':'Backend unavailable'}</b><small>{health.message}</small></div></div><div className="sidebar-foot">PODS · AI & Data Science</div></div>
  </aside>
  <main className="main-area">
   <header className="topbar"><div className="topbar-left"><button className="icon-button menu-toggle" onClick={()=>setCollapsed(!collapsed)} aria-label="Toggle sidebar"><Menu size={20}/></button><div className="breadcrumbs"><span>PODS AML</span><b>/</b><strong>{active.label}</strong></div></div><div className="topbar-right"><span className={`api-chip ${health.state}`}><span className="status-dot"/>{health.state==='ok'?'API online':health.state==='checking'?'Connecting':'API offline'}</span><span className="model-chip">M1 / M2</span></div></header>
   <div className="content"><div className="page-title"><div><div className="eyebrow">ANTI-MONEY LAUNDERING ANALYTICS</div><h1>{active.label}</h1><p>{page==='overview'?'Monitor suspicious activity and model outputs from your dataset.':page==='network'?'Explore account relationships returned by the graph API.':page==='inspector'?'Inspect individual transaction or account records.':page==='results'?'Compare evaluated M1 and M2 model performance.':page==='split'?'Review the temporal train, validation and test split.':'Responsible interpretation and technical safeguards.'}</p></div></div><Page onNavigate={setPage}/><footer>Graph-based AML using XGBoost · Backend-driven values only</footer></div>
  </main>
 </div>;
}