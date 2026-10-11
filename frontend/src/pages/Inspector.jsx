import React, {useState} from 'react';
import {Search, ArrowLeftRight, UserRoundSearch} from 'lucide-react';
import {api} from '../api.js';
import {Status, EmptyState, Loading, RiskPill} from '../components/Status.jsx';

function ResultCard({title,data}) {
 if(!data)return null;
 return <section className="card"><h3>{title}</h3><dl className="detail-list">{Object.entries(data).map(([k,v])=><React.Fragment key={k}><dt>{k.replaceAll('_',' ')}</dt><dd>{v==null?'—':typeof v==='object'?JSON.stringify(v):String(v)}</dd></React.Fragment>)}</dl></section>;
}
export default function Inspector(){
 const [kind,setKind]=useState('transactions'),[id,setId]=useState(''),[data,setData]=useState(null),[loading,setLoading]=useState(false),[error,setError]=useState('');
 async function search(e){e.preventDefault();if(!id.trim())return;setLoading(true);setError('');setData(null);try{setData(kind==='transactions'?await api.transaction(id.trim()):await api.account(id.trim()))}catch(err){setError(err.message)}finally{setLoading(false)}}
 return <div className="page-stack">
  <section className="card"><div className="section-heading"><div><h3>Investigate a transaction or account</h3><p>Search IDs from your processed dataset or prediction output.</p></div></div>
   <form className="search-form" onSubmit={search}><div className="segmented"><button type="button" className={kind==='transactions'?'selected':''} onClick={()=>{setKind('transactions');setData(null);setError('')}}><ArrowLeftRight size={15}/> Transaction</button><button type="button" className={kind==='accounts'?'selected':''} onClick={()=>{setKind('accounts');setData(null);setError('')}}><UserRoundSearch size={15}/> Account</button></div><input value={id} onChange={e=>setId(e.target.value)} placeholder={kind==='transactions'?'Enter transaction ID':'Enter account/node ID'} aria-label="Search identifier"/><button className="button primary" type="submit"><Search size={15}/> Search</button></form>
  </section>
  {loading&&<Loading/>}
  {error&&<Status kind="warning">Lookup failed: {error}. Required backend endpoint: {kind==='transactions'?'/api/transactions/{id}':'/api/accounts/{id}'}.</Status>}
  {data&&<ResultCard title="Backend record" data={data}/>}
  {!data&&!loading&&!error&&<EmptyState title="Ready to investigate" detail="Enter an ID to request a real record from FastAPI. No placeholder account or transaction is assumed."/>}
 </div>;
}