import React, {useEffect, useState} from 'react';
import {api} from '../api.js';
import {Status, EmptyState, Loading} from '../components/Status.jsx';
export default function DataSplit(){
 const [data,setData]=useState(null),[loading,setLoading]=useState(true),[error,setError]=useState('');
 useEffect(()=>{let live=true;api.dataSplit().then(d=>{if(live)setData(d)}).catch(e=>{if(live)setError(e.message)}).finally(()=>{if(live)setLoading(false)});return()=>{live=false}},[]);
 if(loading)return <Loading label="Loading data split details…"/>;
 return <div className="page-stack">
 {error&&<Status kind="warning">Data-split endpoint unavailable: {error}. Expected GET /api/data-split.</Status>}
 {data ? <section className="card"><h3>Dataset split returned by backend</h3><pre className="json">{JSON.stringify(data,null,2)}</pre></section> : <EmptyState title="Data split information not connected" detail="Expose the project's actual train, validation and test date windows, row counts and positive-label counts from GET /api/data-split. Do not estimate these from frontend code."/>}
 </div>;
}