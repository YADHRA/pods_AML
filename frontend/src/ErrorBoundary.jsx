import React from 'react';

/* Catches render errors so a bad API response shows a message, not a white page. */
export default class ErrorBoundary extends React.Component {
  state = { error: null };
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error, info) { console.error('UI crashed:', error, info); }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div style={{margin:24,padding:20,border:'1px solid #FCA5A5',background:'#FDECEC',color:'#991B1B',borderRadius:12}}>
        <b>Something went wrong on this page.</b>
        <div style={{margin:'8px 0 12px',fontFamily:'ui-monospace,monospace',fontSize:13}}>{String(this.state.error.message||this.state.error)}</div>
        <button onClick={()=>this.setState({error:null})} style={{padding:'7px 14px',borderRadius:8,border:'1px solid #FCA5A5',background:'#fff',cursor:'pointer'}}>Try again</button>
      </div>
    );
  }
}
