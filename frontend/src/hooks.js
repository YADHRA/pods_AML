import { useCallback, useEffect, useRef, useState } from 'react';

/* Loads data safely: never an endless spinner (20 s timeout), cancels old requests, offers retry.
   const { data, loading, error, retry } = useAsync(signal => fetchJson('/api/...', signal), [deps]);   */
export function useAsync(fn, deps = [], timeoutMs = 20000) {
  const [state, setState] = useState({ data: null, loading: true, error: '' });
  const [tick, setTick] = useState(0);
  const keep = useRef(null);
  useEffect(() => {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort('timeout'), timeoutMs);
    setState(s => ({ data: s.data ?? keep.current, loading: true, error: '' }));
    Promise.resolve(fn(ctrl.signal))
      .then(data => { keep.current = data; setState({ data, loading: false, error: '' }); })
      .catch(e => {
        if (ctrl.signal.aborted && ctrl.signal.reason !== 'timeout') return;       // superseded request
        setState(s => ({ data: s.data, loading: false, error: ctrl.signal.reason === 'timeout' ? 'The backend took too long to answer (20 s). Is it running?' : String(e.message || e) }));
      })
      .finally(() => clearTimeout(timer));
    return () => { clearTimeout(timer); ctrl.abort(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  const retry = useCallback(() => setTick(t => t + 1), []);
  return { ...state, retry };
}

export async function fetchJson(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try { const j = await res.json(); if (typeof j.detail === 'string') msg = j.detail; } catch { /* keep */ }
    throw new Error(msg);
  }
  return res.json();
}

export const fmtInt = n => (n == null || Number.isNaN(+n) ? '—' : Number(n).toLocaleString('en-IN'));
export const fmtPct = (n, d = 2) => (n == null || Number.isNaN(+n) ? '—' : `${(+n * (+n <= 1 ? 100 : 1)).toFixed(d)}%`);   // accepts 0.1226 or 12.26
export const fmtNum = (n, d = 4) => (n == null || Number.isNaN(+n) ? '—' : (+n).toFixed(d));
export const inr = n => (n == null ? '—' : '₹' + Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 }));
export const riskLevel = s => (s == null ? null : s >= 0.7 ? 'High' : s >= 0.4 ? 'Medium' : 'Low');
