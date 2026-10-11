import React from 'react';
import '../theme.css';

export const Card = ({ title, tag, right, children, className = '', style }) => (
  <section className={`u-card ${className}`} style={style}>
    {(title || tag || right) && <h3>{title}{tag && <span className="u-tag">{tag}</span>}{right && <span className="u-right">{right}</span>}</h3>}
    {children}
  </section>
);
export const Kpi = ({ label, value, sub, tone = '' }) => (
  <div className={`u-card u-kpi ${tone}`}><small>{label}</small><b>{value}</b>{sub && <span>{sub}</span>}</div>
);
export const RiskBadge = ({ level }) => <span className={`u-badge ${level || 'None'}`}><i />{level || 'Not scored'}</span>;
export const Badge = ({ children, tone = 'n' }) => <span className={`u-badge ${tone}`}>{children}</span>;
export const Banner = ({ children, tone = '' }) => <div className={`u-banner ${tone}`}>{children}</div>;

export const Segmented = ({ value, options, onChange }) => (
  <div className="u-seg" role="tablist">{options.map(o => (
    <button key={o.value} className={value === o.value ? 'on' : ''} onClick={() => onChange(o.value)} role="tab" aria-selected={value === o.value}>{o.label}</button>
  ))}</div>
);

export const KeyValue = ({ rows }) => (
  <dl className="u-kv">{rows.filter(r => r[1] !== undefined).map(([k, v]) => <React.Fragment key={k}><dt>{k}</dt><dd>{v ?? '—'}</dd></React.Fragment>)}</dl>
);

export const BarList = ({ items, format = v => v, color }) => {
  const m = Math.max(...items.map(i => Math.abs(i.value)), 1e-9);
  return (
    <div className="u-bars">{items.map(i => (
      <div className="row" key={i.label}><span className="n">{i.label}</span>
        <span className="t"><i style={{ width: `${(Math.abs(i.value) / m) * 100}%`, background: i.color || color }} /></span>
        <span className="v">{format(i.value)}</span></div>
    ))}</div>
  );
};

export const Loading = ({ rows = 5 }) => <div>{Array.from({ length: rows }, (_, i) => <div key={i} className="u-skel" style={{ width: `${100 - i * 7}%` }} />)}</div>;
export const ErrorState = ({ error, onRetry }) => (
  <div className="u-banner err" style={{ alignItems: 'center' }}><span style={{ flex: 1 }}>{error}</span>{onRetry && <button className="u-btn s" onClick={onRetry}>Retry</button>}</div>
);
export const Empty = ({ title = 'Nothing to show', children }) => <div className="u-empty"><h4>{title}</h4><div>{children}</div></div>;

/* Table. columns: [{key,label,right,render(row)}]  */
export function DataTable({ columns, rows, onRowClick, sort, onSort, empty }) {
  if (!rows.length) return empty || <Empty title="No rows" />;
  return (
    <div className="u-scroll"><table className="u-table">
      <thead><tr>{columns.map(c => (
        <th key={c.key} className={`${c.right ? 'r' : ''} ${c.sortable ? 'sort' : ''}`} onClick={() => c.sortable && onSort?.(c.key)}>
          {c.label}{sort?.key === c.key ? (sort.dir < 0 ? ' ↓' : ' ↑') : ''}</th>))}</tr></thead>
      <tbody>{rows.map((r, i) => (
        <tr key={r.id ?? i} className={onRowClick ? 'click' : ''} onClick={() => onRowClick?.(r)}>
          {columns.map(c => <td key={c.key} className={c.right ? 'r' : ''}>{c.render ? c.render(r) : r[c.key]}</td>)}</tr>))}</tbody>
    </table></div>
  );
}
export const Pager = ({ page, pages, onPage }) => (
  <div className="u-pg"><button className="u-btn s" disabled={page <= 1} onClick={() => onPage(page - 1)}>‹</button>
    <span>Page {page} of {pages}</span><button className="u-btn s" disabled={page >= pages} onClick={() => onPage(page + 1)}>›</button></div>
);

/* ---------- charts (SVG, no library) ---------- */
const downsample = (pts, max = 400) => (pts.length <= max ? pts : pts.filter((_, i) => i % Math.ceil(pts.length / max) === 0 || i === pts.length - 1));

/* series: [{name,color,dash?,points:[[x,y],...]}]  x,y in 0..1 (PR / ROC curves) */
export function CurveChart({ series, xLabel, yLabel, diagonal = false, height = 300 }) {
  const W = 520, H = height, L = 42, R = 12, T = 12, B = 36;
  const sx = x => L + x * (W - L - R), sy = y => H - B - y * (H - B - T);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} className="u-svg" role="img" aria-label={`${yLabel} vs ${xLabel}`}>
        {ticks.map(t => <g key={t}>
          <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke="var(--u-grid)" /><text x={L - 6} y={sy(t) + 4} textAnchor="end" fontSize="10" fill="var(--u-mu)">{t}</text>
          <line y1={T} y2={H - B} x1={sx(t)} x2={sx(t)} stroke="var(--u-grid)" /><text x={sx(t)} y={H - B + 14} textAnchor="middle" fontSize="10" fill="var(--u-mu)">{t}</text></g>)}
        {diagonal && <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="var(--u-mu)" strokeDasharray="4 4" strokeOpacity=".6" />}
        {series.map(s => <polyline key={s.name} fill="none" stroke={s.color} strokeWidth="2.4" strokeDasharray={s.dash} strokeLinejoin="round"
          points={downsample(s.points).map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(' ')} />)}
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" fontSize="11" fill="var(--u-mu)">{xLabel}</text>
        <text x="4" y="10" fontSize="11" fill="var(--u-mu)">{yLabel}</text>
      </svg>
      {series.length > 0 && <div className="u-legend">{series.map(s => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}</div>}
    </>
  );
}

/* segments: [{label,value,color}] */
export function Donut({ segments, top, bottom }) {
  const R = 42, C = 2 * Math.PI * R, total = segments.reduce((a, s) => a + s.value, 0) || 1;
  let off = 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
      <svg viewBox="0 0 120 120" width="150" height="150" role="img">
        <g transform="rotate(-90 60 60)" fill="none" strokeWidth="16">{segments.map(s => {
          const d = (s.value / total) * C, el = <circle key={s.label} cx="60" cy="60" r={R} stroke={s.color} strokeDasharray={`${d} ${C - d}`} strokeDashoffset={-off}><title>{`${s.label}: ${s.value}`}</title></circle>;
          off += d; return el; })}</g>
        <text x="60" y="58" textAnchor="middle" fontSize="15" fontWeight="600" fill="var(--u-tx)">{top}</text>
        <text x="60" y="73" textAnchor="middle" fontSize="9" fill="var(--u-mu)">{bottom}</text>
      </svg>
      <div className="u-legend" style={{ flexDirection: 'column', gap: 6, margin: 0 }}>{segments.map(s => <span key={s.label}><i style={{ background: s.color }} />{s.label} {((s.value / total) * 100).toFixed(1)}%</span>)}</div>
    </div>
  );
}

/* Daily line chart: values:[numbers], labels:[strings] */
export function LineChart({ values, labels, height = 220 }) {
  const W = 560, H = height, p = 38, max = Math.max(...values, 1) * 1.1;
  const x = i => p + (i * (W - p - 12)) / Math.max(values.length - 1, 1), y = v => H - 26 - (v / max) * (H - 46);
  const pts = values.map((v, i) => `${x(i)},${y(v)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="u-svg" role="img">
      {[0, 1, 2, 3, 4].map(i => { const yy = H - 26 - (i * (H - 46)) / 4; return <g key={i}><line x1={p} x2={W - 8} y1={yy} y2={yy} stroke="var(--u-grid)" /><text x={p - 6} y={yy + 4} textAnchor="end" fontSize="10" fill="var(--u-mu)">{Math.round((max * i) / 4)}</text></g>; })}
      <polygon points={`${p},${H - 26} ${pts} ${x(values.length - 1)},${H - 26}`} fill="var(--u-blue)" fillOpacity=".1" />
      <polyline points={pts} fill="none" stroke="var(--u-blue)" strokeWidth="2.2" />
      {values.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r="3" fill="var(--u-blue)"><title>{`${labels?.[i] ?? i}: ${v}`}</title></circle>)}
      {labels && values.map((_, i) => i % Math.ceil(values.length / 9) === 0 && <text key={i} x={x(i)} y={H - 8} textAnchor="middle" fontSize="10" fill="var(--u-mu)">{labels[i]}</text>)}
    </svg>
  );
}

export function ConfusionMatrix({ tn, fp, fn, tp, fmt = v => v }) {
  return (
    <div className="u-cm">
      <div /><div className="h">Predicted normal</div><div className="h">Predicted suspicious</div>
      <div className="h">Actually normal</div>
      <div className="c tn"><small>True negative</small><b>{fmt(tn)}</b></div><div className="c fp"><small>False positive</small><b>{fmt(fp)}</b></div>
      <div className="h">Actually laundering</div>
      <div className="c fn"><small>False negative (missed)</small><b>{fmt(fn)}</b></div><div className="c tp"><small>True positive (caught)</small><b>{fmt(tp)}</b></div>
    </div>
  );
}
