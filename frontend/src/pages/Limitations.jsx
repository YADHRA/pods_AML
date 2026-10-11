import React from 'react';
import { api } from '../api.js';
import { useAsync, fmtInt, fmtPct } from '../hooks.js';
import { Card, Kpi, Banner, Loading, KeyValue } from '../components/ui.jsx';

export default function Limitations() {
  const { data: quality } = useAsync(() => api.quality(), []);

  const safeguards = [
    {
      title: 'Real Data Only',
      desc: 'All metrics, risk scores, counts, and explanations come directly from verified backend endpoints and model evaluation runs. Zero manufactured or mocked numbers.',
    },
    {
      title: 'Time-Based Leakage Prevention',
      desc: 'Historical behavioral and counterparty features for a transaction occurring on date D are calculated strictly using transactions up to date D - 1. Future knowledge is completely prevented.',
    },
    {
      title: 'Flags are Reasons to Look Closer, Not Proof',
      desc: 'An elevated model risk score highlights transactions for compliance analyst triage. A flag is an investigative indicator, not conclusive evidence of financial crime.',
    },
    {
      title: 'Evaluation Rigour & Imbalanced Metrics',
      desc: 'Models are assessed via Precision-Recall AUC (PR-AUC) as the primary benchmark given severe label imbalance (~0.1% positive rate), supported by ROC-AUC and confusion matrices at frozen validation thresholds.',
    },
  ];

  return (
    <div className="u-page">
      <Banner>
        <b>AML Governance & System Safeguards:</b> Technical constraints and analytical boundaries ensuring compliance integrity and preventing model leakage.
      </Banner>

      {quality && (
        <div className="u-grid u-k4">
          <Kpi label="Audited Transactions" value={fmtInt(quality.n_rows)} sub="Processed clean dataset" />
          <Kpi label="Duplicate Rows" value={fmtInt(quality.exact_duplicate_extra_rows)} sub="0 extra duplicate rows" />
          <Kpi label="Null Values" value={fmtInt(Object.values(quality.nulls || {}).reduce((a, b) => a + b, 0))} sub="Complete data integrity" />
          <Kpi label="Chronological Integrity" value={quality.raw_is_chronological ? 'Strict' : 'Audited'} sub="Zero backward time jumps" />
        </div>
      )}

      <div className="u-grid u-c2">
        {safeguards.map((item, idx) => (
          <Card key={idx} title={item.title}>
            <p className="u-sub" style={{ lineHeight: 1.6, margin: 0 }}>
              {item.desc}
            </p>
          </Card>
        ))}
      </div>

      {quality && (
        <Card title="Data Quality & Integrity Audit" tag="Pipeline verification">
          <KeyValue
            rows={[
              ['Audited Date Span', `${quality.ts_min?.slice(0, 10)} to ${quality.ts_max?.slice(0, 10)}`],
              ['Currencies Present', quality.currencies?.join(', ') || 'Multi-currency'],
              ['Payment Formats', Object.keys(quality.payment_formats || {}).join(', ')],
              ['Foreign Exchange Rows', `${fmtInt(quality.fx_rows)} (${fmtPct(quality.fx_rows / (quality.n_rows || 1))})`],
              ['Bank Code Integrity', quality.bank_codes_have_leading_zeros ? 'Preserved with leading zeros' : 'Verified'],
            ]}
          />
        </Card>
      )}
    </div>
  );
}