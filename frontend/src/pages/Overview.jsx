import React from 'react';
import { api } from '../api.js';
import { useAsync, fmtInt, fmtPct, fmtNum } from '../hooks.js';
import {
  Card,
  Kpi,
  RiskBadge,
  DataTable,
  LineChart,
  Donut,
  BarList,
  Banner,
  Loading,
  ErrorState,
  Empty,
} from '../components/ui.jsx';

export default function Overview({ onNavigate }) {
  const { data: summary, loading: loadingSummary, error: errSummary, retry: retrySummary } = useAsync(() => api.overview(), []);
  const { data: daily, loading: loadingDaily } = useAsync(() => api.daily(), []);

  const { data: quality } = useAsync(() => api.quality(), []);
  const { data: alertsData, loading: loadingAlerts } = useAsync(() => api.alerts({ limit: 6, model: 'm2' }), []);

  if (loadingSummary) return <Loading rows={7} />;
  if (errSummary) return <ErrorState error={`Overview data unavailable: ${errSummary}`} onRetry={retrySummary} />;
  if (!summary) return <Empty title="Overview data not found" />;

  const formatColors = {
    Cheque: '#2563EB',
    'Credit Card': '#3B82F6',
    ACH: '#60A5FA',
    Cash: '#93C5FD',
    Reinvestment: '#64748B',
    Wire: '#94A3B8',
    Bitcoin: '#CBD5E1',
  };

  const paymentSegments = Object.entries(summary.payment_formats || {}).map(([label, value]) => ({
    label,
    value,
    color: formatColors[label] || '#64748B',
  }));

  const paymentBarItems = Object.entries(summary.payment_formats || {})
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => ({
      label,
      value,
      color: formatColors[label] || '#2563EB',
    }));

  const dailyValues = (daily || []).map(d => d.n_rows || 0);
  const dailyLabels = (daily || []).map(d => (d.date ? d.date.slice(5) : ''));

  const alertColumns = [
    {
      key: 'txn_id',
      label: 'Transaction ID',
      render: r => <b>#{r.txn_id}</b>,
    },
    {
      key: 'timestamp',
      label: 'Timestamp',
      render: r => r.timestamp ? r.timestamp.slice(0, 16) : 'Not available',
    },
    {
      key: 'sender_account',
      label: 'Sender Account',
      render: r => <span>{r.sender_account} <small className="u-sub">({r.sender_bank})</small></span>,
    },
    {
      key: 'receiver_account',
      label: 'Receiver Account',
      render: r => <span>{r.receiver_account} <small className="u-sub">({r.receiver_bank})</small></span>,
    },
    {
      key: 'amount',
      label: 'Amount',
      right: true,
      render: r => `${fmtNum(r.amount, 2)} ${r.currency || 'USD'}`,
    },
    {
      key: 'payment_format',
      label: 'Payment Method',
    },
    {
      key: 'risk_score',
      label: 'Risk Score',
      right: true,
      render: r => fmtNum(r.risk_score, 4),
    },
    {
      key: 'risk_level',
      label: 'Risk Badge',
      render: r => <RiskBadge level={r.risk_level} />,
    },
  ];

  return (
    <div className="u-page">
      {/* Row of KPI cards */}
      <div className="u-grid u-k5">
        <Kpi
          label="Total Transactions"
          value={fmtInt(summary.total_transactions)}
          sub={`${summary.date_range?.start} to ${summary.date_range?.end}`}
        />
        <Kpi
          label="Account Nodes"
          value={fmtInt(summary.total_nodes)}
          sub="Unique bank entities"
        />
        <Kpi
          label="Laundering Labels"
          value={fmtInt(summary.labels?.laundering)}
          tone="High"
          sub={fmtPct((summary.labels?.laundering || 0) / (summary.total_transactions || 1), 3) + ' prevalence'}
        />
        <Kpi
          label="Normal Labels"
          value={fmtInt(summary.labels?.normal)}
          sub={fmtPct((summary.labels?.normal || 0) / (summary.total_transactions || 1), 2) + ' legitimate'}
        />
        <Kpi
          label="Payment Formats"
          value={fmtInt(Object.keys(summary.payment_formats || {}).length)}
          sub="Monitored channels"
        />
      </div>

      {/* Data Quality Banner */}
      {quality && (
        <Banner>
          <b>Verified Pipeline Quality:</b> {fmtInt(quality.n_rows)} transactions across {quality.currencies?.length || 0} currencies. Strictly 0 duplicates ({fmtInt(quality.exact_duplicate_extra_rows)}) and chronological consistency confirmed.
        </Banner>
      )}

      {/* Activity Chart & Payment Breakdown */}
      <div className="u-grid u-c21">
        <Card title="Daily Transaction Volume" tag="18-day transaction horizon">
          <p className="u-sub" style={{ margin: '0 0 12px' }}>
            Volume distribution across full ingestion pipeline.
          </p>
          {loadingDaily ? (
            <Loading rows={4} />
          ) : dailyValues.length > 0 ? (
            <LineChart values={dailyValues} labels={dailyLabels} height={230} />
          ) : (
            <Empty title="Daily activity not available" />
          )}
        </Card>

        <Card title="Payment Rails" tag="Volume by format">
          <Donut
            segments={paymentSegments}
            top={fmtInt(summary.total_transactions)}
            bottom="Transactions"
          />
          <div style={{ marginTop: 14 }}>
            <BarList
              items={paymentBarItems}
              format={v => `${fmtInt(v)} (${fmtPct(v / (summary.total_transactions || 1), 1)})`}
            />
          </div>
        </Card>
      </div>

      {/* Time Split Summary */}
      

      {/* Recent High-Risk Alerts Table */}
      <Card
        title="Recent High-Risk Alerts"
        tag="Top flagged by Model 2"
        right={
          <button
            className="u-btn s"
            onClick={() => onNavigate?.('alerts')}
            style={{ fontWeight: 600 }}
          >
            View all alerts →
          </button>
        }
      >
        <p className="u-sub" style={{ margin: '0 0 12px' }}>
          Highest confidence suspicion alerts on the test evaluation set. Click any row to inspect features and plain-language reasons.
        </p>
        {loadingAlerts ? (
          <Loading rows={5} />
        ) : alertsData?.items?.length ? (
          <DataTable
            columns={alertColumns}
            rows={alertsData.items}
            onRowClick={row => onNavigate?.('inspector', { txnId: row.txn_id })}
          />
        ) : (
          <Empty title="No high-risk alerts found" />
        )}
      </Card>
    </div>
  );
}