import React, { useState } from 'react';
import { api } from '../api.js';
import { useAsync, fmtInt, fmtNum } from '../hooks.js';
import {
  Card,
  Kpi,
  RiskBadge,
  Segmented,
  DataTable,
  Pager,
  Loading,
  ErrorState,
  Empty,
} from '../components/ui.jsx';

export default function Alerts({ onNavigate }) {
  const [tab, setTab] = useState('txns'); // 'txns' | 'accounts'
  const [model, setModel] = useState('m2'); // 'm1' | 'm2'
  const [riskFilter, setRiskFilter] = useState('all');
  const [formatFilter, setFormatFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('score');
  const [page, setPage] = useState(1);
  const pageSize = 10;

  // Retrieve frozen validation thresholds dynamically via API
  const { data: evalData } = useAsync(() => api.evaluation(), []);
  const frozenThreshold =
    evalData?.models?.[model]?.validation_threshold?.threshold ??
    (model === 'm1' ? 0.066114 : 0.262835);

  // Fetch transactions alerts
  const {
    data: txnAlerts,
    loading: loadingTxns,
    error: errTxns,
    retry: retryTxns,
  } = useAsync(
    () =>
      api.alerts({
        model,
        risk_level: riskFilter !== 'all' ? riskFilter : undefined,
        payment_format: formatFilter !== 'all' ? formatFilter : undefined,
        search: search.trim() || undefined,
        sort_by: sortBy,
        limit: pageSize,
        offset: (page - 1) * pageSize,
      }),
    [model, riskFilter, formatFilter, search, sortBy, page, tab]
  );

  // Fetch accounts alerts
  const {
    data: accAlerts,
    loading: loadingAccs,
    error: errAccs,
    retry: retryAccs,
  } = useAsync(
    () =>
      api.accountAlerts({
        model,
        risk_level: riskFilter !== 'all' ? riskFilter : undefined,
        search: search.trim() || undefined,
        limit: pageSize,
        offset: (page - 1) * pageSize,
      }),
    [model, riskFilter, search, page, tab]
  );

  const kpis =
    tab === 'txns'
      ? txnAlerts?.kpis || { total_alerts: 0, high: 0, medium: 0, low: 0 }
      : accAlerts?.kpis || { total_accounts: 0, high: 0, medium: 0, low: 0 };

  const totalItems = tab === 'txns' ? (txnAlerts?.total || 0) : (accAlerts?.total || 0);
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));

  const txnColumns = [
    {
      key: 'txn_id',
      label: 'Transaction ID',
      render: r => <b>#{r.txn_id}</b>,
    },
    {
      key: 'timestamp',
      label: 'Timestamp',
      render: r => (r.timestamp ? r.timestamp.slice(0, 16) : 'Not available'),
    },
    {
      key: 'sender_account',
      label: 'Sender',
      render: r => (
        <span>
          {r.sender_account} <small className="u-sub">({r.sender_bank})</small>
        </span>
      ),
    },
    {
      key: 'receiver_account',
      label: 'Receiver',
      render: r => (
        <span>
          {r.receiver_account} <small className="u-sub">({r.receiver_bank})</small>
        </span>
      ),
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

  const accColumns = [
    {
      key: 'account',
      label: 'Account Number',
      render: r => <b>{r.account}</b>,
    },
    {
      key: 'bank',
      label: 'Bank Entity',
    },
    {
      key: 'alert_count',
      label: 'Flagged Transactions',
      right: true,
      render: r => fmtInt(r.alert_count),
    },
    {
      key: 'highest_score',
      label: 'Highest Score',
      right: true,
      render: r => fmtNum(r.highest_score, 4),
    },
    {
      key: 'risk_level',
      label: 'Risk Level',
      render: r => <RiskBadge level={r.risk_level} />,
    },
  ];

  return (
    <div className="u-page">
      {/* Top Controls: Segmented Tabs & Model Switcher */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <Segmented
          value={tab}
          options={[
            { value: 'txns', label: 'Flagged Transactions' },
            { value: 'accounts', label: 'Suspicious Accounts' },
          ]}
          onChange={val => {
            setTab(val);
            setPage(1);
          }}
        />

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="u-sub" style={{ fontWeight: 500 }}>Active Model:</span>
          <Segmented
            value={model}
            options={[
              { value: 'm1', label: 'M1 (Transaction only)' },
              { value: 'm2', label: 'M2 (Graph & History)' },
            ]}
            onChange={val => {
              setModel(val);
              setPage(1);
            }}
          />
        </div>
      </div>

      {/* KPI Strip */}
      <div className="u-grid u-k4">
        <Kpi
          label={tab === 'txns' ? 'Total Alerts' : 'Total Accounts'}
          value={fmtInt(tab === 'txns' ? kpis.total_alerts : kpis.total_accounts)}
          sub={`Threshold: ≥ ${fmtNum(frozenThreshold, 4)}`}
        />
        <Kpi
          label="High Risk"
          value={fmtInt(kpis.high)}
          tone="High"
          sub="Risk score ≥ 0.50"
        />
        <Kpi
          label="Medium Risk"
          value={fmtInt(kpis.medium)}
          tone="Medium"
          sub={`Score ${fmtNum(frozenThreshold, 3)} to 0.50`}
        />
        <Kpi
          label="Low Risk"
          value={fmtInt(kpis.low)}
          sub={`Score < ${fmtNum(frozenThreshold, 3)}`}
        />
      </div>

      {/* Filters and Search Bar */}
      <Card>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <div style={{ flex: '1 1 200px' }}>
            <input
              type="text"
              placeholder={tab === 'txns' ? 'Search Transaction ID, account, or bank…' : 'Search account number or bank…'}
              value={search}
              onChange={e => {
                setSearch(e.target.value);
                setPage(1);
              }}
              style={{
                width: '100%',
                padding: '8px 12px',
                borderRadius: 8,
                border: '1px solid var(--u-bd)',
                background: 'var(--u-card)',
                color: 'var(--u-tx)',
                fontSize: 13,
              }}
            />
          </div>

          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <label className="u-sub">Risk Level:</label>
            <select
              value={riskFilter}
              onChange={e => {
                setRiskFilter(e.target.value);
                setPage(1);
              }}
              style={{
                padding: '7px 10px',
                borderRadius: 8,
                border: '1px solid var(--u-bd)',
                background: 'var(--u-card)',
                color: 'var(--u-tx)',
                fontSize: 13,
              }}
            >
              <option value="all">All Levels</option>
              <option value="High">High Risk</option>
              <option value="Medium">Medium Risk</option>
              <option value="Low">Low Risk</option>
            </select>
          </div>

          {tab === 'txns' && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <label className="u-sub">Payment Rail:</label>
              <select
                value={formatFilter}
                onChange={e => {
                  setFormatFilter(e.target.value);
                  setPage(1);
                }}
                style={{
                  padding: '7px 10px',
                  borderRadius: 8,
                  border: '1px solid var(--u-bd)',
                  background: 'var(--u-card)',
                  color: 'var(--u-tx)',
                  fontSize: 13,
                }}
              >
                <option value="all">All Formats</option>
                <option value="ACH">ACH</option>
                <option value="Bitcoin">Bitcoin</option>
                <option value="Cash">Cash</option>
                <option value="Cheque">Cheque</option>
                <option value="Credit Card">Credit Card</option>
                <option value="Reinvestment">Reinvestment</option>
                <option value="Wire">Wire</option>
              </select>
            </div>
          )}

          {tab === 'txns' && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <label className="u-sub">Sort By:</label>
              <select
                value={sortBy}
                onChange={e => {
                  setSortBy(e.target.value);
                  setPage(1);
                }}
                style={{
                  padding: '7px 10px',
                  borderRadius: 8,
                  border: '1px solid var(--u-bd)',
                  background: 'var(--u-card)',
                  color: 'var(--u-tx)',
                  fontSize: 13,
                }}
              >
                <option value="score">Risk Score (High to Low)</option>
                <option value="amount">Amount (High to Low)</option>
                <option value="date">Timestamp (Newest First)</option>
              </select>
            </div>
          )}
        </div>
      </Card>

      {/* Main Table */}
      <Card
        title={tab === 'txns' ? 'Flagged Transactions' : 'Suspicious Accounts'}
        tag={`${fmtInt(totalItems)} records matching`}
      >
        <p className="u-sub" style={{ margin: '0 0 12px' }}>
          {tab === 'txns'
            ? 'Server-side paginated alerts scored by ' + (model === 'm1' ? 'Model 1 (transaction attributes only).' : 'Model 2 (graph & account history features included).')
            : 'Aggregated accounts ranked by highest transaction alert score.'}
          {' '}Click any row to open the complete Investigation view.
        </p>

        {tab === 'txns' ? (
          loadingTxns ? (
            <Loading rows={10} />
          ) : errTxns ? (
            <ErrorState error={`Failed to load transaction alerts: ${errTxns}`} onRetry={retryTxns} />
          ) : txnAlerts?.items?.length ? (
            <>
              <DataTable
                columns={txnColumns}
                rows={txnAlerts.items}
                onRowClick={row => onNavigate?.('inspector', { txnId: row.txn_id, model })}
              />
              <div style={{ marginTop: 16 }}>
                <Pager page={page} pages={totalPages} onPage={p => setPage(p)} />
              </div>
            </>
          ) : (
            <Empty title="No transaction alerts found matching your criteria" />
          )
        ) : loadingAccs ? (
          <Loading rows={10} />
        ) : errAccs ? (
          <ErrorState error={`Failed to load account alerts: ${errAccs}`} onRetry={retryAccs} />
        ) : accAlerts?.items?.length ? (
          <>
            <DataTable
              columns={accColumns}
              rows={accAlerts.items}
              onRowClick={row => onNavigate?.('inspector', { account: row.account, model })}
            />
            <div style={{ marginTop: 16 }}>
              <Pager page={page} pages={totalPages} onPage={p => setPage(p)} />
            </div>
          </>
        ) : (
          <Empty title="No suspicious accounts found matching your criteria" />
        )}
      </Card>
    </div>
  );
}
