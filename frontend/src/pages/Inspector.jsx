import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import { useAsync, fmtInt, fmtNum } from '../hooks.js';
import {
  prettifyFeatureName,
  getFeatureExplanation,
  getImpactLevel,
} from '../featureLabels.js';
import {
  Card,
  Kpi,
  RiskBadge,
  Badge,
  Segmented,
  KeyValue,
  BarList,
  DataTable,
  Banner,
  Loading,
  ErrorState,
  Empty,
} from '../components/ui.jsx';

export default function Inspector({ params = {}, onNavigate }) {
  const [viewMode, setViewMode] = useState(params.account ? 'account' : 'transaction');
  const [model, setModel] = useState(params.model || 'm2');

  // Input states
  const [txnInput, setTxnInput] = useState(String(params.txnId || '4269461'));
  const [activeTxnId, setActiveTxnId] = useState(params.txnId || 4269461);

  const [accInput, setAccInput] = useState(String(params.account || '8000EBD30'));
  const [activeAccQuery, setActiveAccQuery] = useState(params.account || '8000EBD30');

  // Sync incoming navigation params
  useEffect(() => {
    if (params.txnId) {
      setActiveTxnId(params.txnId);
      setTxnInput(String(params.txnId));
      setViewMode('transaction');
    } else if (params.account) {
      setActiveAccQuery(params.account);
      setAccInput(String(params.account));
      setViewMode('account');
    }
    if (params.model) {
      setModel(params.model);
    }
  }, [params]);

  // Fetch Transaction Data
  const {
    data: txnData,
    loading: loadingTxn,
    error: errTxn,
    retry: retryTxn,
  } = useAsync(
    () => (viewMode === 'transaction' && activeTxnId ? api.transaction(activeTxnId) : Promise.resolve(null)),
    [activeTxnId, viewMode]
  );

  // Fetch Model Predictions for Transaction
  const { data: predData } = useAsync(
    () => (viewMode === 'transaction' && activeTxnId ? api.prediction(activeTxnId) : Promise.resolve(null)),
    [activeTxnId, viewMode]
  );

  // Fetch SHAP Explanation for Transaction
  const {
    data: explData,
    loading: loadingExpl,
  } = useAsync(
    () => (viewMode === 'transaction' && activeTxnId ? api.transactionExplanation(activeTxnId, model) : Promise.resolve(null)),
    [activeTxnId, model, viewMode]
  );

  // Fetch Account Data
  const {
    data: accData,
    loading: loadingAcc,
    error: errAcc,
    retry: retryAcc,
  } = useAsync(
    () => (viewMode === 'account' && activeAccQuery ? api.accountLookup(activeAccQuery) : Promise.resolve(null)),
    [activeAccQuery, viewMode]
  );

  const handleTxnSearch = (e) => {
    e.preventDefault();
    const val = txnInput.trim().replace(/^#/, '');
    if (val) setActiveTxnId(parseInt(val, 10) || val);
  };

  const handleAccSearch = (e) => {
    e.preventDefault();
    const val = accInput.trim();
    if (val) setActiveAccQuery(val);
  };

  // Determine current transaction model score and level
  const activePred = predData?.[`${model}_full`] || { score: 0, alert: false };
  const currentScore = activePred.score ?? 0;
  const currentRiskLevel =
    currentScore >= 0.50 ? 'High' : currentScore >= (model === 'm1' ? 0.0661 : 0.2628) ? 'Medium' : 'Low';

  // Format Top SHAP bars for "What mattered most"
  const rawFeatures = explData?.features || [];
  const barItems = rawFeatures.map(f => ({
    label: prettifyFeatureName(f.feature),
    value: f.shap_value,
    color: f.shap_value > 0 ? '#DC2626' : '#2563EB',
  }));

  return (
    <div className="u-page">
      {/* Top Controls: Switcher between Transaction and Account views */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <Segmented
          value={viewMode}
          options={[
            { value: 'transaction', label: 'Transaction Details' },
            { value: 'account', label: 'Account Profile' },
          ]}
          onChange={setViewMode}
        />

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="u-sub" style={{ fontWeight: 500 }}>Active Model:</span>
          <Segmented
            value={model}
            options={[
              { value: 'm1', label: 'M1 (Transaction only)' },
              { value: 'm2', label: 'M2 (Graph & History)' },
            ]}
            onChange={setModel}
          />
        </div>
      </div>

      {/* Search Bar */}
      <Card>
        {viewMode === 'transaction' ? (
          <form onSubmit={handleTxnSearch} style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 600, fontSize: 13 }}>Inspect Transaction:</span>
            <input
              type="text"
              placeholder="Enter Transaction ID (e.g. 4269461, 812709, 4215358)"
              value={txnInput}
              onChange={e => setTxnInput(e.target.value)}
              style={{
                flex: '1 1 240px',
                padding: '7px 12px',
                borderRadius: 8,
                border: '1px solid var(--u-bd)',
                background: 'var(--u-card)',
                color: 'var(--u-tx)',
                fontSize: 13,
              }}
            />
            <button type="submit" className="u-btn">Search</button>
            <div style={{ display: 'flex', gap: 6 }}>
              {[4269461, 812709, 4215358].map(id => (
                <button
                  type="button"
                  key={id}
                  className="u-btn s"
                  onClick={() => {
                    setTxnInput(String(id));
                    setActiveTxnId(id);
                  }}
                >
                  #{id}
                </button>
              ))}
            </div>
          </form>
        ) : (
          <form onSubmit={handleAccSearch} style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 600, fontSize: 13 }}>Inspect Account:</span>
            <input
              type="text"
              placeholder="Enter Account String (e.g. 8000EBD30) or Node ID"
              value={accInput}
              onChange={e => setAccInput(e.target.value)}
              style={{
                flex: '1 1 240px',
                padding: '7px 12px',
                borderRadius: 8,
                border: '1px solid var(--u-bd)',
                background: 'var(--u-card)',
                color: 'var(--u-tx)',
                fontSize: 13,
              }}
            />
            <button type="submit" className="u-btn">Search</button>
            <div style={{ display: 'flex', gap: 6 }}>
              {['8000EBD30', '8004DF550', '80491A540'].map(acc => (
                <button
                  type="button"
                  key={acc}
                  className="u-btn s"
                  onClick={() => {
                    setAccInput(acc);
                    setActiveAccQuery(acc);
                  }}
                >
                  {acc}
                </button>
              ))}
            </div>
          </form>
        )}
      </Card>

      {/* Mandatory Regulatory / Safeguard Banner */}
      <Banner>
        <b>Investigative Context:</b> A flag is a reason to look closer, not proof of wrongdoing.
      </Banner>

      {/* ======================================================== */}
      {/* TRANSACTION VIEW                                         */}
      {/* ======================================================== */}
      {viewMode === 'transaction' && (
        loadingTxn ? (
          <Loading rows={8} />
        ) : errTxn ? (
          <ErrorState error={`Transaction #${activeTxnId} not found: ${errTxn}`} onRetry={retryTxn} />
        ) : txnData ? (
          <>
            {/* Risk Card */}
            <div className="u-grid u-k4">
              <Kpi
                label="Risk Score"
                value={fmtNum(currentScore, 4)}
                tone={currentRiskLevel}
                sub={`Scored by ${model.toUpperCase()}`}
              />
              <Kpi
                label="Assigned Risk Level"
                value={currentRiskLevel}
                tone={currentRiskLevel}
                sub={currentScore >= 0.50 ? 'High priority triage' : 'Review recommended'}
              />
              <Kpi
                label="Model Recommendation"
                value={activePred.alert ? 'Flagged for Review' : 'Standard Baseline'}
                tone={activePred.alert ? 'High' : ''}
                sub={`Threshold: ${model === 'm1' ? '0.0661' : '0.2628'}`}
              />
              <Kpi
                label="Active Evaluation Scope"
                value={model === 'm1' ? 'M1 (Transaction only)' : 'M2 (Full Graph & History)'}
                sub={model === 'm1' ? '8 isolated attributes' : '35 multi-relational attributes'}
              />
            </div>

            {/* Three Cards Layout (Screenshot 2 Match) */}
            <div className="u-grid u-k3">
              {/* Card 1: Transaction Details */}
              <Card title="Transaction Details" tag={`#${txnData.txn_id}`}>
                <KeyValue
                  rows={[
                    ['Transaction ID', `#${txnData.txn_id}`],
                    ['Execution Timestamp', txnData.timestamp || 'Not available'],
                    [
                      'Sender Account',
                      <button
                        key="src"
                        className="u-btn s"
                        style={{ padding: '2px 8px', fontSize: 12 }}
                        onClick={() => {
                          setActiveAccQuery(txnData.src_account);
                          setAccInput(txnData.src_account);
                          setViewMode('account');
                        }}
                      >
                        {txnData.src_account} (Bank {txnData.src_bank})
                      </button>,
                    ],
                    [
                      'Receiver Account',
                      <button
                        key="dst"
                        className="u-btn s"
                        style={{ padding: '2px 8px', fontSize: 12 }}
                        onClick={() => {
                          setActiveAccQuery(txnData.dst_account);
                          setAccInput(txnData.dst_account);
                          setViewMode('account');
                        }}
                      >
                        {txnData.dst_account} (Bank {txnData.dst_bank})
                      </button>,
                    ],
                    ['Amount Transferred', `${fmtNum(txnData.amount_paid, 2)} ${txnData.currency_paid}`],
                    ['Amount Received', `${fmtNum(txnData.amount_received, 2)} ${txnData.currency_received}`],
                    ['Payment Channel', txnData.payment_format],
                    ['Self-Loop Entity', txnData.is_self_loop ? 'Yes (internal)' : 'No (external)'],
                  ]}
                />
              </Card>

              {/* Card 2: Why was it flagged? (Plain English) */}
              <Card title="Why was it flagged?" tag={model === 'm1' ? 'Transaction reasons' : 'Network & history reasons'}>
                <p className="u-sub" style={{ margin: '0 0 12px' }}>
                  Natural language interpretation derived from real feature contributions for {model.toUpperCase()}.
                </p>
                {loadingExpl ? (
                  <Loading rows={4} />
                ) : rawFeatures.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {rawFeatures.slice(0, 5).map((f, i) => (
                      <div
                        key={i}
                        style={{
                          padding: '10px 12px',
                          borderRadius: 8,
                          background: 'var(--u-bg)',
                          border: '1px solid var(--u-bd)',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                          <b style={{ fontSize: 13 }}>{prettifyFeatureName(f.feature)}</b>
                          <Badge tone={f.impact === 'High' ? 'High' : f.impact === 'Medium' ? 'Medium' : 'n'}>
                            {f.impact} Impact
                          </Badge>
                        </div>
                        <div style={{ fontSize: 12.5, color: 'var(--u-tx)', lineHeight: 1.4 }}>
                          {getFeatureExplanation(f.feature, f.feature_value, f.shap_value)}
                        </div>
                        {f.feature_value != null && (
                          <div className="u-sub" style={{ fontSize: 11, marginTop: 4 }}>
                            Observed value: <b>{String(f.feature_value)}</b>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <Empty title="No anomalous drivers found" />
                )}
              </Card>

              {/* Card 3: What mattered most (SHAP Bar List) */}
              <Card title="What mattered most" tag="Top contributing factors">
                <p className="u-sub" style={{ margin: '0 0 12px' }}>
                  Relative attribute impact driving the final prediction score.
                </p>
                {loadingExpl ? (
                  <Loading rows={5} />
                ) : barItems.length > 0 ? (
                  <BarList
                    items={barItems}
                    format={v => (v > 0 ? `+${fmtNum(v, 2)}` : fmtNum(v, 2))}
                  />
                ) : (
                  <Empty title="No feature weights available" />
                )}
              </Card>
            </div>
          </>
        ) : (
          <Empty title="Search a transaction to begin investigation" />
        )
      )}

      {/* ======================================================== */}
      {/* ACCOUNT VIEW                                             */}
      {/* ======================================================== */}
      {viewMode === 'account' && (
        loadingAcc ? (
          <Loading rows={8} />
        ) : errAcc ? (
          <ErrorState error={`Account '${activeAccQuery}' lookup failed: ${errAcc}`} onRetry={retryAcc} />
        ) : accData ? (
          <>
            {/* Account Profile KPIs */}
            <div className="u-grid u-k4">
              <Kpi
                label="Account Identifier"
                value={accData.account}
                sub={`Bank Entity: ${accData.bank}`}
              />
              <Kpi
                label="Assigned Risk Status"
                value={accData.risk_level}
                tone={accData.risk_level}
                sub={`Peak transaction risk: ${fmtNum(accData.highest_score, 4)}`}
              />
              <Kpi
                label="Historical Activity"
                value={fmtInt(accData.transaction_count)}
                sub="Lifetime recorded transactions"
              />
              <Kpi
                label="Active Window"
                value={accData.first_seen ? accData.first_seen.slice(0, 10) : 'Not available'}
                sub={`Last seen: ${accData.last_seen ? accData.last_seen.slice(0, 10) : 'Active'}`}
              />
            </div>

            <div className="u-grid u-c2">
              {/* Account Behavioral Characteristics */}
              <Card title="Account Profile & Behavior" tag={`Node ID #${accData.node_id}`}>
                <KeyValue
                  rows={[
                    ['Account Number', accData.account],
                    ['Institution Bank ID', accData.bank],
                    ['System Graph Node ID', String(accData.node_id)],
                    ['Activity Span', `${accData.first_seen?.slice(0, 16)} → ${accData.last_seen?.slice(0, 16)}`],
                    ['Outbound Counterparty Rate', accData.historical_behavior?.unique_out_log ? 'Multiple distinct outward destinations' : 'Low outward dispersion'],
                    ['Inbound Counterparty Rate', accData.historical_behavior?.unique_in_log ? 'Concentrated inward deposits' : 'Standard inward sources'],
                    ['Network Connection Scope', accData.historical_behavior?.wcc_size_log ? 'Connected to multi-account cluster' : 'Peripheral connection'],
                  ]}
                />
              </Card>

              {/* Alerted Transactions for this Account */}
              <Card
                title="Flagged Transactions for Account"
                tag={`${accData.alerted_transactions?.length || 0} alerts`}
              >
                <p className="u-sub" style={{ margin: '0 0 12px' }}>
                  Transactions associated with this account flagged by AML detection. Click any row to inspect details.
                </p>
                {accData.alerted_transactions?.length > 0 ? (
                  <DataTable
                    columns={[
                      {
                        key: 'txn_id',
                        label: 'Txn ID',
                        render: r => <b>#{r.txn_id}</b>,
                      },
                      {
                        key: 'role',
                        label: 'Role',
                        render: r => <span style={{ fontWeight: 600 }}>{r.role}</span>,
                      },
                      {
                        key: 'amount',
                        label: 'Amount',
                        right: true,
                        render: r => `${fmtNum(r.amount, 2)} ${r.currency}`,
                      },
                      {
                        key: 'score',
                        label: 'Risk Score',
                        right: true,
                        render: r => fmtNum(r.score, 4),
                      },
                      {
                        key: 'risk_level',
                        label: 'Badge',
                        render: r => <RiskBadge level={r.risk_level} />,
                      },
                    ]}
                    rows={accData.alerted_transactions}
                    onRowClick={r => {
                      setActiveTxnId(r.txn_id);
                      setTxnInput(String(r.txn_id));
                      setViewMode('transaction');
                    }}
                  />
                ) : (
                  <Empty title="No flagged transactions for this account" />
                )}
              </Card>
            </div>
          </>
        ) : (
          <Empty title="Search an account string or Node ID to inspect" />
        )
      )}
    </div>
  );
}