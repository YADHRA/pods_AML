import React from 'react';
import { api } from '../api.js';
import { useAsync, fmtInt, fmtPct } from '../hooks.js';
import { Card, Kpi, DataTable, Loading, ErrorState, Empty, Banner } from '../components/ui.jsx';

export default function DataSplit() {
  const { data: splits, loading, error, retry } = useAsync(() => api.splits(), []);

  if (loading) return <Loading rows={6} />;
  if (error) return <ErrorState error={`Failed to load dataset split data: ${error}`} onRetry={retry} />;
  if (!splits || !splits.length) return <Empty title="No data split available" />;

  const totalRows = splits.reduce((acc, s) => acc + (s.n_rows || 0), 0);
  const totalPos = splits.reduce((acc, s) => acc + (s.n_pos || 0), 0);
  const trainSplit = splits.find(s => s.split === 'train') || {};
  const testSplit = splits.find(s => s.split === 'test') || {};

  const columns = [
    {
      key: 'split',
      label: 'Split Name',
      render: r => <b style={{ textTransform: 'capitalize' }}>{r.split}</b>,
    },
    {
      key: 'n_rows',
      label: 'Transactions',
      right: true,
      render: r => fmtInt(r.n_rows),
    },
    {
      key: 'n_pos',
      label: 'Laundering Cases',
      right: true,
      render: r => fmtInt(r.n_pos),
    },
    {
      key: 'pos_rate',
      label: 'Prevalence Rate',
      right: true,
      render: r => fmtPct(r.pos_rate, 4),
    },
    {
      key: 'date_window',
      label: 'Date Range',
      render: r => `${r.date_min} to ${r.date_max}`,
    },
  ];

  return (
    <div className="u-page">
      <div className="u-grid u-k4">
        <Kpi label="Total Transactions" value={fmtInt(totalRows)} sub="Entire curated pipeline" />
        <Kpi label="Laundering Transactions" value={fmtInt(totalPos)} tone="High" sub={fmtPct(totalPos / (totalRows || 1), 3) + ' overall prevalence'} />
        <Kpi label="Training Partition" value={fmtInt(trainSplit.n_rows)} sub={`${trainSplit.date_min} to ${trainSplit.date_max}`} />
        <Kpi label="Held-Out Test Partition" value={fmtInt(testSplit.n_rows)} sub={`${testSplit.date_min} to ${testSplit.date_max}`} />
      </div>

      <Card title="Temporal Partitions" tag="Strict chronological boundary">
        <p className="u-sub" style={{ marginBottom: 16 }}>
          Data is split strictly on chronological dates to simulate production deployment and prevent target and feature leakage. Features on transaction date <i>D</i> use only data prior to date <i>D</i>.
        </p>
        <DataTable columns={columns} rows={splits} />
      </Card>

      <Banner>
        <b>Strict Time-Split Discipline:</b> Validation and test evaluations are executed on chronological horizons strictly subsequent to training data. The tail split assesses sustained out-of-time generalisation.
      </Banner>
    </div>
  );
}