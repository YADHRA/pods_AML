
import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Users,
  ArrowLeftRight,
  ShieldCheck,
  Database,
  CreditCard,
} from 'lucide-react';

import { api } from '../api.js';
import {
  Metric,
  Status,
  Loading,
  EmptyState,
} from '../components/Status.jsx';

export default function Overview({ onNavigate }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let live = true;

    api.overview()
      .then((result) => {
        if (live) setData(result);
      })
      .catch((err) => {
        if (live) setError(err.message);
      })
      .finally(() => {
        if (live) setLoading(false);
      });

    return () => {
      live = false;
    };
  }, []);

  if (loading) {
    return <Loading label="Loading real dashboard data..." />;
  }

  if (error || !data) {
    return (
      <div className="page-stack">
        <Status kind="warning">
          Could not load dashboard data: {error || 'Empty API response'}.
          Check that the backend is running and api.overview() uses
          /api/dashboard/summary.
        </Status>
        <EmptyState
          title="Dashboard data unavailable"
          detail="The dashboard will display real values when the summary API responds successfully."
        />
      </div>
    );
  }

  const labels = data.labels ?? {};
  const formats = data.payment_formats ?? {};

  const cards = [
    {
      label: 'Total transactions',
      value: data.total_transactions,
      icon: ArrowLeftRight,
      sub: 'Transactions in dataset',
    },
    {
      label: 'Account nodes',
      value: data.total_nodes,
      icon: Users,
      sub: 'Distinct graph nodes',
    },
    {
      label: 'Dataset laundering labels',
      value: labels.laundering,
      icon: AlertTriangle,
      sub: 'Ground-truth labels, not model alerts',
    },
    {
      label: 'Dataset normal labels',
      value: labels.normal,
      icon: ShieldCheck,
      sub: 'Ground-truth labels',
    },
    {
      label: 'Payment formats',
      value: Object.keys(formats).length,
      icon: CreditCard,
      sub: 'Formats in the dataset',
    },
  ];

  const formatNumber = (value) =>
    typeof value === 'number'
      ? value.toLocaleString('en-IN')
      : value ?? '—';

  return (
    <div className="page-stack">
      <Status kind="success">
        Dashboard loaded successfully from the backend summary API.
      </Status>

      <div className="metric-grid">
        {cards.map(({ label, value, icon: Icon, sub }) => (
          <Metric
            key={label}
            label={label}
            value={formatNumber(value)}
            icon={Icon}
            sub={sub}
          />
        ))}
      </div>

      <div className="two-col">
        <section className="card">
          <div className="section-heading">
            <div>
              <h3>Dataset summary</h3>
              <p>Values returned by the backend</p>
            </div>
            <span className="eyebrow">LIVE API</span>
          </div>

          <dl className="detail-list">
            <dt>Total transactions</dt>
            <dd>{formatNumber(data.total_transactions)}</dd>

            <dt>Total graph nodes</dt>
            <dd>{formatNumber(data.total_nodes)}</dd>

            <dt>Dataset start date</dt>
            <dd>{data.date_range?.start ?? '—'}</dd>

            <dt>Dataset end date</dt>
            <dd>{data.date_range?.end ?? '—'}</dd>

            <dt>Normal labels</dt>
            <dd>{formatNumber(labels.normal)}</dd>

            <dt>Laundering labels</dt>
            <dd>{formatNumber(labels.laundering)}</dd>
          </dl>
        </section>

        <section className="card">
          <div className="section-heading">
            <div>
              <h3>Payment format distribution</h3>
              <p>Transaction counts by payment format</p>
            </div>
            <Activity size={20} />
          </div>

          {Object.keys(formats).length > 0 ? (
            <div className="payment-list">
              {Object.entries(formats)
                .sort((a, b) => b[1] - a[1])
                .map(([name, count]) => {
                  const percentage =
                    data.total_transactions > 0
                      ? (count / data.total_transactions) * 100
                      : 0;

                  return (
                    <div className="payment-item" key={name}>
                      <div className="payment-item-heading">
                        <span>{name}</span>
                        <span>{formatNumber(count)}</span>
                      </div>

                      <div className="payment-bar">
                        <div
                          className="payment-bar-fill"
                          style={{ width: `${percentage}%` }}
                        />
                      </div>

                      <p className="muted small">
                        {percentage.toFixed(2)}% of all transactions
                      </p>
                    </div>
                  );
                })}
            </div>
          ) : (
            <EmptyState
              title="No payment format data"
              detail="The backend did not return payment format counts."
            />
          )}
        </section>
      </div>

      <section className="card">
        <div className="section-heading">
          <div>
            <h3>Model selection</h3>
            <p>Models planned for AML investigation</p>
          </div>
        </div>

        <div className="model-tile">
          <b>M1 · Transaction-only</b>
          <p>Uses transaction-level features only.</p>
        </div>

        <div className="model-tile active">
          <b>M2 · Transaction + Account + Graph</b>
          <p>
            Combines transaction features with historical account and
            graph features, using only information available before
            the transaction date.
          </p>
        </div>

        <p className="muted small">
          Model-generated alert counts and performance metrics are not
          displayed because they are not provided by this summary endpoint.
        </p>
      </section>
    </div>
  );
}