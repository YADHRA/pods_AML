
import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import { Status, Loading } from '../components/Status.jsx';

function formatNumber(value) {
  return Number(value).toLocaleString('en-IN');
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatAuc(value) {
  return Number(value).toFixed(4);
}

function Metric({ label, value, description }) {
  return (
    <div className="card">
      <p>{label}</p>
      <h2>{value}</h2>
      {description && <p>{description}</p>}
    </div>
  );
}

function ConfusionMatrix({ test }) {
  return (
    <div>
      <h4>Confusion matrix</h4>
      <p>Counts from the saved test evaluation.</p>

      <table>
        <thead>
          <tr>
            <th></th>
            <th>Predicted normal</th>
            <th>Predicted laundering</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>Actually normal</th>
            <td>{formatNumber(test.tn)} (TN)</td>
            <td>{formatNumber(test.fp)} (FP)</td>
          </tr>
          <tr>
            <th>Actually laundering</th>
            <td>{formatNumber(test.fn)} (FN)</td>
            <td>{formatNumber(test.tp)} (TP)</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function ModelSection({ title, model }) {
  const test = model.test;
  const validation = model.validation_threshold;

  return (
    <section className="card">
      <div className="section-heading">
        <div>
          <h3>{title}</h3>
          <p>
            Run: {model.run} · Seed: {model.seed} · Variant: {model.variant}
          </p>
        </div>
      </div>

      <div className="two-col">
        <Metric
          label="PR-AUC"
          value={formatAuc(test.pr_auc)}
          description={formatPercent(test.pr_auc)}
        />
        <Metric
          label="ROC-AUC"
          value={formatAuc(test.roc_auc)}
        />
        <Metric
          label="Precision"
          value={formatPercent(test.precision)}
        />
        <Metric
          label="Recall"
          value={formatPercent(test.recall)}
        />
        <Metric
          label="F1 score"
          value={formatPercent(test.f1)}
        />
        <Metric
          label="Accuracy"
          value={formatPercent(test.accuracy)}
        />
      </div>

      <h4>Test-set details</h4>
      <div className="detail-list">
        <dt>Transactions evaluated</dt>
        <dd>{formatNumber(test.n)}</dd>

        <dt>Actual laundering transactions</dt>
        <dd>{formatNumber(test.n_pos)}</dd>

        <dt>Alerted transactions</dt>
        <dd>{formatNumber(test.n_alerts)}</dd>

        <dt>False alerts</dt>
        <dd>{formatNumber(test.fp)}</dd>

        <dt>Missed laundering transactions</dt>
        <dd>{formatNumber(test.fn)}</dd>

        <dt>Lift over baseline prevalence</dt>
        <dd>{Number(test.lift).toFixed(2)}×</dd>
      </div>

      <ConfusionMatrix test={test} />

      <div className="card">
        <h4>Frozen validation threshold</h4>
        <p>
          Threshold: <strong>{Number(validation.threshold).toFixed(6)}</strong>
        </p>
        <p>
          Selection rule: {validation.rule}
        </p>
        <p>
          Validation F1: {formatPercent(validation.val_f1)}
        </p>
        <p>
          Validation precision: {formatPercent(validation.val_precision)}
        </p>
        <p>
          Validation recall: {formatPercent(validation.val_recall)}
        </p>
        <p>
          The threshold was selected on validation data and then used
          for the test evaluation.
        </p>
      </div>
    </section>
  );
}

export default function Results() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let live = true;

    api.evaluation()
      .then((result) => {
        if (live) setData(result);
      })
      .catch((err) => {
        if (live) {
          setError(err.message || 'Unable to load model evaluation results.');
        }
      })
      .finally(() => {
        if (live) setLoading(false);
      });

    return () => {
      live = false;
    };
  }, []);

  if (loading) {
    return <Loading label="Loading model evaluation results..." />;
  }

  if (error) {
    return (
      <section className="card">
        <h3>Unable to load model performance</h3>
        <Status kind="warning">{error}</Status>
        <p>
          Check that the backend is running and that
          /api/models/evaluation returns JSON.
        </p>
      </section>
    );
  }

  if (!data?.models?.m1 || !data?.models?.m2) {
    return (
      <section className="card">
        <Status kind="warning">
          The evaluation response does not contain both M1 and M2 results.
        </Status>
      </section>
    );
  }

  const m1 = data.models.m1.test;
  const m2 = data.models.m2.test;

  return (
    <div className="page-stack">
      <section className="card">
        <div className="section-heading">
          <div>
            <h3>Model performance</h3>
            <p>
              Real saved evaluation results for M1 and M2.
              Evaluation split: {data.evaluation_split}.
            </p>
          </div>
        </div>

        <Status kind="success">
          Evaluation results loaded from the backend.
        </Status>

        <p>
          Both models were tested on {formatNumber(m1.n)} transactions,
          including {formatNumber(m1.n_pos)} laundering transactions.
        </p>

        <p>
          The laundering prevalence in this test set is
          {' '}{formatPercent(m1.prevalence)}.
          PR-AUC is especially useful here because the classes are
          highly imbalanced.
        </p>
      </section>

      <section className="card">
        <h3>M1 vs M2 comparison</h3>

        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>M1 — Transaction-only</th>
              <th>M2 — Account + graph</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>PR-AUC</td>
              <td>{formatAuc(m1.pr_auc)}</td>
              <td>{formatAuc(m2.pr_auc)}</td>
            </tr>
            <tr>
              <td>ROC-AUC</td>
              <td>{formatAuc(m1.roc_auc)}</td>
              <td>{formatAuc(m2.roc_auc)}</td>
            </tr>
            <tr>
              <td>Precision</td>
              <td>{formatPercent(m1.precision)}</td>
              <td>{formatPercent(m2.precision)}</td>
            </tr>
            <tr>
              <td>Recall</td>
              <td>{formatPercent(m1.recall)}</td>
              <td>{formatPercent(m2.recall)}</td>
            </tr>
            <tr>
              <td>F1 score</td>
              <td>{formatPercent(m1.f1)}</td>
              <td>{formatPercent(m2.f1)}</td>
            </tr>
            <tr>
              <td>True positives</td>
              <td>{formatNumber(m1.tp)}</td>
              <td>{formatNumber(m2.tp)}</td>
            </tr>
            <tr>
              <td>False positives</td>
              <td>{formatNumber(m1.fp)}</td>
              <td>{formatNumber(m2.fp)}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <ModelSection
        title="M1 — Transaction-only model"
        model={data.models.m1}
      />

      <ModelSection
        title="M2 — Transaction + account + graph model"
        model={data.models.m2}
      />

      <section className="card">
        <h3>Interpretation</h3>
        <p>
          In this test run, M2 achieved a PR-AUC of
          {' '}{formatAuc(m2.pr_auc)}, compared with
          {' '}{formatAuc(m1.pr_auc)} for M1.
        </p>
        <p>
          At their respective frozen thresholds, M2 detected
          {' '}{formatNumber(m2.tp)} laundering transactions and generated
          {' '}{formatNumber(m2.fp)} false alerts. M1 detected
          {' '}{formatNumber(m1.tp)} laundering transactions and generated
          {' '}{formatNumber(m1.fp)} false alerts.
        </p>
        <p>
          These are results for seed 42 and the full-feature variants,
          not an average across all three seeds. They show performance
          on this test set, not a guarantee of future performance.
        </p>
      </section>
    </div>
  );
}