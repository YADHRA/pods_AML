# PODS AML Frontend

React + Vite UI inspired by `PODS AML — UI prototype.html`. The interface calls the FastAPI backend and does not create pretend transactions or model metrics.

## Requirements
- Node.js 20+ recommended
- Backend running at `http://127.0.0.1:8000`

## Run on Windows PowerShell
```powershell
cd path\to\pods_aml_frontend
npm install
Copy-Item .env.example .env
npm run dev
```
Open the local URL Vite prints (usually `http://localhost:5173`).

## Run in WSL / Ubuntu
```bash
cd ~/pods_aml_frontend
npm install
cp .env.example .env
npm run dev -- --host 0.0.0.0
```
Open the URL Vite prints in your browser.

## Current backend status
The uploaded backend implements only `GET /api/health`. The UI checks it and calls the following additional endpoints, which must be implemented before those sections can display real values:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Existing backend health check |
| GET | `/api/overview` | Dataset/model dashboard metrics |
| GET | `/api/alerts` | Filterable predicted alerts |
| GET | `/api/graph?snapshot_date=2022-09-10&depth=2` | Nodes and edges for graph |
| GET | `/api/transactions/{txn_id}` | One transaction plus prediction/explanation |
| GET | `/api/accounts/{account_id}` | Account summary/history/network metrics |
| GET | `/api/models` | M1/M2 test metrics, frozen thresholds, confusion matrices |
| GET | `/api/data-split` | Temporal split boundaries and counts |

## API response contracts

### `GET /api/overview`
```json
{
  "metrics": {
    "total_transactions": 5078345,
    "total_alerts": 0,
    "high_risk_alerts": 0,
    "accounts_involved": 515088,
    "alert_rate": 0
  }
}
```
Numbers above only illustrate the response shape. Replace with computed project results; do not copy illustrative alert counts.

### `GET /api/alerts`
```json
{
  "items": [
    {
      "txn_id": "your-transaction-id",
      "timestamp": "2022-09-10T12:30:00",
      "sender_account": "sender-id",
      "receiver_account": "receiver-id",
      "amount_paid": 1250.0,
      "risk_score": 0.91,
      "risk_level": "High",
      "prediction": 1,
      "model": "M2"
    }
  ],
  "total": 1
}
```
The row is a schema example only, not a claim about your model.

### `GET /api/graph`
```json
{
  "snapshot_date": "2022-09-10",
  "nodes": [
    {"id":"account-id-1","label":"account-id-1","bank":"bank-id","risk_level":"High","transactions":12}
  ],
  "edges": [
    {"id":"edge-1","source":"account-id-1","target":"account-id-2","amount":1250.0,"transaction_count":2,"depth":1}
  ]
}
```
The graph renderer expects `nodes` and `edges`. Every edge's `source` and `target` must match a node's `id`. Build nodes using the project's chosen account identity, e.g. `(Bank, Account)` if that is the agreed key. Return only historical edges strictly before the chosen snapshot/transaction date when generating historical graph features.

### `GET /api/transactions/{txn_id}` and `/api/accounts/{account_id}`
Return one JSON object containing the record's actual fields. Transaction response should include prediction/risk score only if produced by the model. Do not derive SHAP values in the UI.

### `GET /api/models`
Return evaluated M1 and M2 values, e.g.:
```json
{
  "models": {
    "M1": {"pr_auc": null, "roc_auc": null, "precision": null, "recall": null, "f1": null, "threshold": null, "confusion_matrix": null},
    "M2": {"pr_auc": null, "roc_auc": null, "precision": null, "recall": null, "f1": null, "threshold": null, "confusion_matrix": null}
  }
}
```
Replace `null` with the actual held-out test results and validation-selected threshold. Never fill metrics with fabricated or illustrative scores.

### `GET /api/data-split`
Return the project's final documented train/validation/test date windows and exact counts. Use the same split that was used for the model evaluation.

## CORS
When Vite runs on port 5173 and FastAPI on port 8000, configure FastAPI `CORSMiddleware` to allow `http://localhost:5173` and `http://127.0.0.1:5173`.

## Notes
- The UI displays an honest empty state when an endpoint is missing.
- `/api/health` being healthy does not mean model/data endpoints are implemented.
- A working frontend build checks syntax with `npm run build`; this does not certify that the API endpoints or model pipeline work.
