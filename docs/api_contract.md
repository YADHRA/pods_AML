# API Contract

**SHARED — FastAPI, read-only GET, prefix `/api`. P2 owns implementation.**

Stub mode: `AML_USE_STUB=1` reads `results_stub/` + `data_stub/`.

## Endpoints

- `/overview`
- `/data/daily`
- `/data/splits`
- `/data/shortcuts`
- `/graph/degree`
- `/graph/hubs`
- `/graph/new-pair-lift`
- `/graph/subgraphs`
- `/graph/subgraphs/{id}`
- `/models/runs`
- `/models/runs/{run_id}`
- `/models/curves`
- `/models/permutation`
- `/models/shap`
- `/models/hard-subset`
- `/transactions?outcome=&model=`
- `/transactions/{txn_id}`

## Response shapes

Each endpoint returns the matching results file as-is (shapes in `docs/data_contract.md`), except where noted.

**Proposed by P1, P2 reviews.**

| Endpoint | Source | Notes |
|---|---|---|
| `/data/daily` | `eda/daily.json` | list |
| `/data/splits` | `eda/splits.json` | list |
| `/data/quality` (new) | `eda/data_quality.json` | proposed addition |
| `/data/drift` (new) | `eda/feature_drift.json` | proposed addition |
| `/data/shortcuts` | not a P1 file | owner to define |
| `/models/runs` | `metrics/*.json` | list of all 12 runs, one entry per run |
| `/models/runs/{run_id}` | `metrics/{run_id}.json` | 404 if unknown |
| `/models/curves?run_id=&split=` | `curves/{run_id}_{split}.json` | split = val, test or tail |
| `/models/permutation` | `permutation.json` | 4 runs |
| `/models/shap?run_id=` | `shap/{run_id}.json` | 404 for runs other than the 4 seed-42 runs |
| `/models/hard-subset` | `hard_subset.json` | optional `?run_id=` filters `runs` |
| `/transactions?outcome=&model=&limit=&offset=` | `transaction_inspector.parquet` | outcome in TP, FP, FN, TN; model in m1, m2 filters `outcome_m1` or `outcome_m2` |
| `/transactions/{txn_id}` | `transaction_inspector.parquet` | one row, 404 if not in the sample |