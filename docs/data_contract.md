# Data & Results Contract  (SHARED — change only via `contract/*` PR, both approve)

## Locked rules
- Split (inclusive): warmup Sep 1 · train Sep 2–6 · val Sep 7–8 · test Sep 9–10 · tail Sep 11–18 (report separately, never pooled).
- History rule: a day-D row uses only transactions with `date < D` (strictly earlier DAYS, not earlier minutes).
- `txn_id` = original 0-based CSV data-row index (int32), assigned right after load, never recomputed. Join ONLY on `txn_id` or `(date, node_id)`, never by row position.
- Node = (bank, account) as strings (bank codes have leading zeros). Self-loops stay as rows but are excluded from graph topology and account history.
- Exact duplicates (9) are audited, NOT deleted. Repeated timestamps are normal.

## Files  (real: `data/processed/`  ·  stub: `data_stub/`)
| File | Producer → Consumer | Key | Columns | Frozen |
|---|---|---|---|---|
| transactions_clean.parquet | P1 → P2 | txn_id | see `schema.py` (superset; P2 needs src_node, dst_node, date, split, is_self_loop, txn_id, is_laundering) · all 5,078,345 rows · sorted (ts, txn_id) | stub now; real ≈ H5 |
| node_map.parquet | P1 → P2 | node_id | node_id, bank, account | same |
| graph_node_day.parquet | P2 → P1 | (feature_date, node_id) | g_pagerank_log, g_wcc_size_log, g_n_unique_out_log, g_n_unique_in_log, g_has_sent_before, g_has_received_before · from date < feature_date, non-self edges | schema now; values ≈ H10 (tag `graph-v1`) |
| graph_pair_txn.parquet | P2 → P1 | txn_id | g_pair_seen_before, g_pair_prior_count_log, g_reverse_pair_seen · rows only for train/val/test/tail | same |
| matrix_model1/2.parquet | P1 (internal) | txn_id | txn_id, split, is_laundering + columns from `configs/feature_sets.yaml` | P1 only |

Assembly naming (done by P1's `assemble.py`): node columns are joined on `(date, src_node)` and `(date, dst_node)`; e.g. `g_pagerank_log` becomes `g_src_pagerank_log` / `g_dst_pagerank_log`.

## Results files  (`results/`, small, committed; stub in `results_stub/`)
Producer P1, consumer P2 (API). Exact JSON shapes are added to this file in Step 0.2 (P1 drafts, P2 reviews).
- `metrics/{run_id}.json`, `curves/{run_id}_{split}.json`, `thresholds.json`, `permutation.json`, `shap/{run_id}.json`
- `eda/*.json`, `hard_subset.json`, `transaction_inspector.parquet`
- P2 produces `results/graph/*.json` and `results/subgraphs/*.json`.
- `run_id` = `YYYYMMDD_m{1|2}_{full|noflags}_seed{N}`

## Change rule
Contract change = its own PR touching this file + `schema.py` (+ `feature_sets.yaml` if columns change). Both approve.
