# Data & Results Contract

**SHARED — change only via `contract/*` PR, both approve**

## Locked rules

- Split (inclusive): warmup Sep 1 · train Sep 2–6 · val Sep 7–8 · test Sep 9–10 · tail Sep 11–18 (report separately, never pooled).

- History rule: a day-D row uses only transactions with `date < D` (strictly earlier DAYS, not earlier minutes).

- `txn_id` = original 0-based CSV data-row index (int32), assigned right after load, never recomputed. Join ONLY on `txn_id` or `(date, node_id)`, never by row position.

- Node = (bank, account) as strings (bank codes have leading zeros). Self-loops stay as rows but are excluded from graph topology and account history.

- Exact duplicates (9) are audited, NOT deleted. Repeated timestamps are normal.

## Files

| File | Producer → Consumer | Key | Columns | Frozen |

|---|---|---|---|---|

| transactions_clean.parquet | P1 → P2 | txn_id | see `schema.py` (superset; P2 needs src_node, dst_node, date, split, is_self_loop, txn_id, is_laundering) · all 5,078,345 rows · sorted (ts, txn_id) | real |

| node_map.parquet | P1 → P2 | node_id | node_id, bank, account | same |

| graph_node_day.parquet | P2 → P1 | (feature_date, node_id) | g_pagerank_log, g_wcc_size_log, g_n_unique_out_log, g_n_unique_in_log, g_has_sent_before, g_has_received_before · from date < feature_date, non-self edges | real |

| graph_pair_txn.parquet | P2 → P1 | txn_id | g_pair_seen_before, g_pair_prior_count_log, g_reverse_pair_seen | real |

## Results file shapes

All files are JSON unless noted.

Six rules to know:

- The run-id variant is `noflags`, but `thresholds.json` keys use `noshortcut` (`m2_noshortcut_seed42`). The meaning is the same.

- Split names are `val`, `test`, `tail` everywhere, except `metrics/{run_id}.json`, which uses the full word `validation`.

- Tail is always reported separately, never pooled.

- Prevalence differs by split and by subset, so PR-AUC is only comparable within the same split and subset.

- Only some files exist for all 12 runs.

- `transaction_inspector.parquet` is a biased sample. Never compute rates from it.

### thresholds.json

Object keyed:

`{m1|m2}_{full|noshortcut}_seed{N}`

There are 12 keys.

Each value contains:

`rule`, `threshold`, `val_f1`, `val_n_alerts`, `val_precision`, `val_recall`.

### metrics/{run_id}.json

There are 12 files.

Top-level fields:

`run_id`, `model`, `variant`, `seed`, `n_features`, `features`, `threshold`, `validation`, `test`, `tail`.

`features` is a list of strings.

`threshold` contains:

`value`, `rule`, `frozen_on`.

Each of `validation`, `test`, and `tail` contains:

`n`, `n_pos`, `prevalence`, `pr_auc`, `lift`, `roc_auc`, `threshold`, `precision`, `recall`, `f1`, `tn`, `fp`, `fn`, `tp`, `n_alerts`, `accuracy`, `budget_frac`, `budget_k`, `recall_at_budget`, `precision_at_budget`.

### curves/{run_id}_{split}.json

There are 36 files: 12 runs × `val`, `test`, `tail`.

Top-level fields:

`run_id`, `split`, `n`, `n_pos`, `prevalence`, `pr`, `roc`, `budget`, `operating_point`.

`pr` contains `recall` and `precision` arrays of equal length.

`roc` contains `fpr` and `tpr` arrays of equal length.

The curve arrays contain at most 300 points and are rounded to 5 decimals.

`budget` is a list of 8 objects containing:

`frac`, `k`, `recall`, `precision`.

`operating_point` contains:

`threshold`, `n_alerts`, `precision`, `recall`.

### permutation.json

There are 4 runs, seed 42 only.

Top-level fields:

`metric`, `split`, `n_repeats`, `note`, `runs`.

`runs` is keyed by run ID.

Each run contains:

`baseline_pr_auc_weighted`, `n_rows`, `n_positive`, `n_negative_sampled`, `features`.

Each `features` item contains:

`feature`, `importance_mean`, `importance_std`.

Runs:

- `20261007_m1_full_seed42`

- `20261007_m1_noflags_seed42`

- `20261007_m2_full_seed42`

- `20261007_m2_noflags_seed42`

Negatives were sampled, so the baseline is weighted and differs from the PR-AUC in `metrics`.

### shap/{run_id}.json

There are 4 runs, seed 42 only.

Top-level fields:

`run_id`, `split`, `space`, `base_value`, `global`, `sample`.

Each `global` item contains:

`feature`, `mean_abs_all`, `mean_abs_fraud`.

`sample` contains:

`label`, `features`, `categorical_features_as_codes`, `values`, `shap`.

The sample contains 600 rows.

`values` and `shap` have shape `600 × n_features`.

Categorical values in `values` are integer category codes.

### hard_subset.json

Top-level fields:

`rule`, `note`, `runs`.

`rule` is:

`g_pair_seen_before == 0`.

There are 12 runs.

Each run contains:

`run_id`, `model`, `variant`, `seed`, `splits`.

`splits` contains `val`, `test`, and `tail`.

Each split contains:

`n`, `n_pos`, `base_rate`, `pr_auc`, `threshold`, `n_alerts`, `precision`, `recall`, `f1`, `informative`.

`tail.informative` is false because 1,076 of the 1,108 tail rows are new pairs.

Thresholds are the frozen thresholds.

The subset base rate is about 1.4%, much higher than the overall test rate of about 0.11%, so subset PR-AUC must not be compared directly with overall PR-AUC.

### transaction_inspector.parquet

Test split only, using M1 full seed 42 and M2 full seed 42.

It contains every positive, every alert from either model, and 20,000 random true negatives (seed 42).

It is not a random sample, so rates must not be computed from it.

Columns:

`txn_id`, `is_laundering`, `score_m1`, `score_m2`,

`outcome_m1`, `outcome_m2`, `reason`,

`shap1_feature`, `shap1_value`,

`shap2_feature`, `shap2_value`,

`shap3_feature`, `shap3_value`,

`ts`, `src_node`, `dst_node`,

`src_bank`, `dst_bank`,

`amount_paid`, `cur_paid`,

`amount_received`, `cur_recv`,

`payment_format`, `split`,

`thr_m1`, `thr_m2`.

The three SHAP feature/value pairs are the top 3 M2 SHAP contributions by absolute value, in log-odds space.

### predictions_seed42.parquet

Per-transaction scores for seed 42 only, for val, test, and tail. Join on `txn_id` only.

Columns:

`txn_id`, `split`, `is_laundering`, `new_pair`, `score_m2_full`, `score_m2_noflags`, `score_m1_full`, `score_m1_noflags`, `alert_m2_full`, `alert_m2_noflags`, `alert_m1_full`, `alert_m1_noflags`.

There are 1,829,424 rows: val 965,524 (1,036 positives), test 862,792 (956), tail 1,108 (655).

Alerts use the frozen validation thresholds from `thresholds.json`. Scores are probabilities, not log-odds.

Tail is reported separately and never pooled with val or test.

### shap_by_txn_m2_full_seed42.parquet

Per-transaction SHAP for M2 full, seed 42, on test rows only. The rows are the same `txn_id`s as `transaction_inspector.parquet`, so this is a biased sample and no rates may be computed from it.

Long format, one row per `txn_id` and feature (22,370 transactions x 35 features = 782,950 rows).

Columns:

`txn_id`, `feature`, `feature_value`, `feature_value_str`, `shap_value`.

Categorical features have `feature_value` = NaN and the label in `feature_value_str`. Numeric features have `feature_value_str` = null. Some numeric `feature_value` cells are NaN because the value is missing.

`shap_value` is in log-odds (XGBoost `pred_contribs`, TreeSHAP). The sum of `shap_value` over the 35 features plus `base_value` equals the raw model score. A sigmoid of that sum gives the probability score.

### shap_by_txn_meta.json

Describes the file above.

Fields:

`run_id`, `model`, `variant`, `seed`, `split`, `space`, `base_value`, `n_txn`, `n_features`, `rows`, `row_set`, `note`.

`base_value` is the same for every transaction.

### comparison.json

M1 vs M2, computed from the 12 metrics files.

Top-level fields:

`note`, `seeds`, `variants`.

`variants` has `full` and `noflags`. Each contains `val`, `test`, and `tail`. Each split contains `n`, `n_pos`, and `metrics`.

`metrics` has one entry per metric:

`pr_auc`, `roc_auc`, `precision`, `recall`, `f1`, `lift`, `recall_at_budget`, `precision_at_budget`, `tp`, `fp`, `fn`, `tn`, `n_alerts`.

Each metric has `m1`, `m2`, and `delta`. Each of these has `mean`, `std`, and `per_seed` (keys "42", "43", "44").

`delta` is M2 minus M1, computed per seed and then averaged. `std` uses 3 seeds, so it is rough.

Tail has about 59% fraud, so its PR-AUC and lift are mostly a base-rate effect.

### hard_subset_ach.json

Second hard subset. It has the same shape as `hard_subset.json`, with one extra field per split.

`rule` is:

`payment_format == "ACH" and not self-loop and not FX`.

There are 12 runs. Each run contains `run_id`, `model`, `variant`, `seed`, `splits`.

Each split contains the `hard_subset.json` fields plus `low_positive`, which is true when the split has fewer than 30 positives.

Subset sizes (rows / positives): val 101,311 / 890, test 105,812 / 832, tail 714 / 652.

`tail.informative` is false. The subset base rate is far above the overall rate, so subset PR-AUC must not be compared directly with overall PR-AUC.

This subset is different from the new-pair subset in `hard_subset.json`.

### Model files

Not in git. Shared outside the repo as `models/{m1|m2}_{full|noshortcut}_seed{N}.json` (XGBoost Booster JSON). Note that `thresholds.json` keys use `noshortcut`, while `run_id` and metrics file names use `noflags`.

### eda/*.json


#### splits.json

List of 5 objects, one per split.

Each object:

`split`, `n_rows`, `n_pos`, `date_min`, `date_max`, `pos_rate`.

#### daily.json

List of 18 objects, one per day.

Each object:

`date`, `n_rows`, `n_pos`, `pos_rate`.

#### feature_drift.json

List of 33 objects.

Each object:

`feature`, `ks`, `nan_share_train`, `nan_share_val`, `mean_train`, `mean_val`.

This contains numeric/binary features only, comparing train vs validation.

The two M2 categorical features, `t_payment_format` and `t_cur_paid`, are intentionally not included.

#### data_quality.json

Top-level fields:

`n_rows`, `n_positives`, `positive_rate`, `nulls`,

`ts_min`, `ts_max`, `n_distinct_timestamps`,

`raw_is_chronological`, `rows_stepping_backwards_in_time`,

`exact_duplicate_extra_rows`,

`exact_duplicate_extra_rows_laundering`,

`currencies`, `payment_formats`,

`amount_paid_min`, `amount_paid_max`,

`amount_paid_nonpositive`, `amount_paid_le_0.01`,

`fx_rows`, `fx_rows_positive`,

`non_fx_rows_with_unequal_amounts`,

`bank_codes_have_leading_zeros`.

`nulls` is an object mapping column names to null counts.

`currencies` and `payment_formats` are lists of strings.

