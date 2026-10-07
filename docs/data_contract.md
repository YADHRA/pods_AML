\# Data \& Results Contract



\*\*SHARED — change only via `contract/\*` PR, both approve\*\*



\## Locked rules



\- Split (inclusive): warmup Sep 1 · train Sep 2–6 · val Sep 7–8 · test Sep 9–10 · tail Sep 11–18 (report separately, never pooled).

\- History rule: a day-D row uses only transactions with `date < D` (strictly earlier DAYS, not earlier minutes).

\- `txn\_id` = original 0-based CSV data-row index (int32), assigned right after load, never recomputed. Join ONLY on `txn\_id` or `(date, node\_id)`, never by row position.

\- Node = (bank, account) as strings (bank codes have leading zeros). Self-loops stay as rows but are excluded from graph topology and account history.

\- Exact duplicates (9) are audited, NOT deleted. Repeated timestamps are normal.



\## Files



| File | Producer → Consumer | Key | Columns | Frozen |

|---|---|---|---|---|

| transactions\_clean.parquet | P1 → P2 | txn\_id | see `schema.py` (superset; P2 needs src\_node, dst\_node, date, split, is\_self\_loop, txn\_id, is\_laundering) · all 5,078,345 rows · sorted (ts, txn\_id) | real |

| node\_map.parquet | P1 → P2 | node\_id | node\_id, bank, account | same |

| graph\_node\_day.parquet | P2 → P1 | (feature\_date, node\_id) | g\_pagerank\_log, g\_wcc\_size\_log, g\_n\_unique\_out\_log, g\_n\_unique\_in\_log, g\_has\_sent\_before, g\_has\_received\_before · from date < feature\_date, non-self edges | real |

| graph\_pair\_txn.parquet | P2 → P1 | txn\_id | g\_pair\_seen\_before, g\_pair\_prior\_count\_log, g\_reverse\_pair\_seen | real |



\## Results file shapes



All files are JSON unless noted.



Six rules to know:



\- The run-id variant is `noflags`, but `thresholds.json` keys use `noshortcut` (`m2\_noshortcut\_seed42`). The meaning is the same.

\- Split names are `val`, `test`, `tail` everywhere, except `metrics/{run\_id}.json`, which uses the full word `validation`.

\- Tail is always reported separately, never pooled.

\- Prevalence differs by split and by subset, so PR-AUC is only comparable within the same split and subset.

\- Only some files exist for all 12 runs.

\- `transaction\_inspector.parquet` is a biased sample. Never compute rates from it.



\### thresholds.json



Object keyed:



`{m1|m2}\_{full|noshortcut}\_seed{N}`



There are 12 keys.



Each value contains:



`rule`, `threshold`, `val\_f1`, `val\_n\_alerts`, `val\_precision`, `val\_recall`.



\### metrics/{run\_id}.json



There are 12 files.



Top-level fields:



`run\_id`, `model`, `variant`, `seed`, `n\_features`, `features`, `threshold`, `validation`, `test`, `tail`.



`features` is a list of strings.



`threshold` contains:



`value`, `rule`, `frozen\_on`.



Each of `validation`, `test`, and `tail` contains:



`n`, `n\_pos`, `prevalence`, `pr\_auc`, `lift`, `roc\_auc`, `threshold`, `precision`, `recall`, `f1`, `tn`, `fp`, `fn`, `tp`, `n\_alerts`, `accuracy`, `budget\_frac`, `budget\_k`, `recall\_at\_budget`, `precision\_at\_budget`.



\### curves/{run\_id}\_{split}.json



There are 36 files: 12 runs × `val`, `test`, `tail`.



Top-level fields:



`run\_id`, `split`, `n`, `n\_pos`, `prevalence`, `pr`, `roc`, `budget`, `operating\_point`.



`pr` contains `recall` and `precision` arrays of equal length.



`roc` contains `fpr` and `tpr` arrays of equal length.



The curve arrays contain at most 300 points and are rounded to 5 decimals.



`budget` is a list of 8 objects containing:



`frac`, `k`, `recall`, `precision`.



`operating\_point` contains:



`threshold`, `n\_alerts`, `precision`, `recall`.



\### permutation.json



There are 4 runs, seed 42 only.



Top-level fields:



`metric`, `split`, `n\_repeats`, `note`, `runs`.



`runs` is keyed by run ID.



Each run contains:



`baseline\_pr\_auc\_weighted`, `n\_rows`, `n\_positive`, `n\_negative\_sampled`, `features`.



Each `features` item contains:



`feature`, `importance\_mean`, `importance\_std`.



Runs:



\- `20261007\_m1\_full\_seed42`

\- `20261007\_m1\_noflags\_seed42`

\- `20261007\_m2\_full\_seed42`

\- `20261007\_m2\_noflags\_seed42`



Negatives were sampled, so the baseline is weighted and differs from the PR-AUC in `metrics`.



\### shap/{run\_id}.json



There are 4 runs, seed 42 only.



Top-level fields:



`run\_id`, `split`, `space`, `base\_value`, `global`, `sample`.



Each `global` item contains:



`feature`, `mean\_abs\_all`, `mean\_abs\_fraud`.



`sample` contains:



`label`, `features`, `categorical\_features\_as\_codes`, `values`, `shap`.



The sample contains 600 rows.



`values` and `shap` have shape `600 × n\_features`.



Categorical values in `values` are integer category codes.



\### hard\_subset.json



Top-level fields:



`rule`, `note`, `runs`.



`rule` is:



`g\_pair\_seen\_before == 0`.



There are 12 runs.



Each run contains:



`run\_id`, `model`, `variant`, `seed`, `splits`.



`splits` contains `val`, `test`, and `tail`.



Each split contains:



`n`, `n\_pos`, `base\_rate`, `pr\_auc`, `threshold`, `n\_alerts`, `precision`, `recall`, `f1`, `informative`.



`tail.informative` is false because 1,076 of the 1,108 tail rows are new pairs.



Thresholds are the frozen thresholds.



The subset base rate is about 1.4%, much higher than the overall test rate of about 0.11%, so subset PR-AUC must not be compared directly with overall PR-AUC.



\### transaction\_inspector.parquet



Test split only, using M1 full seed 42 and M2 full seed 42.



It contains every positive, every alert from either model, and 20,000 random true negatives (seed 42).



It is not a random sample, so rates must not be computed from it.



Columns:



`txn\_id`, `is\_laundering`, `score\_m1`, `score\_m2`,

`outcome\_m1`, `outcome\_m2`, `reason`,

`shap1\_feature`, `shap1\_value`,

`shap2\_feature`, `shap2\_value`,

`shap3\_feature`, `shap3\_value`,

`ts`, `src\_node`, `dst\_node`,

`src\_bank`, `dst\_bank`,

`amount\_paid`, `cur\_paid`,

`amount\_received`, `cur\_recv`,

`payment\_format`, `split`,

`thr\_m1`, `thr\_m2`.



The three SHAP feature/value pairs are the top 3 M2 SHAP contributions by absolute value, in log-odds space.



\### eda/\*.json



\#### splits.json



List of 5 objects, one per split.



Each object:



`split`, `n\_rows`, `n\_pos`, `date\_min`, `date\_max`, `pos\_rate`.



\#### daily.json



List of 18 objects, one per day.



Each object:



`date`, `n\_rows`, `n\_pos`, `pos\_rate`.



\#### feature\_drift.json



List of 33 objects.



Each object:



`feature`, `ks`, `nan\_share\_train`, `nan\_share\_val`, `mean\_train`, `mean\_val`.



This contains numeric/binary features only, comparing train vs validation.



The two M2 categorical features, `t\_payment\_format` and `t\_cur\_paid`, are intentionally not included.



\#### data\_quality.json



Top-level fields:



`n\_rows`, `n\_positives`, `positive\_rate`, `nulls`,

`ts\_min`, `ts\_max`, `n\_distinct\_timestamps`,

`raw\_is\_chronological`, `rows\_stepping\_backwards\_in\_time`,

`exact\_duplicate\_extra\_rows`,

`exact\_duplicate\_extra\_rows\_laundering`,

`currencies`, `payment\_formats`,

`amount\_paid\_min`, `amount\_paid\_max`,

`amount\_paid\_nonpositive`, `amount\_paid\_le\_0.01`,

`fx\_rows`, `fx\_rows\_positive`,

`non\_fx\_rows\_with\_unequal\_amounts`,

`bank\_codes\_have\_leading\_zeros`.



`nulls` is an object mapping column names to null counts.



`currencies` and `payment\_formats` are lists of strings.

