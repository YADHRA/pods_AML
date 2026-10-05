import json
import time

import pandas as pd

from aml.common import paths
from aml.features import assemble as A

t0 = time.time()
P = paths.PROCESSED
fs = A.load_feature_sets()
m1_cols, m2_cols = A.model_columns(fs)

clean = pd.read_parquet(P / "transactions_clean.parquet",
                        columns=["txn_id", "ts", "date", "src_node", "dst_node", "is_laundering", "split"])
txf = pd.read_parquet(P / "txn_features.parquet")
acc = pd.read_parquet(P / "account_node_day.parquet")

pair_f, gnd_f = P / "graph_pair_txn.parquet", P / "graph_node_day.parquet"
if pair_f.exists() != gnd_f.exists():
    raise SystemExit("Only one of graph_pair_txn / graph_node_day exists in data/processed - need both.")
have_graph = pair_f.exists()
pair = pd.read_parquet(pair_f) if have_graph else None
gnd = pd.read_parquet(gnd_f) if have_graph else None
print(f"loaded inputs ({time.time() - t0:.0f}s); graph tables: {'yes' if have_graph else 'NOT YET (building Model 1 only)'}")

wide = A.build_wide(clean, txf, acc, pair, gnd, fs)
A.check_counts(wide)
print("split counts match the locked numbers")

g = wide.groupby("split")["is_laundering"].agg(["size", "sum"])
g["prevalence_%"] = (100 * g["sum"] / g["size"]).round(3)
print(g.loc[["train", "val", "test", "tail"]].to_string())

m1 = wide[A.ID_COLS + m1_cols]
m1.to_parquet(P / "matrix_model1.parquet", index=False)
print("matrix_model1:", m1.shape)
if have_graph:
    m2 = wide[A.ID_COLS + m2_cols]
    assert (m1["txn_id"].to_numpy() == m2["txn_id"].to_numpy()).all()
    m2.to_parquet(P / "matrix_model2.parquet", index=False)
    print("matrix_model2:", m2.shape)

# train-vs-val drift on every numeric feature we have so far
feat_cols = [c for c in wide.columns if c not in A.ID_COLS]
rows = A.drift_report(wide, feat_cols)
out = paths.RESULTS / "eda" / "feature_drift.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2))
print("\nTop drift (KS: 0 = same distribution, 1 = completely different):")
print(pd.DataFrame(rows).head(10).round(3).to_string(index=False))
print(f"\nwrote {out}  (total {time.time() - t0:.0f}s)")