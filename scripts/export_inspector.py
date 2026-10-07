import json

import numpy as np
import pandas as pd
import xgboost as xgb

from aml.common import paths
from aml.features import assemble as A
from aml.models import xgb as X

SEED = 42
N_NEG = 20000
SPLIT = "test"

thresholds = json.loads((paths.RESULTS / "thresholds.json").read_text())
fs = A.load_feature_sets()


def predict(booster, x, **kw):
    d = xgb.DMatrix(x, enable_categorical=True)
    return booster.predict(
        d, iteration_range=(0, int(booster.best_iteration) + 1), **kw
    )


def f1_at(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    n_pred = int(pred.sum())
    n_pos = int(y.sum())
    pr = tp / n_pred if n_pred else 0.0
    rc = tp / n_pos if n_pos else 0.0
    return 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0


def outcome(y, alert):
    return np.select(
        [alert & (y == 1), alert & (y == 0), ~alert & (y == 1)],
        ["TP", "FP", "FN"],
        default="TN",
    )


def load(model_name):
    key = f"{model_name}_full_seed{SEED}"
    matrix = paths.PROCESSED / (
        "matrix_model1.parquet" if model_name == "m1" else "matrix_model2.parquet"
    )
    df = pd.read_parquet(matrix)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")
    cols = X.feature_columns(fs, model_name, "full")
    booster = xgb.Booster()
    booster.load_model(str(paths.ROOT / "models" / f"{key}.json"))
    thr = thresholds[key]["threshold"]

    val = df[df["split"] == "val"]
    f1 = f1_at(val["is_laundering"].astype(int).to_numpy(), predict(booster, val[cols]), thr)
    if abs(f1 - thresholds[key]["val_f1"]) > 1e-6:
        raise SystemExit(f"STOP: {key} val F1 {f1:.12f} != frozen {thresholds[key]['val_f1']:.12f}")

    test = df[df["split"] == SPLIT].reset_index(drop=True)
    scores = predict(booster, test[cols])
    print(f"{key}: self-check ok")
    return booster, cols, test, scores, thr


b1, c1, t1, s1, thr1 = load("m1")
b2, c2, t2, s2, thr2 = load("m2")

# align M1 scores to M2 rows on txn_id only
s1_map = pd.Series(s1, index=t1["txn_id"].to_numpy())
score_m1 = s1_map.reindex(t2["txn_id"].to_numpy()).to_numpy()
if np.isnan(score_m1).any():
    raise SystemExit("STOP: some test txn_ids missing in M1 matrix")

y = t2["is_laundering"].astype(int).to_numpy()
alert1 = score_m1 >= thr1
alert2 = s2 >= thr2

must_keep = (y == 1) | alert1 | alert2
rng = np.random.default_rng(SEED)
pool = np.flatnonzero(~must_keep)
neg_idx = rng.choice(pool, size=min(N_NEG, len(pool)), replace=False)
keep = must_keep.copy()
keep[neg_idx] = True

reason = np.where(y == 1, "positive", np.where(alert1 | alert2, "alert", "neg_sample"))

# SHAP top 3 for M2, only on kept rows
contrib = predict(b2, t2.loc[keep, c2], pred_contribs=True)[:, :-1]
top = np.argsort(-np.abs(contrib), axis=1)[:, :3]
names = np.array(c2)

out = pd.DataFrame(
    {
        "txn_id": t2.loc[keep, "txn_id"].to_numpy(),
        "is_laundering": y[keep],
        "score_m1": score_m1[keep],
        "score_m2": s2[keep],
        "outcome_m1": outcome(y, alert1)[keep],
        "outcome_m2": outcome(y, alert2)[keep],
        "reason": reason[keep],
    }
)
for k in range(3):
    out[f"shap{k+1}_feature"] = names[top[:, k]]
    out[f"shap{k+1}_value"] = np.take_along_axis(contrib, top[:, [k]], axis=1).ravel()

raw = pd.read_parquet(
    paths.PROCESSED / "transactions_clean.parquet",
    columns=["txn_id", "ts", "src_node", "dst_node", "src_bank", "dst_bank",
             "amount_paid", "cur_paid", "amount_received", "cur_recv", "payment_format"],
)
out = out.merge(raw, on="txn_id", how="left", validate="one_to_one")
if out["ts"].isna().any():
    raise SystemExit("STOP: join to transactions_clean lost rows")

out["split"] = SPLIT
out["thr_m1"] = thr1
out["thr_m2"] = thr2

path = paths.RESULTS / "transaction_inspector.parquet"
out.to_parquet(path, index=False)
print(f"wrote {path}: {len(out)} rows")
print(out["reason"].value_counts())
print(pd.crosstab(out["outcome_m1"], out["outcome_m2"]))