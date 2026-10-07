import json

import numpy as np
import pandas as pd
import xgboost as xgb

from aml.common import paths
from aml.features import assemble as A
from aml.models import xgb as X


RUN_DATE = "20261007"
SEED = 42
KEY = f"m2_full_seed{SEED}"
RUN_ID = f"{RUN_DATE}_m2_full_seed{SEED}"

thresholds = json.loads((paths.RESULTS / "thresholds.json").read_text())
fs = A.load_feature_sets()
cols = X.feature_columns(fs, "m2", "full")
print(f"{len(cols)} features")


def margin_or_contribs(booster, x, **kw):
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


df = pd.read_parquet(paths.PROCESSED / "matrix_model2.parquet")
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype("category")

booster = xgb.Booster()
booster.load_model(str(paths.ROOT / "models" / f"{KEY}.json"))
thr = thresholds[KEY]["threshold"]

# Self-check 1: frozen validation F1.
val = df[df["split"] == "val"]
val_scores = margin_or_contribs(booster, val[cols])
f1 = f1_at(val["is_laundering"].astype(int).to_numpy(), val_scores, thr)
if abs(f1 - thresholds[KEY]["val_f1"]) > 1e-6:
    raise SystemExit(
        f"STOP: {KEY} val F1 {f1:.12f} != frozen {thresholds[KEY]['val_f1']:.12f}"
    )
print(f"{KEY}: self-check ok")

# Row set = inspector txn_ids (join on txn_id only).
insp = pd.read_parquet(
    paths.RESULTS / "transaction_inspector.parquet",
    columns=["txn_id", "score_m2"],
)
if not insp["txn_id"].is_unique:
    raise SystemExit("STOP: txn_id not unique in inspector")

test = df[df["split"] == "test"]
sel = test[test["txn_id"].isin(insp["txn_id"])].reset_index(drop=True)
if len(sel) != len(insp):
    raise SystemExit(
        f"STOP: inspector has {len(insp)} txn_ids, found {len(sel)} in test"
    )
print(f"rows: {len(sel)}")

x = sel[cols].copy()
contrib = margin_or_contribs(booster, x, pred_contribs=True)
margin = margin_or_contribs(booster, x, output_margin=True)
shap_vals = contrib[:, :-1]
base = contrib[:, -1]

# Self-check 2: additivity.
gap = float(np.abs(contrib.sum(axis=1) - margin).max())
if gap > 1e-3:
    raise SystemExit(f"STOP: SHAP does not add up to margin (gap {gap:.6f})")
if np.ptp(base) > 1e-6:
    raise SystemExit("STOP: base value differs between rows")
print(f"additivity gap: {gap:.6f}")

# Self-check 3: score matches inspector, by txn_id.
score = 1.0 / (1.0 + np.exp(-margin.astype(float)))
ref = insp.set_index("txn_id")["score_m2"].reindex(sel["txn_id"]).to_numpy()
sgap = float(np.abs(score - ref).max())
if sgap > 1e-4:
    raise SystemExit(f"STOP: score differs from inspector (gap {sgap:.6f})")
print(f"inspector score gap: {sgap:.8f}")

# Long format.
txn = sel["txn_id"].to_numpy()
pieces = []
for j, f in enumerate(cols):
    col = sel[f]
    if str(col.dtype) == "category":
        fv = np.full(len(sel), np.nan)
        fvs = col.astype(str).to_numpy(dtype=object)
    else:
        fv = col.astype(float).to_numpy()
        fvs = np.full(len(sel), None, dtype=object)
    pieces.append(
        pd.DataFrame(
            {
                "txn_id": txn,
                "feature": f,
                "feature_value": fv,
                "feature_value_str": fvs,
                "shap_value": shap_vals[:, j].astype("float32"),
            }
        )
    )
out_df = pd.concat(pieces, ignore_index=True)
out_df["txn_id"] = out_df["txn_id"].astype("int32")
out_df["feature"] = out_df["feature"].astype("category")
out_df = out_df.sort_values(["txn_id", "feature"]).reset_index(drop=True)

out = paths.RESULTS / "shap_by_txn_m2_full_seed42.parquet"
out_df.to_parquet(out, index=False)

meta = {
    "run_id": RUN_ID,
    "model": "m2",
    "variant": "full",
    "seed": SEED,
    "split": "test",
    "space": "log-odds (XGBoost pred_contribs, TreeSHAP)",
    "base_value": float(base[0]),
    "n_txn": int(len(sel)),
    "n_features": len(cols),
    "rows": int(len(out_df)),
    "row_set": "same txn_ids as transaction_inspector.parquet (biased sample)",
    "note": (
        "shap_value summed over features + base_value = raw model score "
        "(log-odds); sigmoid gives the probability score. "
        "Categorical features have feature_value = NaN and the label in "
        "feature_value_str. Never compute rates from this sample."
    ),
}
(paths.RESULTS / "shap_by_txn_meta.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {out} ({len(out_df)} rows, {out.stat().st_size/1e6:.1f} MB)")