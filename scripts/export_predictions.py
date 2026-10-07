import json

import numpy as np
import pandas as pd
import xgboost as xgb

from aml.common import paths
from aml.features import assemble as A
from aml.models import xgb as X

SEED = 42
SPLITS = ["val", "test", "tail"]

thresholds = json.loads((paths.RESULTS / "thresholds.json").read_text())
fs = A.load_feature_sets()


def predict(booster, x):
    d = xgb.DMatrix(x, enable_categorical=True)
    return booster.predict(
        d, iteration_range=(0, int(booster.best_iteration) + 1)
    )


def f1_at(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    n_pred = int(pred.sum())
    n_pos = int(y.sum())
    pr = tp / n_pred if n_pred else 0.0
    rc = tp / n_pos if n_pos else 0.0
    return 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0


meta = None
scores = {}
thr_used = {}

for model_name in ["m2", "m1"]:
    matrix = paths.PROCESSED / (
        "matrix_model1.parquet" if model_name == "m1" else "matrix_model2.parquet"
    )
    print(f"\nLoading {matrix.name}...")
    df = pd.read_parquet(matrix)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")
    df = df[df["split"].isin(SPLITS)].reset_index(drop=True)

    if meta is None:  # M2 matrix defines the row set and the new-pair flag
        meta = pd.DataFrame(
            {
                "txn_id": df["txn_id"].to_numpy(),
                "split": df["split"].astype(str).to_numpy(),
                "is_laundering": df["is_laundering"].astype(int).to_numpy(),
                "new_pair": (df["g_pair_seen_before"] == 0).to_numpy(),
            }
        )
        if not meta["txn_id"].is_unique:
            raise SystemExit("STOP: txn_id not unique")

    val_mask = (df["split"] == "val").to_numpy()
    y_val = df.loc[val_mask, "is_laundering"].astype(int).to_numpy()

    for variant in ["full", "noshortcut"]:
        file_variant = "noflags" if variant == "noshortcut" else variant
        label = f"{model_name}_{file_variant}"
        key = f"{model_name}_{variant}_seed{SEED}"
        thr = thresholds[key]["threshold"]

        booster = xgb.Booster()
        booster.load_model(str(paths.ROOT / "models" / f"{key}.json"))
        cols = X.feature_columns(fs, model_name, variant)
        p = predict(booster, df[cols])

        f1 = f1_at(y_val, p[val_mask], thr)
        if abs(f1 - thresholds[key]["val_f1"]) > 1e-6:
            raise SystemExit(
                f"STOP: {key} val F1 {f1:.12f} != frozen {thresholds[key]['val_f1']:.12f}"
            )

        scores[label] = pd.Series(p.astype("float32"), index=df["txn_id"].to_numpy())
        thr_used[label] = thr
        print(f"{label}: self-check ok")

out = meta.copy()
for label, s in scores.items():
    v = s.reindex(out["txn_id"]).to_numpy()
    if np.isnan(v).any():
        raise SystemExit(f"STOP: {label} missing txn_ids")
    out[f"score_{label}"] = v
    out[f"alert_{label}"] = v >= thr_used[label]

out = out.sort_values("txn_id").reset_index(drop=True)
out["txn_id"] = out["txn_id"].astype("int32")

path = paths.RESULTS / "predictions_seed42.parquet"
out.to_parquet(path, index=False, compression="zstd")
print(f"\nwrote {path}: {len(out)} rows, {path.stat().st_size / 1e6:.1f} MB")
print(out.groupby("split")["is_laundering"].agg(["size", "sum"]))
print(out.groupby("split")[[c for c in out.columns if c.startswith("alert_")]].sum())