import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from aml.common import paths
from aml.features import assemble as A
from aml.models import xgb as X

RUN_DATE = "20261007"
SEED = 42
N_RANDOM = 50_000
N_SAMPLE_EACH = 300

thresholds = json.loads((paths.RESULTS / "thresholds.json").read_text())
fs = A.load_feature_sets()
rng = np.random.default_rng(1)
out_dir = paths.RESULTS / "shap"
out_dir.mkdir(parents=True, exist_ok=True)


def prep(part, cols):
    x = part[cols].copy().reset_index(drop=True)
    for c in cols:
        if x[c].dtype == object:
            x[c] = x[c].astype("category")
    return x


def kw(booster, use_best):
    if use_best and booster.attr("best_iteration") is not None:
        return {"iteration_range": (0, int(booster.attr("best_iteration")) + 1)}
    return {}


def f1_at(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    prec = tp / pred.sum() if pred.sum() else 0.0
    rec = tp / y.sum()
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


for model_name in ["m1", "m2"]:
    matrix = paths.PROCESSED / ("matrix_model1.parquet" if model_name == "m1" else "matrix_model2.parquet")
    df = pd.read_parquet(matrix)
    val = df[df["split"] == "val"]
    y = val["is_laundering"].values
    for variant in ["full", "noshortcut"]:
        cols = X.feature_columns(fs, model_name, variant)
        key = f"{model_name}_{variant}_seed{SEED}"
        file_variant = "noflags" if variant == "noshortcut" else variant
        run_id = f"{RUN_DATE}_{model_name}_{file_variant}_seed{SEED}"
        thr = thresholds[key]["threshold"]
        booster = xgb.Booster()
        booster.load_model(str(paths.ROOT / "models" / f"{key}.json"))

        x_full = prep(val, cols)
        d_full = xgb.DMatrix(x_full, enable_categorical=True)
        use_best = False
        p = booster.predict(d_full)
        if abs(f1_at(y, p, thr) - thresholds[key]["val_f1"]) > 0.005:
            use_best = True
            p = booster.predict(d_full, **kw(booster, True))
        if abs(f1_at(y, p, thr) - thresholds[key]["val_f1"]) > 0.005:
            sys.exit(f"STOP: {key} does not reproduce its frozen val F1.")

        def contribs(idx):
            d = xgb.DMatrix(x_full.iloc[idx], enable_categorical=True)
            c = booster.predict(d, pred_contribs=True, **kw(booster, use_best))
            m = booster.predict(d, output_margin=True, **kw(booster, use_best))
            return c, m

        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        c_rand, _ = contribs(np.sort(rng.choice(len(y), N_RANDOM, replace=False)))
        c_pos, _ = contribs(pos_idx)
        s_idx = np.r_[rng.choice(pos_idx, N_SAMPLE_EACH, replace=False),
                      rng.choice(neg_idx, N_SAMPLE_EACH, replace=False)]
        c_s, margin_s = contribs(s_idx)
        gap = float(np.abs(c_s.sum(axis=1) - margin_s).max())
        if gap > 1e-3:
            sys.exit(f"STOP: {key} SHAP does not add up to model score (gap {gap:.5f}).")

        g_all = np.abs(c_rand[:, :-1]).mean(axis=0)
        g_pos = np.abs(c_pos[:, :-1]).mean(axis=0)
        glob = [{"feature": c, "mean_abs_all": round(float(a), 5), "mean_abs_fraud": round(float(b), 5)}
                for c, a, b in zip(cols, g_all, g_pos)]
        glob.sort(key=lambda r: -r["mean_abs_fraud"])

        xs = x_full.iloc[s_idx]
        cat_cols = [c for c in cols if str(xs[c].dtype) == "category"]
        values = np.column_stack([xs[c].cat.codes.values if c in cat_cols else xs[c].values.astype(float)
                                  for c in cols])
        doc = {
            "run_id": run_id, "split": "val",
            "space": "log-odds (XGBoost pred_contribs, TreeSHAP)",
            "base_value": round(float(c_s[0, -1]), 5),
            "global": glob,
            "sample": {
                "label": [1] * N_SAMPLE_EACH + [0] * N_SAMPLE_EACH,
                "features": cols,
                "categorical_features_as_codes": cat_cols,
                "values": np.round(values, 4).tolist(),
                "shap": np.round(c_s[:, :-1], 4).tolist(),
            },
        }
        (out_dir / f"{run_id}.json").write_text(json.dumps(doc))
        print(f"\n{run_id}: additivity gap {gap:.6f} | top 8 by mean |SHAP| on fraud rows:")
        for r in glob[:8]:
            print(f"  {r['feature']:28s} fraud {r['mean_abs_fraud']:.3f} | all {r['mean_abs_all']:.3f}")

print(f"\nwrote 4 files to {out_dir}")
