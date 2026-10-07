import json

import numpy as np
import pandas as pd
import xgboost as xgb

from aml.common import paths
from aml.features import assemble as A
from aml.models import xgb as X


RUN_DATE = "20261007"
SEEDS = [42, 43, 44]
SPLITS = ["val", "test", "tail"]
RULE = 'payment_format == "ACH" and not self-loop and not FX'
LOW_POS = 30

# Sizes from the sizing command (must match, or we stop)
EXPECTED = {
    "val": (101311, 890),
    "test": (105812, 832),
    "tail": (714, 652),
}

thresholds = json.loads((paths.RESULTS / "thresholds.json").read_text())
fs = A.load_feature_sets()

# Subset flag lives in matrix_model1; join on txn_id only.
m1 = pd.read_parquet(
    paths.PROCESSED / "matrix_model1.parquet",
    columns=["txn_id", "t_payment_format", "t_is_self_loop", "t_is_fx"],
)
if not m1["txn_id"].is_unique:
    raise SystemExit("STOP: txn_id is not unique in matrix_model1")
m1["_ach"] = (
    (m1["t_payment_format"].astype(str) == "ACH")
    & (m1["t_is_self_loop"] == 0)
    & (m1["t_is_fx"] == 0)
)
flag = m1.set_index("txn_id")["_ach"]
del m1


def predict(booster, x):
    d = xgb.DMatrix(x, enable_categorical=True)
    return booster.predict(
        d, iteration_range=(0, int(booster.best_iteration) + 1)
    )


def prf(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    n_alerts = int(pred.sum())
    n_pos = int(y.sum())
    precision = tp / n_alerts if n_alerts else 0.0
    recall = tp / n_pos if n_pos else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return tp, n_alerts, precision, recall, f1


def average_precision(y, scores):
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y = y[order]
    total_pos = int(y.sum())
    if total_pos == 0:
        return 0.0
    cumulative_pos = np.cumsum(y)
    precision = cumulative_pos / np.arange(1, len(y) + 1)
    return float(precision[y == 1].sum() / total_pos)


runs = []

for model_name in ["m1", "m2"]:
    matrix = paths.PROCESSED / (
        "matrix_model1.parquet"
        if model_name == "m1"
        else "matrix_model2.parquet"
    )
    print(f"\nLoading {matrix.name}...")
    df = pd.read_parquet(matrix)

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")

    sub = flag.reindex(df["txn_id"])
    if sub.isna().any():
        raise SystemExit(f"STOP: {matrix.name} has txn_id missing from flag")
    df["_ach"] = sub.to_numpy().astype(bool)

    # Size check against the sizing command, once per matrix.
    for s in SPLITS:
        part = df[(df["split"] == s) & df["_ach"]]
        got = (len(part), int(part["is_laundering"].sum()))
        if got != EXPECTED[s]:
            raise SystemExit(
                f"STOP: {matrix.name} {s} size {got} != expected {EXPECTED[s]}"
            )

    for variant in ["full", "noshortcut"]:
        cols = X.feature_columns(fs, model_name, variant)
        name = f"{model_name}_{variant}"
        file_variant = "noflags" if variant == "noshortcut" else variant

        parts = {s: df[df["split"] == s] for s in SPLITS}
        ys = {
            s: parts[s]["is_laundering"].astype(int).to_numpy()
            for s in SPLITS
        }

        for seed in SEEDS:
            key = f"{name}_seed{seed}"
            run_id = f"{RUN_DATE}_{model_name}_{file_variant}_seed{seed}"
            thr = thresholds[key]["threshold"]

            booster = xgb.Booster()
            booster.load_model(str(paths.ROOT / "models" / f"{key}.json"))

            # Self-check on full validation before using anything.
            val_scores = predict(booster, parts["val"][cols])
            _, _, _, _, val_f1 = prf(ys["val"], val_scores, thr)
            frozen = thresholds[key]["val_f1"]
            if abs(val_f1 - frozen) > 1e-6:
                raise SystemExit(
                    f"\nSTOP: {key}\n"
                    f"Recomputed val F1 = {val_f1:.12f}\n"
                    f"Frozen val F1     = {frozen:.12f}\n"
                    f"Nothing was written."
                )

            split_docs = {}
            for s in SPLITS:
                scores = val_scores if s == "val" else predict(
                    booster, parts[s][cols]
                )
                mask = parts[s]["_ach"].to_numpy()
                y = ys[s][mask]
                p = scores[mask]

                n = int(len(y))
                n_pos = int(y.sum())
                tp, n_alerts, prec, rec, f1 = prf(y, p, thr)

                split_docs[s] = {
                    "n": n,
                    "n_pos": n_pos,
                    "base_rate": round(n_pos / n, 6) if n else 0.0,
                    "pr_auc": round(average_precision(y, p), 5)
                    if n_pos
                    else None,
                    "threshold": thr,
                    "n_alerts": n_alerts,
                    "precision": round(prec, 5),
                    "recall": round(rec, 5),
                    "f1": round(f1, 5),
                    "informative": s != "tail",
                    "low_positive": n_pos < LOW_POS,
                }

            runs.append(
                {
                    "run_id": run_id,
                    "model": model_name,
                    "variant": file_variant,
                    "seed": seed,
                    "splits": split_docs,
                }
            )
            print(f"{run_id}: ok")

doc = {
    "rule": RULE,
    "note": (
        "Subset of ACH transactions that are not self-loops and not FX. "
        "It holds about 12% of test rows but 87% of test fraud, so base "
        "rates differ from the full split and PR-AUC here is not "
        "comparable with the overall PR-AUC. Tail is reported separately "
        "and is not informative (714 rows, 91% fraud). low_positive is "
        f"true when a split has fewer than {LOW_POS} positives."
    ),
    "runs": runs,
}

out = paths.RESULTS / "hard_subset_ach.json"
out.write_text(json.dumps(doc, indent=2))
print(f"\nwrote {out} ({len(runs)} runs)")