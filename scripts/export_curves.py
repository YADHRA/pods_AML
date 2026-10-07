import json

import numpy as np
import pandas as pd
import xgboost as xgb

from aml.common import paths
from aml.features import assemble as A
from aml.models import xgb as X


RUN_DATE = "20261007"
SEEDS = [42, 43, 44]
MAX_PTS = 300
BUDGETS = [
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.01,
    0.02,
    0.05,
]

thresholds = json.loads(
    (paths.RESULTS / "thresholds.json").read_text()
)

fs = A.load_feature_sets()

out_dir = paths.RESULTS / "curves"
out_dir.mkdir(parents=True, exist_ok=True)


def predict(booster, x):
    d = xgb.DMatrix(
        x,
        enable_categorical=True,
    )

    return booster.predict(
        d,
        iteration_range=(
            0,
            int(booster.best_iteration) + 1,
        ),
    )


def f1_at(y, p, thr):
    pred = p >= thr

    tp = int((pred & (y == 1)).sum())
    n_pred = int(pred.sum())
    n_pos = int(y.sum())

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_pos if n_pos else 0.0

    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )


def pr_curve(y, scores):
    """
    NumPy implementation of a precision-recall curve.

    Returns arrays corresponding to threshold sweep points.
    """

    order = np.argsort(-scores, kind="mergesort")

    y_sorted = y[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)

    # Include the conventional starting point.
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    return precision, recall


def roc_curve(y, scores):
    """
    NumPy implementation of an ROC curve.
    """

    order = np.argsort(-scores, kind="mergesort")

    y_sorted = y[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    pos = int(y.sum())
    neg = len(y) - pos

    tpr = tp / max(pos, 1)
    fpr = fp / max(neg, 1)

    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]

    return fpr, tpr


def thin(arrays, k=MAX_PTS):
    n = len(arrays[0])

    if n <= k:
        return arrays

    idx = np.unique(
        np.linspace(
            0,
            n - 1,
            k,
        ).astype(int)
    )

    return [a[idx] for a in arrays]


def rounded(values):
    return [
        round(float(v), 5)
        for v in values
    ]


def curves_doc(
    run_id,
    split,
    y,
    scores,
    threshold,
):
    n = len(y)
    pos = int(y.sum())

    # -----------------------------
    # PR CURVE
    # -----------------------------

    precision, recall = pr_curve(
        y,
        scores,
    )

    precision, recall = thin(
        [precision, recall]
    )

    # -----------------------------
    # ROC CURVE
    # -----------------------------

    fpr, tpr = roc_curve(
        y,
        scores,
    )

    fpr, tpr = thin(
        [fpr, tpr]
    )

    # -----------------------------
    # ALERT BUDGET
    # -----------------------------

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    y_sorted = y[order]

    cumulative_tp = np.cumsum(
        y_sorted
    )

    budget = []

    for frac in BUDGETS:

        k = max(
            1,
            int(round(frac * n)),
        )

        tp = int(
            cumulative_tp[k - 1]
        )

        budget.append(
            {
                "frac": frac,
                "k": k,
                "recall": round(
                    tp / pos,
                    5,
                ),
                "precision": round(
                    tp / k,
                    5,
                ),
            }
        )

    # -----------------------------
    # FROZEN OPERATING POINT
    # -----------------------------

    predicted = scores >= threshold

    tp = int(
        (predicted & (y == 1)).sum()
    )

    n_alerts = int(
        predicted.sum()
    )

    return {
        "run_id": run_id,
        "split": split,
        "n": n,
        "n_pos": pos,
        "prevalence": pos / n,

        "pr": {
            "recall": rounded(recall),
            "precision": rounded(precision),
        },

        "roc": {
            "fpr": rounded(fpr),
            "tpr": rounded(tpr),
        },

        "budget": budget,

        "operating_point": {
            "threshold": threshold,
            "n_alerts": n_alerts,
            "precision": round(
                tp / n_alerts,
                5,
            ) if n_alerts else 0.0,
            "recall": round(
                tp / pos,
                5,
            ),
        },
    }


count = 0

for model_name in ["m1", "m2"]:

    matrix = paths.PROCESSED / (
        "matrix_model1.parquet"
        if model_name == "m1"
        else "matrix_model2.parquet"
    )

    print(
        f"\nLoading {matrix.name}..."
    )

    df = pd.read_parquet(matrix)

    for variant in [
        "full",
        "noshortcut",
    ]:

        cols = X.feature_columns(
            fs,
            model_name,
            variant,
        )

        name = (
            f"{model_name}_{variant}"
        )

        file_variant = (
            "noflags"
            if variant == "noshortcut"
            else variant
        )

        parts = {
            split: df[
                df["split"] == split
            ]
            for split in [
                "val",
                "test",
                "tail",
            ]
        }

        ys = {
            split: parts[split][
                "is_laundering"
            ].astype(int).to_numpy()
            for split in parts
        }

        for seed in SEEDS:

            key = (
                f"{name}_seed{seed}"
            )

            run_id = (
                f"{RUN_DATE}_"
                f"{model_name}_"
                f"{file_variant}_"
                f"seed{seed}"
            )

            threshold = thresholds[
                key
            ]["threshold"]

            booster = xgb.Booster()

            booster.load_model(
                str(
                    paths.ROOT
                    / "models"
                    / f"{key}.json"
                )
            )

            # -----------------------------
            # VALIDATION SELF-CHECK
            # -----------------------------

            val_x = parts["val"][cols]

            val_scores = predict(
                booster,
                val_x,
            )

            val_f1 = f1_at(
                ys["val"],
                val_scores,
                threshold,
            )

            frozen_f1 = thresholds[
                key
            ]["val_f1"]

            if abs(
                val_f1 - frozen_f1
            ) > 1e-6:

                raise SystemExit(
                    f"\nSTOP: {key}\n"
                    f"Recomputed validation F1 = "
                    f"{val_f1:.12f}\n"
                    f"Frozen validation F1 = "
                    f"{frozen_f1:.12f}\n"
                    f"Difference = "
                    f"{abs(val_f1 - frozen_f1):.12f}\n"
                    f"No curve files were written "
                    f"for this run."
                )

            # -----------------------------
            # EXPORT CURVES
            # -----------------------------

            for split in [
                "val",
                "test",
                "tail",
            ]:

                if split == "val":
                    scores = val_scores
                else:
                    x = parts[split][cols]

                    scores = predict(
                        booster,
                        x,
                    )

                doc = curves_doc(
                    run_id,
                    split,
                    ys[split],
                    scores,
                    threshold,
                )

                output = (
                    out_dir
                    / f"{run_id}_{split}.json"
                )

                output.write_text(
                    json.dumps(
                        doc,
                        indent=2,
                    )
                )

                count += 1

            print(
                f"{run_id}: ok"
            )


print(
    f"\nwrote {count} files "
    f"to {out_dir}"
)