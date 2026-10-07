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
N_REPEATS = 3
N_NEG = 120_000

thresholds = json.loads(
    (paths.RESULTS / "thresholds.json").read_text()
)

fs = A.load_feature_sets()

rng = np.random.default_rng(0)


def prep(part, cols):
    return part[cols].copy().reset_index(drop=True)


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

    tp = int(
        (pred & (y == 1)).sum()
    )

    n_pred = int(pred.sum())
    n_pos = int(y.sum())

    precision = (
        tp / n_pred
        if n_pred
        else 0.0
    )

    recall = (
        tp / n_pos
        if n_pos
        else 0.0
    )

    return (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )


def weighted_average_precision(
    y,
    scores,
    weights,
):
    """
    Weighted average precision for binary labels.

    Samples are ranked by descending score.
    Precision is evaluated at each positive observation,
    using the supplied sample weights.
    """

    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    y = y[order]
    weights = weights[order]

    weighted_pos = (
        weights * (y == 1)
    )

    total_pos = float(
        weighted_pos.sum()
    )

    if total_pos <= 0:
        return 0.0

    cumulative_weight = np.cumsum(
        weights
    )

    cumulative_pos = np.cumsum(
        weighted_pos
    )

    positive_mask = y == 1

    precision = (
        cumulative_pos[positive_mask]
        / cumulative_weight[positive_mask]
    )

    positive_weight = (
        weighted_pos[positive_mask]
    )

    return float(
        np.sum(
            precision * positive_weight
        )
        / total_pos
    )


out = {
    "metric": "pr_auc_drop",
    "split": "val",
    "n_repeats": N_REPEATS,
    "note": (
        "importance = baseline PR-AUC minus PR-AUC "
        "with one feature shuffled; higher = model "
        "relies on it more"
    ),
    "runs": {},
}


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

    val = df[
        df["split"] == "val"
    ]

    y = (
        val["is_laundering"]
        .astype(int)
        .to_numpy()
    )

    for variant in [
        "full",
        "noshortcut",
    ]:

        cols = X.feature_columns(
            fs,
            model_name,
            variant,
        )

        key = (
            f"{model_name}_"
            f"{variant}_"
            f"seed{SEED}"
        )

        file_variant = (
            "noflags"
            if variant == "noshortcut"
            else variant
        )

        run_id = (
            f"{RUN_DATE}_"
            f"{model_name}_"
            f"{file_variant}_"
            f"seed{SEED}"
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

        x_full = prep(
            val,
            cols,
        )

        p = predict(
            booster,
            x_full,
        )

        reproduced_f1 = f1_at(
            y,
            p,
            threshold,
        )

        frozen_f1 = thresholds[
            key
        ]["val_f1"]

        if abs(
            reproduced_f1
            - frozen_f1
        ) > 1e-6:

            sys.exit(
                f"\nSTOP: {key}\n"
                f"Recomputed validation F1 = "
                f"{reproduced_f1:.12f}\n"
                f"Frozen validation F1 = "
                f"{frozen_f1:.12f}\n"
                f"Difference = "
                f"{abs(reproduced_f1 - frozen_f1):.12f}"
            )

        print(
            f"{run_id}: validation check OK"
        )

        # -----------------------------
        # SAMPLE ALL POSITIVES +
        # 120K NEGATIVES
        # -----------------------------

        pos_idx = np.where(
            y == 1
        )[0]

        neg_idx = np.where(
            y == 0
        )[0]

        if len(neg_idx) < N_NEG:
            sys.exit(
                f"STOP: only {len(neg_idx):,} "
                f"negative validation rows available; "
                f"{N_NEG:,} requested."
            )

        selected_neg = rng.choice(
            neg_idx,
            N_NEG,
            replace=False,
        )

        idx = np.sort(
            np.r_[
                pos_idx,
                selected_neg,
            ]
        )

        xs = (
            x_full
            .iloc[idx]
            .reset_index(drop=True)
        )

        ys = y[idx]

        # Weight each sampled negative so the sampled
        # negative population represents the full
        # validation negative population.
        weights = np.where(
            ys == 1,
            1.0,
            len(neg_idx) / N_NEG,
        )

        baseline_scores = predict(
            booster,
            xs,
        )

        baseline = (
            weighted_average_precision(
                ys,
                baseline_scores,
                weights,
            )
        )

        rows = []

        # -----------------------------
        # PERMUTATION IMPORTANCE
        # -----------------------------

        for c in cols:

            original = (
                xs[c].copy()
            )

            drops = []

            for _ in range(
                N_REPEATS
            ):

                permutation = (
                    rng.permutation(
                        len(xs)
                    )
                )

                xs[c] = (
                    original
                    .iloc[permutation]
                    .reset_index(drop=True)
                )

                shuffled_scores = predict(
                    booster,
                    xs,
                )

                ap = (
                    weighted_average_precision(
                        ys,
                        shuffled_scores,
                        weights,
                    )
                )

                drops.append(
                    baseline - ap
                )

            xs[c] = original

            rows.append(
                {
                    "feature": c,
                    "importance_mean": round(
                        float(
                            np.mean(drops)
                        ),
                        5,
                    ),
                    "importance_std": round(
                        float(
                            np.std(drops)
                        ),
                        5,
                    ),
                }
            )

        rows.sort(
            key=lambda r:
            -r["importance_mean"]
        )

        out["runs"][run_id] = {
            "baseline_pr_auc_weighted":
                round(
                    baseline,
                    5,
                ),
            "n_rows":
                int(len(xs)),
            "n_positive":
                int(ys.sum()),
            "n_negative_sampled":
                int(N_NEG),
            "features":
                rows,
        }

        print(
            f"\n{run_id} "
            f"(baseline {baseline:.4f}) "
            f"top 8:"
        )

        for row in rows[:8]:
            print(
                f"  "
                f"{row['feature']:28s} "
                f"{row['importance_mean']:.4f} "
                f"+/- "
                f"{row['importance_std']:.4f}"
            )


output = (
    paths.RESULTS
    / "permutation.json"
)

output.write_text(
    json.dumps(
        out,
        indent=2,
    )
)

print(
    f"\nwrote {output}"
)