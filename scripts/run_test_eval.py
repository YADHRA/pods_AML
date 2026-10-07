import json

import pandas as pd
import xgboost as xgb

from aml.common import paths
from aml.features import assemble as A
from aml.models import xgb as X


OUT = paths.RESULTS / "test_results.json"

if OUT.exists():
    raise SystemExit(
        f"{OUT} already exists. Test evaluation has already been performed."
    )

thresholds = json.loads(
    (paths.RESULTS / "thresholds.json").read_text()
)

fs = A.load_feature_sets()
SEEDS = [42, 43, 44]


def load_model(name):
    model = xgb.Booster()
    model.load_model(
        str(paths.ROOT / "models" / f"{name}.json")
    )
    return model


def score(model, df):
    dmatrix = xgb.DMatrix(
        df,
        enable_categorical=True,
    )

    return model.predict(
        dmatrix,
        iteration_range=(
            0,
            int(model.best_iteration) + 1,
        ),
    )


results = {}

for model_name in ["m1", "m2"]:

    matrix = paths.PROCESSED / (
        "matrix_model1.parquet"
        if model_name == "m1"
        else "matrix_model2.parquet"
    )

    print(f"\nLoading {matrix.name}...")
    df = pd.read_parquet(matrix)

    for variant in ["full", "noshortcut"]:

        cols = X.feature_columns(
            fs,
            model_name,
            variant,
        )

        name = f"{model_name}_{variant}"

        parts = {
            split: df[df["split"] == split]
            for split in ["val", "test", "tail"]
        }

        print(
            f"\n{name}: "
            f"val={len(parts['val']):,}, "
            f"test={len(parts['test']):,}, "
            f"tail={len(parts['tail']):,}"
        )

        for seed in SEEDS:

            key = f"{name}_seed{seed}"

            if key not in thresholds:
                raise SystemExit(
                    f"Missing frozen threshold for {key}"
                )

            threshold = thresholds[key]["threshold"]

            model = load_model(key)

            # -----------------------------
            # VALIDATION SELF-CHECK
            # -----------------------------

            val_x = parts["val"][cols]
            val_y = (
                parts["val"]["is_laundering"]
                .astype(int)
                .to_numpy()
            )

            val_scores = score(model, val_x)

            val_metrics = X.evaluate(
                val_y,
                val_scores,
                threshold,
            )

            frozen_f1 = thresholds[key]["val_f1"]

            if abs(val_metrics["f1"] - frozen_f1) > 1e-6:
                raise SystemExit(
                    f"\nSTOP: {key}\n"
                    f"Recomputed validation F1 = "
                    f"{val_metrics['f1']:.12f}\n"
                    f"Frozen validation F1      = "
                    f"{frozen_f1:.12f}\n"
                    f"Difference                 = "
                    f"{abs(val_metrics['f1'] - frozen_f1):.12f}\n"
                    f"Test and tail were NOT evaluated "
                    f"for this model."
                )

            print(
                f"{key}: validation check OK "
                f"(F1={val_metrics['f1']:.4f}, "
                f"threshold={threshold:.6f})"
            )

            # -----------------------------
            # TEST + TAIL
            # -----------------------------

            test_x = parts["test"][cols]
            test_y = (
                parts["test"]["is_laundering"]
                .astype(int)
                .to_numpy()
            )

            tail_x = parts["tail"][cols]
            tail_y = (
                parts["tail"]["is_laundering"]
                .astype(int)
                .to_numpy()
            )

            test_scores = score(model, test_x)
            tail_scores = score(model, tail_x)

            test_metrics = X.evaluate(
                test_y,
                test_scores,
                threshold,
            )

            tail_metrics = X.evaluate(
                tail_y,
                tail_scores,
                threshold,
            )

            results[key] = {
                "model": model_name,
                "variant": variant,
                "seed": seed,
                "features": cols,
                "validation_check": val_metrics,
                "test": test_metrics,
                "tail": tail_metrics,
            }

            print(
                f"{key}: "
                f"TEST PR-AUC={test_metrics['pr_auc']:.4f} | "
                f"ROC-AUC={test_metrics['roc_auc']:.4f} | "
                f"P={test_metrics['precision']:.3f} | "
                f"R={test_metrics['recall']:.3f} | "
                f"F1={test_metrics['f1']:.3f}"
            )

            print(
                f"        TAIL PR-AUC={tail_metrics['pr_auc']:.4f} | "
                f"P={tail_metrics['precision']:.3f} | "
                f"R={tail_metrics['recall']:.3f} | "
                f"F1={tail_metrics['f1']:.3f}"
            )


OUT.write_text(
    json.dumps(results, indent=2)
)

print(f"\nWrote {OUT}")
print("Test and tail evaluation completed successfully.")