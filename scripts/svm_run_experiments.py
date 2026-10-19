"""Train and validate the fixed linear SVM baseline."""

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from aml.common import paths
from aml.features import assemble as A
from aml.evaluation.metrics import average_precision, evaluate
from aml.evaluation.threshold import (
    best_f1_threshold,
    freeze_threshold,
)
from aml.models import svm_baseline as S


MATRIX = paths.PROCESSED / "matrix_model1.parquet"
MODEL_DIR = paths.ROOT / "models"
RESULT_DIR = paths.RESULTS / "models"

VAL_JSON = RESULT_DIR / "svm_m1_full_val.json"
THRESHOLDS = paths.RESULTS / "svm_thresholds.json"
VERIFIED = paths.RESULTS / "svm_validation_verified.json"
TEST_RESULTS = paths.RESULTS / "svm_test_results.json"
TEST_STARTED = paths.RESULTS / "svm_test_eval_started.json"

MODEL_PATHS = {
    seed: MODEL_DIR / f"svm_m1_full_seed{seed}.joblib"
    for seed in S.SEEDS
}


def refuse_existing_outputs():
    outputs = [
        VAL_JSON,
        THRESHOLDS,
        VERIFIED,
        TEST_RESULTS,
        TEST_STARTED,
        *MODEL_PATHS.values(),
    ]
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise SystemExit(
            "SVM outputs already exist. Refusing to overwrite:\n"
            + "\n".join(existing)
            + "\nInspect these SVM-only files before deciding "
              "whether cleanup is appropriate."
        )


def main():
    refuse_existing_outputs()

    if not MATRIX.exists():
        raise SystemExit(f"Missing matrix: {MATRIX}")

    fs = A.load_feature_sets()
    features = list(fs["m1_transaction"])
    categorical = list(fs.get("categorical", []))

    # Predicate filters request train and validation rows only.
    df = pd.read_parquet(
        MATRIX,
        columns=["split", "is_laundering", *features],
        filters=[
            [("split", "=", "train")],
            [("split", "=", "val")],
        ],
    )

    if not set(df["split"].unique()).issubset({"train", "val"}):
        raise SystemExit("Unexpected split detected; stopping.")

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    X_train = train[features]
    y_train = train["is_laundering"].astype(int).to_numpy()
    X_val = val[features]
    y_val = val["is_laundering"].astype(int).to_numpy()

    del df, train, val

    print(
        f"Train: {len(y_train):,} rows, "
        f"{int(y_train.sum()):,} positives"
    )
    print(
        f"Validation: {len(y_val):,} rows, "
        f"{int(y_val.sum()):,} positives"
    )

    # Preprocessing learns statistics and categories from train only.
    preprocessor = S.build_preprocessor(features, categorical)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    print(
        f"Transformed shapes: train={X_train_t.shape}, "
        f"validation={X_val_t.shape}"
    )
    print(f"Sparse training matrix: {hasattr(X_train_t, 'tocsr')}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    paths.RESULTS.mkdir(parents=True, exist_ok=True)

    search_results = []
    best_ap = -1.0
    best_config = None
    best_model = None

    # The six configurations are all evaluated using seed 42.
    for index, config in enumerate(S.CONFIGS, start=1):
        start = time.time()
        model = S.make_classifier(config, seed=42)
        model.fit(X_train_t, y_train)

        val_scores = S.score(model, X_val_t)
        ap = average_precision(y_val, val_scores)
        elapsed = round(time.time() - start, 1)

        row = {
            "config_id": index,
            **config,
            "seed": 42,
            "max_iter": S.MAX_ITER,
            "val_pr_auc": float(ap),
            "seconds": elapsed,
        }
        search_results.append(row)

        print(
            f"Config {index}/6: {config} | "
            f"validation PR-AUC={ap:.6f} | {elapsed}s"
        )

        if ap > best_ap:
            best_ap = float(ap)
            best_config = dict(config)
            best_model = model

    print(
        f"\nSelected on seed 42 validation PR-AUC: "
        f"{best_config} ({best_ap:.6f})"
    )

    val_results = {}

    for seed in S.SEEDS:
        # Reuse the selected seed-42 fit rather than fitting it again.
        if seed == 42:
            model = best_model
        else:
            model = S.make_classifier(best_config, seed)
            model.fit(X_train_t, y_train)

        val_scores = S.score(model, X_val_t)
        threshold_info = best_f1_threshold(y_val, val_scores)

        key = f"svm_m1_full_seed{seed}"

        # Uses the project's existing threshold-freezing API,
        # but writes to the separate SVM threshold file.
        freeze_threshold(
            THRESHOLDS,
            key,
            threshold_info,
            overwrite=False,
        )

        metrics = evaluate(
            y_val,
            val_scores,
            threshold_info["threshold"],
        )
        val_results[str(seed)] = {
            **metrics,
            "threshold_info": threshold_info,
        }

        joblib.dump(
            {
                "model": model,
                "preprocessor": preprocessor,
                "features": features,
                "categorical_features": [
                    c for c in categorical if c in features
                ],
                "config": best_config,
                "seed": seed,
                "max_iter": S.MAX_ITER,
            },
            MODEL_PATHS[seed],
        )

        print(
            f"Seed {seed}: PR-AUC={metrics['pr_auc']:.6f}, "
            f"threshold={threshold_info['threshold']:.6f}, "
            f"F1={metrics['f1']:.6f}"
        )

    aps = [val_results[str(seed)]["pr_auc"] for seed in S.SEEDS]

    output = {
        "name": "svm_m1_full",
        "features": features,
        "seeds": list(S.SEEDS),
        "selection_seed": 42,
        "selection_metric": "validation PR-AUC",
        "grid": S.CONFIGS,
        "max_iter": S.MAX_ITER,
        "best_config": best_config,
        "search": search_results,
        "val": val_results,
        "val_pr_auc_mean": sum(aps) / len(aps),
        "val_pr_auc_min": min(aps),
        "val_pr_auc_max": max(aps),
        "protocol": (
            "Train-only preprocessing; configuration selected using "
            "seed 42 validation PR-AUC; thresholds frozen on validation. "
            "Test and tail rows were not used by this script."
        ),
    }

    VAL_JSON.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {VAL_JSON}")
    print(f"Wrote {THRESHOLDS}")
    print("Test and tail partitions were not scored.")


if __name__ == "__main__":
    main()