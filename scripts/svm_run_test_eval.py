"""One-time final evaluation for the frozen SVM baseline."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from aml.common import paths
from aml.evaluation.metrics import evaluate
from aml.models import svm_baseline as S


MATRIX = paths.PROCESSED / "matrix_model1.parquet"
VAL_JSON = paths.RESULTS / "models" / "svm_m1_full_val.json"
THRESHOLDS = paths.RESULTS / "svm_thresholds.json"
VERIFIED = paths.RESULTS / "svm_validation_verified.json"
OUT = paths.RESULTS / "svm_test_results.json"
STARTED = paths.RESULTS / "svm_test_eval_started.json"

MODEL_PATHS = {
    seed: paths.ROOT / "models" / f"svm_m1_full_seed{seed}.joblib"
    for seed in S.SEEDS
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists() or STARTED.exists():
        raise SystemExit(
            "SVM test evaluation has already started or completed. "
            "Refusing to score test data again."
        )

    if not VERIFIED.exists():
        raise SystemExit(
            "Validation has not been verified. Run "
            "scripts/svm_verify_validation.py first."
        )

    verification = json.loads(VERIFIED.read_text())
    if verification.get("test_evaluated") is not False:
        raise SystemExit("Invalid validation verification record.")

    expected_hashes = verification["artifact_sha256"]

    for relative_path, expected_hash in expected_hashes.items():
        path = paths.ROOT / Path(relative_path)
        if not path.exists() or sha256(path) != expected_hash:
            raise SystemExit(
                f"Frozen artifact changed or missing: {relative_path}. "
                "Test evaluation blocked."
            )

    val_info = json.loads(VAL_JSON.read_text())
    thresholds = json.loads(THRESHOLDS.read_text())
    features = val_info["features"]

    # Load and validate saved model files before opening test data.
    bundles = {
        seed: joblib.load(MODEL_PATHS[seed])
        for seed in S.SEEDS
    }

    for seed, bundle in bundles.items():
        if bundle["features"] != features:
            raise SystemExit(f"Feature mismatch for seed {seed}.")

    # This marker prevents a second run, even if the first crashes.
    STARTED.write_text(json.dumps({
        "started_at": datetime.now().isoformat(),
        "status": "started",
        "warning": (
            "Do not remove this marker to retry automatically. "
            "Inspect the execution state first."
        ),
    }, indent=2))

    # This is the first point at which this script reads test/tail rows.
    df = pd.read_parquet(
        MATRIX,
        columns=["split", "is_laundering", *features],
        filters=[
            [("split", "=", "test")],
            [("split", "=", "tail")],
        ],
    )

    allowed = {"test", "tail"}
    if not set(df["split"].unique()).issubset(allowed):
        raise SystemExit("Unexpected split in final evaluation data.")

    parts = {
        split: df[df["split"] == split]
        for split in ("test", "tail")
    }

    if parts["test"].empty:
        raise SystemExit("Test partition is empty.")

    preprocessor = bundles[42]["preprocessor"]
    transformed = {
        split: preprocessor.transform(part[features])
        for split, part in parts.items()
        if not part.empty
    }

    results = {}

    for seed in S.SEEDS:
        key = f"svm_m1_full_seed{seed}"

        if key not in thresholds:
            raise SystemExit(f"Missing frozen threshold: {key}")

        bundle = bundles[seed]
        threshold = thresholds[key]["threshold"]

        seed_result = {
            "model": "svm_m1_full",
            "seed": seed,
            "threshold": threshold,
        }

        for split in ("test", "tail"):
            part = parts[split]
            if part.empty:
                continue

            y = part["is_laundering"].astype(int).to_numpy()
            scores = S.score(bundle["model"], transformed[split])
            metrics = evaluate(y, scores, threshold)
            seed_result[split] = metrics

            print(
                f"{key} | {split.upper()} | "
                f"PR-AUC={metrics['pr_auc']:.6f} | "
                f"precision={metrics['precision']:.4f} | "
                f"recall={metrics['recall']:.4f} | "
                f"F1={metrics['f1']:.4f}"
            )

        results[key] = seed_result

    OUT.write_text(json.dumps(results, indent=2))

    # Update the marker after successful completion.
    STARTED.write_text(json.dumps({
        "started_at": json.loads(STARTED.read_text())["started_at"],
        "completed_at": datetime.now().isoformat(),
        "status": "completed",
        "results": str(OUT.relative_to(paths.ROOT)),
    }, indent=2))

    print(f"\nWrote {OUT}")
    print("Final SVM evaluation completed. Further runs are blocked.")


if __name__ == "__main__":
    main()