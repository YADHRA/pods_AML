"""Verify frozen SVM validation results before final evaluation."""

import hashlib
import json
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

MODEL_PATHS = {
    seed: paths.ROOT / "models" / f"svm_m1_full_seed{seed}.joblib"
    for seed in S.SEEDS
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_hashes():
    files = [VAL_JSON, THRESHOLDS, *MODEL_PATHS.values()]
    return {
        p.relative_to(paths.ROOT).as_posix(): sha256(p)
        for p in files
    }


def main():
    if not VAL_JSON.exists() or not THRESHOLDS.exists():
        raise SystemExit(
            "Validation output or frozen SVM thresholds are missing."
        )

    if not all(p.exists() for p in MODEL_PATHS.values()):
        raise SystemExit("One or more saved SVM models are missing.")

    # If already verified, do not silently replace the verification record.
    if VERIFIED.exists():
        saved = json.loads(VERIFIED.read_text())
        if saved["artifact_sha256"] != artifact_hashes():
            raise SystemExit(
                "SVM validation artifacts changed after verification. "
                "Stop and investigate; do not proceed to test evaluation."
            )
        print("Validation artifacts already verified and unchanged.")
        return

    val_info = json.loads(VAL_JSON.read_text())
    thresholds = json.loads(THRESHOLDS.read_text())
    features = val_info["features"]

    # Read validation rows only.
    df = pd.read_parquet(
        MATRIX,
        columns=["split", "is_laundering", *features],
        filters=[[("split", "=", "val")]],
    )

    if not (df["split"] == "val").all():
        raise SystemExit("Unexpected split in validation data.")

    y_val = df["is_laundering"].astype(int).to_numpy()
    X_val = df[features]

    seed42_bundle = joblib.load(MODEL_PATHS[42])
    preprocessor = seed42_bundle["preprocessor"]
    X_val_t = preprocessor.transform(X_val)

    checks = {}

    for seed in S.SEEDS:
        key = f"svm_m1_full_seed{seed}"

        if key not in thresholds:
            raise SystemExit(f"Missing frozen threshold: {key}")

        bundle = joblib.load(MODEL_PATHS[seed])
        if bundle["features"] != features:
            raise SystemExit(f"Feature mismatch in saved model: {key}")

        scores = S.score(bundle["model"], X_val_t)
        threshold = thresholds[key]["threshold"]
        metrics = evaluate(y_val, scores, threshold)
        frozen_f1 = thresholds[key]["val_f1"]
        difference = abs(metrics["f1"] - frozen_f1)

        if difference > 1e-6:
            raise SystemExit(
                f"Validation F1 mismatch for {key}: "
                f"recomputed={metrics['f1']:.12f}, "
                f"frozen={frozen_f1:.12f}. Test evaluation blocked."
            )

        checks[key] = {
            "threshold": threshold,
            "frozen_val_f1": frozen_f1,
            "recomputed_val_f1": metrics["f1"],
            "f1_difference": difference,
            "pr_auc": metrics["pr_auc"],
            "passed": True,
        }

        print(
            f"{key}: PASS | validation PR-AUC={metrics['pr_auc']:.6f}"
        )

    record = {
        "model": "svm_m1_full",
        "validation_only": True,
        "checks": checks,
        "artifact_sha256": artifact_hashes(),
        "test_evaluated": False,
    }

    VERIFIED.write_text(json.dumps(record, indent=2))
    print(f"\nValidation verified. Wrote {VERIFIED}")
    print("You may now run the final SVM test evaluation.")


if __name__ == "__main__":
    main()