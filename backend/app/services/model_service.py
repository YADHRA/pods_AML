
import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[3] / "results"

PREDICTIONS_FILE = DATA_DIR / "predictions_seed42.parquet"


PREDICTION_COLUMNS = [
    "txn_id",
    "split",
    "is_laundering",
    "new_pair",
    "score_m2_full",
    "alert_m2_full",
    "score_m2_noflags",
    "alert_m2_noflags",
    "score_m1_full",
    "alert_m1_full",
    "score_m1_noflags",
    "alert_m1_noflags",
]


def get_prediction(txn_id: int):
    predictions = pd.read_parquet(
        PREDICTIONS_FILE,
        columns=PREDICTION_COLUMNS,
        filters=[("txn_id", "==", txn_id)],
    )

    if predictions.empty:
        return None

    row = predictions.iloc[0]

    return {
        "txn_id": int(row["txn_id"]),
        "split": str(row["split"]),
        "is_laundering": int(row["is_laundering"]),
        "new_pair": bool(row["new_pair"]),

        "m1_full": {
            "score": float(row["score_m1_full"]),
            "alert": bool(row["alert_m1_full"]),
        },
        "m1_noflags": {
            "score": float(row["score_m1_noflags"]),
            "alert": bool(row["alert_m1_noflags"]),
        },
        "m2_full": {
            "score": float(row["score_m2_full"]),
            "alert": bool(row["alert_m2_full"]),
        },
        "m2_noflags": {
            "score": float(row["score_m2_noflags"]),
            "alert": bool(row["alert_m2_noflags"]),
        },
    }

def get_evaluation():
    """Load saved evaluation metrics and frozen validation thresholds."""
    results_dir = Path(__file__).resolve().parents[3] / "results"

    with open(results_dir / "test_results.json", encoding="utf-8") as f:
        test_results = json.load(f)

    with open(results_dir / "thresholds.json", encoding="utf-8") as f:
        thresholds = json.load(f)

    evaluation = {}

    for model_name in ("m1", "m2"):
        run_name = f"{model_name}_full_seed42"

        if run_name not in test_results or run_name not in thresholds:
            raise ValueError(f"Missing saved results for {run_name}")

        run = test_results[run_name]
        test = run["test"]
        threshold_info = thresholds[run_name]

        evaluation[model_name] = {
            "run": run_name,
            "variant": run["variant"],
            "seed": run["seed"],
            "features": run["features"],
            "test": {
                "n": test["n"],
                "n_pos": test["n_pos"],
                "prevalence": test["prevalence"],
                "pr_auc": test["pr_auc"],
                "roc_auc": test["roc_auc"],
                "precision": test["precision"],
                "recall": test["recall"],
                "f1": test["f1"],
                "accuracy": test["accuracy"],
                "lift": test["lift"],
                "threshold": test["threshold"],
                "tn": test["tn"],
                "fp": test["fp"],
                "fn": test["fn"],
                "tp": test["tp"],
                "n_alerts": test["n_alerts"],
                "recall_at_budget": test["recall_at_budget"],
                "precision_at_budget": test["precision_at_budget"],
            },
            "validation_threshold": threshold_info,
        }

    return {
        "evaluation_split": "test",
        "threshold_selection": "max_f1_on_validation",
        "models": evaluation,
    }