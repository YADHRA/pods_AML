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