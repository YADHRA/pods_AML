from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "results"
PREDICTIONS_FILE = DATA_DIR / "predictions_seed42.parquet"

# Keep the data source and alert flags consistent with Person 1's outputs.

MODEL_COLUMNS = {
"m1": {
"full": ("score_m1_full", "alert_m1_full"),
"noflags": ("score_m1_noflags", "alert_m1_noflags"),
},
"m2": {
"full": ("score_m2_full", "alert_m2_full"),
"noflags": ("score_m2_noflags", "alert_m2_noflags"),
},
}

VALID_SPLITS = {"val", "test", "tail"}
def get_alerts(
    split: str = "test",
    model: str = "m2",
    variant: str = "full",
    limit: int = 50,
    offset: int = 0,
):
    split = split.lower()
    model = model.lower()
    variant = variant.lower()

    if split not in VALID_SPLITS:
        raise ValueError("split must be val, test, or tail")

    if model not in MODEL_COLUMNS:
        raise ValueError("model must be m1 or m2")

    if variant == "noshortcut":
        variant = "noflags"

    if variant not in MODEL_COLUMNS[model]:
        raise ValueError("variant must be full or noflags")

    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")

    if offset < 0:
        raise ValueError("offset must be non-negative")

    score_col, alert_col = MODEL_COLUMNS[model][variant]

    columns = [
        "txn_id",
        "split",
        "is_laundering",
        "new_pair",
        score_col,
        alert_col,
    ]

    predictions = pd.read_parquet(
        PREDICTIONS_FILE,
        columns=columns,
        filters=[
            ("split", "==", split),
            (alert_col, "==", True),
        ],
    )

    predictions = predictions.sort_values(
        score_col, ascending=False, kind="stable"
    )

    total = len(predictions)
    page = predictions.iloc[offset : offset + limit]

    items = []
    for _, row in page.iterrows():
        items.append({
            "txn_id": int(row["txn_id"]),
            "split": str(row["split"]),
            "is_laundering": int(row["is_laundering"]),
            "new_pair": bool(row["new_pair"]),
            "score": float(row[score_col]),
            "alert": bool(row[alert_col]),
        })

    return {
        "split": split,
        "model": model,
        "variant": variant,
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
    }