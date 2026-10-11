import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

PREDICTIONS_FILE = RESULTS_DIR / "predictions_seed42.parquet"
TRANSACTIONS_FILE = DATA_DIR / "transactions_clean.parquet"
NODE_MAP_FILE = DATA_DIR / "node_map.parquet"
THRESHOLDS_FILE = RESULTS_DIR / "thresholds.json"

_cached_alerts_df = {}
_cached_thresholds = None
_cached_node_map = None

def get_thresholds():
    global _cached_thresholds
    if _cached_thresholds is None:
        if THRESHOLDS_FILE.exists():
            with open(THRESHOLDS_FILE, "r", encoding="utf-8") as f:
                _cached_thresholds = json.load(f)
        else:
            _cached_thresholds = {}
    return _cached_thresholds


def get_node_map():
    global _cached_node_map
    if _cached_node_map is None:
        _cached_node_map = pd.read_parquet(NODE_MAP_FILE)
    return _cached_node_map


def get_base_alerts_df(model: str = "m2", split: str = "test"):
    cache_key = f"{model}_{split}"
    if cache_key in _cached_alerts_df:
        return _cached_alerts_df[cache_key]

    thr_data = get_thresholds()
    thr_key = f"{model}_full_seed42"
    threshold = thr_data.get(thr_key, {}).get("threshold", 0.2628 if model == "m2" else 0.0661)

    score_col = f"score_{model}_full"
    alert_col = f"alert_{model}_full"

    preds = pd.read_parquet(
        PREDICTIONS_FILE,
        columns=["txn_id", "split", "is_laundering", "new_pair", score_col, alert_col],
        filters=[
            ("split", "==", split),
            (alert_col, "==", True),
        ],
    )

    txn_ids = preds["txn_id"].tolist()
    txns = pd.read_parquet(
        TRANSACTIONS_FILE,
        columns=[
            "txn_id",
            "ts",
            "date",
            "src_node",
            "dst_node",
            "src_bank",
            "dst_bank",
            "amount_paid",
            "cur_paid",
            "payment_format",
        ],
        filters=[("txn_id", "in", txn_ids)],
    )

    nm = get_node_map()
    merged = preds.merge(txns, on="txn_id")

    # Merge sender and receiver accounts
    nm_src = nm.rename(columns={"node_id": "src_node", "account": "src_account", "bank": "nm_src_bank"})
    nm_dst = nm.rename(columns={"node_id": "dst_node", "account": "dst_account", "bank": "nm_dst_bank"})

    merged = merged.merge(nm_src[["src_node", "src_account"]], on="src_node", how="left")
    merged = merged.merge(nm_dst[["dst_node", "dst_account"]], on="dst_node", how="left")

    merged["src_account"] = merged["src_account"].fillna(merged["src_node"].astype(str))
    merged["dst_account"] = merged["dst_account"].fillna(merged["dst_node"].astype(str))

    # Compute risk level
    scores = merged[score_col].astype(float)
    merged["risk_score"] = scores

    def assign_risk(s):
        if s >= 0.50:
            return "High"
        if s >= threshold:
            return "Medium"
        return "Low"

    merged["risk_level"] = scores.apply(assign_risk)
    merged["timestamp"] = merged["ts"].astype(str)

    _cached_alerts_df[cache_key] = merged
    return merged


def get_alerts(
    split: str = "test",
    model: str = "m2",
    variant: str = "full",
    risk_level: str | None = None,
    payment_format: str | None = None,
    search: str | None = None,
    sort_by: str = "score",
    sort_order: str = "desc",
    limit: int = 10,
    offset: int = 0,
):
    model = model.lower()
    split = split.lower()
    df = get_base_alerts_df(model=model, split=split).copy()

    # Pre-calculate KPI counts before filtering
    total_alerts = len(df)
    high_count = int((df["risk_level"] == "High").sum())
    medium_count = int((df["risk_level"] == "Medium").sum())
    low_count = int((df["risk_level"] == "Low").sum())

    # Apply filters
    if risk_level and risk_level.lower() != "all":
        df = df[df["risk_level"].str.lower() == risk_level.lower()]

    if payment_format and payment_format.lower() != "all":
        df = df[df["payment_format"].str.lower() == payment_format.lower()]

    if search:
        s_term = search.strip().lower()
        df = df[
            df["txn_id"].astype(str).str.contains(s_term)
            | df["src_account"].str.lower().str.contains(s_term)
            | df["dst_account"].str.lower().str.contains(s_term)
            | df["src_bank"].str.lower().str.contains(s_term)
            | df["dst_bank"].str.lower().str.contains(s_term)
        ]

    # Sorting
    ascending = (sort_order.lower() == "asc")
    sort_col = "risk_score"
    if sort_by == "amount":
        sort_col = "amount_paid"
    elif sort_by == "date" or sort_by == "timestamp":
        sort_col = "ts"
    elif sort_by == "txn_id":
        sort_col = "txn_id"

    df = df.sort_values(sort_col, ascending=ascending, kind="stable")

    filtered_total = len(df)
    page_rows = df.iloc[offset : offset + limit]

    items = []
    for _, row in page_rows.iterrows():
        items.append({
            "txn_id": int(row["txn_id"]),
            "timestamp": str(row["ts"]),
            "date": str(row["date"]),
            "sender_account": str(row["src_account"]),
            "sender_bank": str(row["src_bank"]),
            "sender_node": int(row["src_node"]),
            "receiver_account": str(row["dst_account"]),
            "receiver_bank": str(row["dst_bank"]),
            "receiver_node": int(row["dst_node"]),
            "split": str(row["split"]),
            "score": float(row["risk_score"]),
            "amount": float(row["amount_paid"]),
            "currency": str(row["cur_paid"]),
            "payment_format": str(row["payment_format"]),
            "risk_score": float(row["risk_score"]),
            "risk_level": str(row["risk_level"]),
            "is_laundering": int(row["is_laundering"]),
            "new_pair": bool(row["new_pair"]),
            "alert": True,
        })

    return {
        "split": split,
        "model": model,
        "variant": variant,
        "total": filtered_total,
        "limit": limit,
        "offset": offset,
        "kpis": {
            "total_alerts": total_alerts,
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
        },
        "items": items,
    }


def get_account_alerts(
    split: str = "test",
    model: str = "m2",
    risk_level: str | None = None,
    search: str | None = None,
    limit: int = 10,
    offset: int = 0,
):
    model = model.lower()
    split = split.lower()
    df = get_base_alerts_df(model=model, split=split)

    # Gather all accounts involved in alerts (both senders and receivers)
    src_accounts = df[["src_account", "src_bank", "src_node", "risk_score", "risk_level"]].rename(
        columns={"src_account": "account", "src_bank": "bank", "src_node": "node_id"}
    )
    dst_accounts = df[["dst_account", "dst_bank", "dst_node", "risk_score", "risk_level"]].rename(
        columns={"dst_account": "account", "dst_bank": "bank", "dst_node": "node_id"}
    )
    combined = pd.concat([src_accounts, dst_accounts], ignore_index=True)

    # Group by account and bank
    grouped = combined.groupby(["account", "bank", "node_id"]).agg(
        alert_count=("risk_score", "count"),
        highest_score=("risk_score", "max"),
    ).reset_index()

    thr_data = get_thresholds()
    thr_key = f"{model}_full_seed42"
    threshold = thr_data.get(thr_key, {}).get("threshold", 0.2628 if model == "m2" else 0.0661)

    def assign_risk(s):
        if s >= 0.50:
            return "High"
        if s >= threshold:
            return "Medium"
        return "Low"

    grouped["risk_level"] = grouped["highest_score"].apply(assign_risk)

    total_accounts = len(grouped)
    high_count = int((grouped["risk_level"] == "High").sum())
    medium_count = int((grouped["risk_level"] == "Medium").sum())
    low_count = int((grouped["risk_level"] == "Low").sum())

    if risk_level and risk_level.lower() != "all":
        grouped = grouped[grouped["risk_level"].str.lower() == risk_level.lower()]

    if search:
        s_term = search.strip().lower()
        grouped = grouped[
            grouped["account"].str.lower().str.contains(s_term)
            | grouped["bank"].str.lower().str.contains(s_term)
            | grouped["node_id"].astype(str).str.contains(s_term)
        ]

    grouped = grouped.sort_values(["highest_score", "alert_count"], ascending=[False, False])

    filtered_total = len(grouped)
    page_rows = grouped.iloc[offset : offset + limit]

    items = []
    for _, row in page_rows.iterrows():
        items.append({
            "account": str(row["account"]),
            "bank": str(row["bank"]),
            "node_id": int(row["node_id"]),
            "alert_count": int(row["alert_count"]),
            "highest_score": float(row["highest_score"]),
            "risk_level": str(row["risk_level"]),
        })

    return {
        "split": split,
        "model": model,
        "total": filtered_total,
        "limit": limit,
        "offset": offset,
        "kpis": {
            "total_accounts": total_accounts,
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
        },
        "items": items,
    }