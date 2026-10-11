from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

GRAPH_NODE_DAY_FILE = DATA_DIR / "graph_node_day.parquet"
NODE_MAP_FILE = DATA_DIR / "node_map.parquet"
TRANSACTIONS_FILE = DATA_DIR / "transactions_clean.parquet"
PREDICTIONS_FILE = RESULTS_DIR / "predictions_seed42.parquet"

GRAPH_COLUMNS = [
    "feature_date",
    "node_id",
    "g_pagerank_log",
    "g_wcc_size_log",
    "g_n_unique_out_log",
    "g_n_unique_in_log",
    "g_has_sent_before",
    "g_has_received_before",
]

_node_map_cache = None

def get_node_map():
    global _node_map_cache
    if _node_map_cache is None:
        _node_map_cache = pd.read_parquet(NODE_MAP_FILE)
    return _node_map_cache


def get_account(node_id: int, date: str = "2022-09-18"):
    requested_date = pd.to_datetime(date).normalize()

    graph = pd.read_parquet(
        GRAPH_NODE_DAY_FILE,
        columns=GRAPH_COLUMNS,
        filters=[
            ("node_id", "==", node_id),
            ("feature_date", "==", requested_date),
        ],
    )

    if graph.empty:
        # Fallback to any date for this node if exact date has no row
        graph = pd.read_parquet(
            GRAPH_NODE_DAY_FILE,
            columns=GRAPH_COLUMNS,
            filters=[("node_id", "==", node_id)],
        )
        if graph.empty:
            return None
        row = graph.sort_values("feature_date", ascending=False).iloc[0]
    else:
        row = graph.iloc[0]

    nm = get_node_map()
    match = nm[nm["node_id"] == node_id]
    if match.empty:
        return None

    node_row = match.iloc[0]

    return {
        "node_id": int(row["node_id"]),
        "bank": str(node_row["bank"]),
        "account": str(node_row["account"]),
        "date": str(row["feature_date"].date()),
        "pagerank_log": float(row["g_pagerank_log"]),
        "wcc_size_log": float(row["g_wcc_size_log"]),
        "unique_out_log": float(row["g_n_unique_out_log"]),
        "unique_in_log": float(row["g_n_unique_in_log"]),
        "has_sent_before": int(row["g_has_sent_before"]),
        "has_received_before": int(row["g_has_received_before"]),
    }


def lookup_account(q: str):
    nm = get_node_map()
    q_str = str(q).strip()

    match = pd.DataFrame()
    if q_str.isdigit():
        node_num = int(q_str)
        match = nm[nm["node_id"] == node_num]

    if match.empty:
        match = nm[nm["account"].str.upper() == q_str.upper()]

    if match.empty:
        match = nm[nm["account"].str.contains(q_str, case=False, na=False)]

    if match.empty:
        return None

    node_row = match.iloc[0]
    node_id = int(node_row["node_id"])
    bank = str(node_row["bank"])
    account = str(node_row["account"])

    # Historical behaviour from graph_node_day
    acc_details = get_account(node_id) or {
        "node_id": node_id,
        "bank": bank,
        "account": account,
        "date": "2022-09-18",
        "pagerank_log": 0.0,
        "wcc_size_log": 0.0,
        "unique_out_log": 0.0,
        "unique_in_log": 0.0,
        "has_sent_before": 0,
        "has_received_before": 0,
    }

    # Transactions summary
    txns = pd.read_parquet(
        TRANSACTIONS_FILE,
        columns=["txn_id", "ts", "src_node", "dst_node", "amount_paid", "cur_paid", "payment_format"],
        filters=[[("src_node", "==", node_id)], [("dst_node", "==", node_id)]],
    )

    total_txns = len(txns)
    first_seen = str(txns["ts"].min()) if not txns.empty else "Not available"
    last_seen = str(txns["ts"].max()) if not txns.empty else "Not available"

    # Find alerted transactions for this account
    alerted_list = []
    max_score = 0.0

    if not txns.empty and PREDICTIONS_FILE.exists():
        txn_ids = txns["txn_id"].tolist()
        # chunk check if many
        preds = pd.read_parquet(
            PREDICTIONS_FILE,
            columns=["txn_id", "score_m2_full", "alert_m2_full"],
            filters=[("txn_id", "in", txn_ids[:500])],
        )
        if not preds.empty:
            merged = txns.merge(preds, on="txn_id")
            if not merged.empty:
                max_score = float(merged["score_m2_full"].max())
                for _, r in merged.sort_values("score_m2_full", ascending=False).head(10).iterrows():
                    role = "Sender" if r["src_node"] == node_id else "Receiver"
                    other_node = int(r["dst_node"] if role == "Sender" else r["src_node"])
                    other_match = nm[nm["node_id"] == other_node]
                    other_acc = str(other_match["account"].iloc[0]) if not other_match.empty else f"Node {other_node}"
                    other_bank = str(other_match["bank"].iloc[0]) if not other_match.empty else "Unknown"

                    score = float(r["score_m2_full"])
                    level = "High" if score >= 0.50 else "Medium" if score >= 0.2628 else "Low"

                    alerted_list.append({
                        "txn_id": int(r["txn_id"]),
                        "timestamp": str(r["ts"]),
                        "role": role,
                        "counterparty_account": other_acc,
                        "counterparty_bank": other_bank,
                        "amount": float(r["amount_paid"]),
                        "currency": str(r["cur_paid"]),
                        "score": score,
                        "risk_level": level,
                        "payment_format": str(r["payment_format"]),
                        "alert": bool(r["alert_m2_full"]),
                    })

    risk_level = "High" if max_score >= 0.50 else "Medium" if max_score >= 0.2628 else "Low" if max_score > 0 else "Low"

    return {
        "node_id": node_id,
        "bank": bank,
        "account": account,
        "transaction_count": total_txns,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "risk_level": risk_level,
        "highest_score": max_score,
        "historical_behavior": {
            "pagerank_log": acc_details.get("pagerank_log", 0.0),
            "wcc_size_log": acc_details.get("wcc_size_log", 0.0),
            "unique_out_log": acc_details.get("unique_out_log", 0.0),
            "unique_in_log": acc_details.get("unique_in_log", 0.0),
            "has_sent_before": acc_details.get("has_sent_before", 0),
            "has_received_before": acc_details.get("has_received_before", 0),
        },
        "alerted_transactions": alerted_list,
    }