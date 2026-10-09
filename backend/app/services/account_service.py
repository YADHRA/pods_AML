from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

GRAPH_NODE_DAY_FILE = DATA_DIR / "graph_node_day.parquet"
NODE_MAP_FILE = DATA_DIR / "node_map.parquet"


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


def get_account(node_id: int, date: str):
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
        return None

    row = graph.iloc[0]

    node_map = pd.read_parquet(
        NODE_MAP_FILE,
        columns=["node_id", "bank", "account"],
        filters=[("node_id", "==", node_id)],
    )

    if node_map.empty:
        return None

    node_row = node_map.iloc[0]

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