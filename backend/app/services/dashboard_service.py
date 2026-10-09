from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

TRANSACTIONS_FILE = DATA_DIR / "transactions_clean.parquet"
NODE_MAP_FILE = DATA_DIR / "node_map.parquet"


def get_dashboard_summary():
    # Read only the columns needed for the dashboard.
    transactions = pd.read_parquet(
        TRANSACTIONS_FILE,
        columns=[
            "txn_id",
            "date",
            "payment_format",
            "is_laundering",
        ],
    )

    nodes = pd.read_parquet(
        NODE_MAP_FILE,
        columns=["node_id"],
    )

    total_transactions = len(transactions)
    total_nodes = len(nodes)

    laundering_count = int(
        transactions["is_laundering"].sum()
    )

    normal_count = int(
        (transactions["is_laundering"] == 0).sum()
    )

    date_min = transactions["date"].min()
    date_max = transactions["date"].max()

    payment_format_counts = (
        transactions["payment_format"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "total_transactions": total_transactions,
        "total_nodes": total_nodes,
        "date_range": {
            "start": str(date_min.date()),
            "end": str(date_max.date()),
        },
        "labels": {
            "normal": normal_count,
            "laundering": laundering_count,
        },
        "payment_formats": {
            str(k): int(v)
            for k, v in payment_format_counts.items()
        },
    }