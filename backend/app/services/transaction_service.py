from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

TRANSACTIONS_FILE = DATA_DIR / "transactions_clean.parquet"


TRANSACTION_COLUMNS = [
    "txn_id",
    "ts",
    "date",
    "src_node",
    "dst_node",
    "src_bank",
    "dst_bank",
    "amount_paid",
    "amount_received",
    "cur_paid",
    "cur_recv",
    "payment_format",
    "is_laundering",
    "split",
    "is_self_loop",
]


def get_transaction(txn_id: int):
    transactions = pd.read_parquet(
        TRANSACTIONS_FILE,
        columns=TRANSACTION_COLUMNS,
        filters=[("txn_id", "==", txn_id)],
    )

    if transactions.empty:
        return None

    row = transactions.iloc[0]

    return {
        "txn_id": int(row["txn_id"]),
        "timestamp": str(row["ts"]),
        "date": str(row["date"]),
        "src_node": int(row["src_node"]),
        "dst_node": int(row["dst_node"]),
        "src_bank": str(row["src_bank"]),
        "dst_bank": str(row["dst_bank"]),
        "amount_paid": float(row["amount_paid"]),
        "amount_received": float(row["amount_received"]),
        "currency_paid": str(row["cur_paid"]),
        "currency_received": str(row["cur_recv"]),
        "payment_format": str(row["payment_format"]),
        "is_laundering": int(row["is_laundering"]),
        "split": str(row["split"]),
        "is_self_loop": bool(row["is_self_loop"]),
    }