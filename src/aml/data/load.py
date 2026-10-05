"""P1. Read the raw IBM CSV safely and assign txn_id FIRST (before any sorting/filtering)."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

EXPECTED_HEADER = ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
                   "Amount Paid,Payment Currency,Payment Format,Is Laundering")
# The CSV has TWO columns named 'Account', so we name columns ourselves (header=0, names=...).
NAMES = ["ts", "from_bank", "from_account", "to_bank", "to_account", "amount_received",
         "cur_recv", "amount_paid", "cur_paid", "payment_format", "is_laundering"]
DTYPES = {
    "from_bank": "category", "from_account": "category",   # keep as strings: leading zeros ('010')
    "to_bank": "category", "to_account": "category",
    "cur_recv": "category", "cur_paid": "category", "payment_format": "category",
    "amount_received": "float64", "amount_paid": "float64", "is_laundering": "int8",
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_raw(path: Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip()
    if header != EXPECTED_HEADER:
        raise ValueError(f"Unexpected CSV header:\n  got      {header}\n  expected {EXPECTED_HEADER}")
    df = pd.read_csv(path, header=0, names=NAMES, dtype=DTYPES, nrows=nrows)
    # txn_id = original 0-based data-row index. Assigned BEFORE anything else. Never recomputed.
    df.insert(0, "txn_id", np.arange(len(df), dtype="int32"))
    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M")
    return df
