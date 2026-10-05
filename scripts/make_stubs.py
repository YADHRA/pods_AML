"""Write tiny, schema-valid FAKE files into data_stub/ so P1 and P2 can build in parallel.
Values are random; only columns/dtypes/keys are real. Usage: python scripts/make_stubs.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aml.common import schema as S  # noqa: E402
from aml.common.paths import STUB  # noqa: E402

FORMATS = ["ACH", "Cheque", "Credit Card", "Cash", "Reinvestment", "Wire", "Bitcoin"]
CURRENCIES = ["US Dollar", "Euro", "Saudi Riyal", "Bitcoin"]


def split_for(d: pd.Timestamp) -> str:
    day = d.day
    return ("warmup" if day == 1 else "train" if day <= 6 else "val" if day <= 8
            else "test" if day <= 10 else "tail")


def build(n=3000, n_nodes=200, seed=0, out: Path = STUB) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp("2022-09-01") + pd.to_timedelta(rng.integers(0, 18 * 24 * 60, n), unit="m")
    src = rng.integers(0, n_nodes, n).astype("int32")
    dst = rng.integers(0, n_nodes, n).astype("int32")
    node_map = pd.DataFrame({
        "node_id": np.arange(n_nodes, dtype="int32"),
        "bank": [f"{b:03d}" for b in rng.integers(1, 40, n_nodes)],   # leading zeros on purpose
        "account": [f"8{i:07X}" for i in range(n_nodes)],
    }).astype({"bank": "string", "account": "string"})
    df = pd.DataFrame({
        "txn_id": rng.permutation(n).astype("int32"),
        "ts": ts, "date": ts.floor("D"),
        "src_node": src, "dst_node": dst,
        "src_bank": node_map.bank.values[src], "dst_bank": node_map.bank.values[dst],
        "amount_paid": np.round(rng.lognormal(7, 2, n), 2),
        "cur_paid": rng.choice(CURRENCIES, n), "cur_recv": rng.choice(CURRENCIES, n),
        "payment_format": rng.choice(FORMATS, n),
        "is_laundering": (rng.random(n) < 0.02).astype("int8"),
    })
    df["amount_received"] = df["amount_paid"]
    df["split"] = [split_for(d) for d in df["date"]]
    df["is_self_loop"] = df["src_node"] == df["dst_node"]
    df = df.sort_values(["ts", "txn_id"]).reset_index(drop=True)
    df = df[S.TRANSACTIONS_CLEAN_COLS].astype(
        {k: v for k, v in S.TRANSACTIONS_CLEAN_DTYPES.items() if k not in ("ts", "date")})
    S.assert_transactions_clean(df)
    df.to_parquet(out / "transactions_clean.parquet", index=False)
    node_map.to_parquet(out / "node_map.parquet", index=False)

    # fake P2 outputs (so P1 can test assemble.py before P2 delivers)
    days = pd.date_range("2022-09-02", "2022-09-18")
    gnd = pd.MultiIndex.from_product([days, node_map.node_id], names=S.GRAPH_NODE_DAY_KEYS).to_frame(index=False)
    for c in S.GRAPH_NODE_DAY_COLS:
        gnd[c] = (rng.integers(0, 2, len(gnd)) if "has_" in c else rng.random(len(gnd))).astype("float32")
    S.assert_graph_node_day(gnd)
    gnd.to_parquet(out / "graph_node_day.parquet", index=False)

    pred = df.loc[df.split != "warmup", ["txn_id"]].copy()
    pred["g_pair_seen_before"] = rng.integers(0, 2, len(pred)).astype("int8")
    pred["g_pair_prior_count_log"] = rng.random(len(pred)).astype("float32")
    pred["g_reverse_pair_seen"] = rng.integers(0, 2, len(pred)).astype("int8")
    S.assert_graph_pair_txn(pred)
    pred.to_parquet(out / "graph_pair_txn.parquet", index=False)


if __name__ == "__main__":
    build()
    print("stubs written to", STUB)
