"""SHARED CONTRACT. Change only via a `contract/*` PR approved by both people."""
import pandas as pd

SPLITS = ["warmup", "train", "val", "test", "tail"]

# ---- P1 -> P2 -------------------------------------------------------------
TRANSACTIONS_CLEAN_DTYPES = {
    "txn_id": "int32",        # original 0-based CSV data-row index. NEVER recomputed.
    "ts": "datetime64[ns]",
    "date": "datetime64[ns]", # ts floored to midnight
    "src_node": "int32",      # (from_bank, from_account) -> node_id
    "dst_node": "int32",
    "src_bank": "string",     # strings: bank codes have leading zeros
    "dst_bank": "string",
    "amount_paid": "float64",
    "amount_received": "float64",
    "cur_paid": "string",
    "cur_recv": "string",
    "payment_format": "string",
    "is_laundering": "int8",
    "split": "string",        # one of SPLITS
    "is_self_loop": "bool",   # src_node == dst_node
}
TRANSACTIONS_CLEAN_COLS = list(TRANSACTIONS_CLEAN_DTYPES)

NODE_MAP_DTYPES = {"node_id": "int32", "bank": "string", "account": "string"}

# ---- P2 -> P1 -------------------------------------------------------------
# key: (feature_date, node_id); values computed from date < feature_date, non-self edges only
GRAPH_NODE_DAY_KEYS = ["feature_date", "node_id"]
GRAPH_NODE_DAY_COLS = [
    "g_pagerank_log", "g_wcc_size_log", "g_n_unique_out_log",
    "g_n_unique_in_log", "g_has_sent_before", "g_has_received_before",
]
# key: txn_id; rows only for split in {train,val,test,tail}
GRAPH_PAIR_TXN_KEYS = ["txn_id"]
GRAPH_PAIR_TXN_COLS = ["g_pair_seen_before", "g_pair_prior_count_log", "g_reverse_pair_seen"]


def assert_transactions_clean(df: pd.DataFrame) -> None:
    missing = [c for c in TRANSACTIONS_CLEAN_COLS if c not in df.columns]
    assert not missing, f"transactions_clean missing columns: {missing}"
    assert df["txn_id"].is_unique, "txn_id must be unique"
    bad = set(df["split"].unique()) - set(SPLITS)
    assert not bad, f"unknown split labels: {bad}"
    assert df["is_laundering"].isin([0, 1]).all(), "is_laundering must be 0/1"
    assert (df["is_self_loop"] == (df["src_node"] == df["dst_node"])).all(), "is_self_loop inconsistent"
    t = df["ts"].to_numpy().astype("datetime64[ns]").astype("int64")
    i = df["txn_id"].to_numpy().astype("int64")
    ok = (t[1:] > t[:-1]) | ((t[1:] == t[:-1]) & (i[1:] > i[:-1]))
    assert ok.all(), "must be sorted by (ts, txn_id)"


def assert_graph_node_day(df: pd.DataFrame) -> None:
    need = GRAPH_NODE_DAY_KEYS + GRAPH_NODE_DAY_COLS
    missing = [c for c in need if c not in df.columns]
    assert not missing, f"graph_node_day missing columns: {missing}"
    assert not df.duplicated(GRAPH_NODE_DAY_KEYS).any(), "graph_node_day keys must be unique"


def assert_graph_pair_txn(df: pd.DataFrame) -> None:
    need = GRAPH_PAIR_TXN_KEYS + GRAPH_PAIR_TXN_COLS
    missing = [c for c in need if c not in df.columns]
    assert not missing, f"graph_pair_txn missing columns: {missing}"
    assert df["txn_id"].is_unique, "graph_pair_txn txn_id must be unique"
