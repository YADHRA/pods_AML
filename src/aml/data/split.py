"""P1. Build transactions_clean + node_map and assign chronological split labels."""
import numpy as np
import pandas as pd
from aml.common import schema as S


def assign_split(date: pd.Series, cfg: dict) -> pd.Series:
    conds, names = [], []
    for name, (lo, hi) in cfg["split"].items():
        conds.append((date >= pd.Timestamp(lo)) & (date <= pd.Timestamp(hi)))
        names.append(name)
    out = np.select(conds, names, default="UNASSIGNED")
    assert (out != "UNASSIGNED").all(), "some dates fall outside all split ranges"
    return pd.Series(out, index=date.index, dtype="string")


def _joint_codes(a: pd.Series, b: pd.Series):
    cats = pd.Index(a.cat.categories.union(b.cat.categories))
    ca = pd.Categorical(a.astype(object), categories=cats).codes.astype("int64")
    cb = pd.Categorical(b.astype(object), categories=cats).codes.astype("int64")
    return ca, cb, cats


def build_clean(raw: pd.DataFrame, cfg: dict):
    """Node = (bank, account). Composite integer key avoids building 10M strings."""
    bank_s, bank_d, bank_cats = _joint_codes(raw["from_bank"], raw["to_bank"])
    acc_s, acc_d, acc_cats = _joint_codes(raw["from_account"], raw["to_account"])
    n_acc = len(acc_cats)
    key_s, key_d = bank_s * n_acc + acc_s, bank_d * n_acc + acc_d
    codes, uniques = pd.factorize(np.concatenate([key_s, key_d]))
    n = len(raw)
    src, dst = codes[:n].astype("int32"), codes[n:].astype("int32")
    node_map = pd.DataFrame({
        "node_id": np.arange(len(uniques), dtype="int32"),
        "bank": pd.array(bank_cats.to_numpy()[uniques // n_acc], dtype="string"),
        "account": pd.array(acc_cats.to_numpy()[uniques % n_acc], dtype="string"),
    })
    date = raw["ts"].dt.floor("D")
    clean = pd.DataFrame({
        "txn_id": raw["txn_id"].to_numpy(),
        "ts": raw["ts"].astype("datetime64[ns]"), "date": date.astype("datetime64[ns]"),
        "src_node": src, "dst_node": dst,
        "src_bank": pd.array(node_map["bank"].to_numpy()[src], dtype="string"),
        "dst_bank": pd.array(node_map["bank"].to_numpy()[dst], dtype="string"),
        "amount_paid": raw["amount_paid"].to_numpy(), "amount_received": raw["amount_received"].to_numpy(),
        "cur_paid": raw["cur_paid"].astype("string"), "cur_recv": raw["cur_recv"].astype("string"),
        "payment_format": raw["payment_format"].astype("string"),
        "is_laundering": raw["is_laundering"].to_numpy(),
        "split": assign_split(date, cfg),
        "is_self_loop": src == dst,
    })
    clean = clean.sort_values(["ts", "txn_id"], kind="stable").reset_index(drop=True)
    clean = clean[S.TRANSACTIONS_CLEAN_COLS].astype(
        {k: v for k, v in S.TRANSACTIONS_CLEAN_DTYPES.items() if k not in ("ts", "date")})
    S.assert_transactions_clean(clean)
    return clean, node_map


def split_summary(clean: pd.DataFrame) -> list[dict]:
    g = clean.groupby("split").agg(n_rows=("txn_id", "size"), n_pos=("is_laundering", "sum"),
                                   date_min=("date", "min"), date_max=("date", "max"))
    g["pos_rate"] = g["n_pos"] / g["n_rows"]
    g = g.reindex(S.SPLITS).reset_index()
    g["date_min"], g["date_max"] = g["date_min"].astype(str), g["date_max"].astype(str)
    return g.astype({"n_rows": int, "n_pos": int}).to_dict("records")


def daily_summary(clean: pd.DataFrame) -> list[dict]:
    g = clean.groupby("date").agg(n_rows=("txn_id", "size"), n_pos=("is_laundering", "sum"))
    g["pos_rate"] = g["n_pos"] / g["n_rows"]
    g = g.reset_index()
    g["date"] = g["date"].dt.strftime("%Y-%m-%d")
    return g.astype({"n_rows": int, "n_pos": int}).to_dict("records")
