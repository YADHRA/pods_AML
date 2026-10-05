"""P1. Transaction-level features (t_*): known from the transaction itself, no history needed.
Statistics (per-currency amount mean/std) are fit on TRAIN rows only, then applied to every row."""
import json
import numpy as np
import pandas as pd

COLS = ["txn_id", "ts", "amount_paid", "cur_paid", "cur_recv", "payment_format",
        "src_bank", "dst_bank", "is_self_loop", "split"]


def fit_currency_stats(df: pd.DataFrame) -> dict:
    """mean/std of log10(amount_paid) per paid-currency, from split == 'train' ONLY."""
    tr = df[df["split"] == "train"]
    g = np.log10(tr["amount_paid"]).groupby(tr["cur_paid"]).agg(["mean", "std"])
    g["std"] = g["std"].fillna(1.0).replace(0.0, 1.0)
    return {k: {"mean": float(r["mean"]), "std": float(r["std"])} for k, r in g.iterrows()}


def transform(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    logamt = np.log10(df["amount_paid"].to_numpy())
    mean = df["cur_paid"].map({k: v["mean"] for k, v in stats.items()}).astype("float64").to_numpy()
    std = df["cur_paid"].map({k: v["std"] for k, v in stats.items()}).astype("float64").to_numpy()
    return pd.DataFrame({
        "txn_id": df["txn_id"].to_numpy(),
        "t_log_amount_paid": logamt.astype("float32"),
        "t_amount_z_in_currency": ((logamt - mean) / std).astype("float32"),  # NaN if currency unseen in train
        "t_payment_format": df["payment_format"].astype("string"),
        "t_cur_paid": df["cur_paid"].astype("string"),
        "t_is_fx": (df["cur_paid"] != df["cur_recv"]).to_numpy().astype("int8"),
        "t_is_self_loop": df["is_self_loop"].to_numpy().astype("int8"),
        "t_same_bank": (df["src_bank"] == df["dst_bank"]).to_numpy().astype("int8"),
        "t_hour": df["ts"].dt.hour.to_numpy().astype("int8"),
    })


def build_txn_features(clean: pd.DataFrame):
    stats = fit_currency_stats(clean)
    return transform(clean, stats), stats


def save_stats(stats: dict, path) -> None:
    path.write_text(json.dumps(stats, indent=2))