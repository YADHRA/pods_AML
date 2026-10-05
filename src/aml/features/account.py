"""Step 3 - account-history features.

One row per (feature_date, node_id) for every node that appears in a prediction row
(train / val / test / tail, any transaction including self-loops) on that date.
Feature values use ONLY non-self transactions on days STRICTLY BEFORE feature_date.

Design: build per-(node, day) daily totals once, take a cumulative sum per node, then
look up each (node, D) with merge_asof(allow_exact_matches=False) = "last active day < D".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z = "t_amount_z_in_currency"

FEATURES = [
    "a_n_out_rate",
    "a_n_in_rate",
    "a_mean_z_out",
    "a_mean_z_in",
    "a_max_z_out",
    "a_days_active_frac",
    "a_days_since_last",
    "a_fmt_diversity",
]

REQUIRED = ["ts", "src_node", "dst_node", "payment_format", "is_self_loop", "split", Z]


def build_account_node_day(df: pd.DataFrame) -> pd.DataFrame:
    """df: ALL rows (incl. warm-up) with the columns in REQUIRED. Returns account_node_day."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    split = df["split"].astype(str).to_numpy()
    if "warmup" not in set(np.unique(split)):
        raise ValueError("split column has no 'warmup' value - check the split labels")

    ts = pd.to_datetime(df["ts"])
    base = ts.min().normalize()
    day = (ts.dt.normalize() - base).dt.days.to_numpy().astype(np.int32)  # Sep 1 -> 0
    src = df["src_node"].to_numpy().astype(np.int32)
    dst = df["dst_node"].to_numpy().astype(np.int32)
    is_self = df["is_self_loop"].to_numpy().astype(bool)
    pred = split != "warmup"

    # ---- history table: non-self transactions only ----
    m = ~is_self
    h = pd.DataFrame({
        "src": src[m],
        "dst": dst[m],
        "day": day[m],
        "fmt": pd.factorize(df["payment_format"].to_numpy()[m])[0].astype(np.int16),
        "z": df[Z].to_numpy(dtype="float64")[m],
    })

    # ---- daily totals per (node, day) ----
    g = h.groupby(["src", "day"], sort=False)["z"]
    out = pd.DataFrame({"out_n": g.size(), "out_zn": g.count(),
                        "out_zsum": g.sum(), "out_zmax": g.max()})
    out.index.names = ["node", "day"]
    g = h.groupby(["dst", "day"], sort=False)["z"]
    inn = pd.DataFrame({"in_n": g.size(), "in_zn": g.count(), "in_zsum": g.sum()})
    inn.index.names = ["node", "day"]

    act = out.join(inn, how="outer")
    cnt_cols = ["out_n", "out_zn", "out_zsum", "in_n", "in_zn", "in_zsum"]
    act[cnt_cols] = act[cnt_cols].fillna(0.0)
    act["out_zmax"] = act["out_zmax"].fillna(-np.inf)

    # distinct formats (sent + received combined): count each (node, format) on its first day
    nd = pd.concat([
        h[["src", "fmt", "day"]].rename(columns={"src": "node"}),
        h[["dst", "fmt", "day"]].rename(columns={"dst": "node"}),
    ], ignore_index=True)
    first = nd.groupby(["node", "fmt"], sort=False)["day"].min().reset_index()
    newf = first.groupby(["node", "day"]).size().rename("new_fmt")
    act = act.join(newf, how="left")
    act["new_fmt"] = act["new_fmt"].fillna(0.0)
    act["active"] = 1.0

    act = act.reset_index().sort_values(["node", "day"], kind="stable")
    grp = act.groupby("node", sort=False)
    cum_cols = cnt_cols + ["new_fmt", "active"]
    for c in cum_cols:
        act["c_" + c] = grp[c].cumsum()
    act["c_out_zmax"] = grp["out_zmax"].cummax()
    act["last_day"] = act["day"]

    right = act[["node", "day", "last_day"] + ["c_" + c for c in cum_cols] + ["c_out_zmax"]].copy()
    right["node"] = right["node"].astype(np.int32)
    right["day"] = right["day"].astype(np.int32)
    right = right.sort_values("day", kind="stable").reset_index(drop=True)

    # ---- query keys: every node seen in a prediction row, per day ----
    q = pd.concat([
        pd.DataFrame({"node": src[pred], "day": day[pred]}),
        pd.DataFrame({"node": dst[pred], "day": day[pred]}),
    ]).drop_duplicates()
    q["node"] = q["node"].astype(np.int32)
    q["day"] = q["day"].astype(np.int32)
    q = q.sort_values("day", kind="stable").reset_index(drop=True)

    # last active day strictly BEFORE the query day
    mm = pd.merge_asof(q, right, on="day", by="node",
                       allow_exact_matches=False, direction="backward")

    nh = mm["day"].to_numpy(dtype="float64")  # number of history days (Sep 1 = day 0)
    if nh.size and nh.min() < 1:
        raise AssertionError("prediction row on day 0: warm-up day leaked into predictions")

    def col(name):
        return mm[name].fillna(0.0).to_numpy(dtype="float64")

    c_out_n, c_in_n = col("c_out_n"), col("c_in_n")
    c_out_zn, c_in_zn = col("c_out_zn"), col("c_in_zn")
    c_out_zs, c_in_zs = col("c_out_zsum"), col("c_in_zsum")
    zmax = mm["c_out_zmax"].to_numpy(dtype="float64")
    zmax = np.where(np.isfinite(zmax), zmax, np.nan)

    res = pd.DataFrame({
        "feature_date": (base + pd.to_timedelta(mm["day"].to_numpy(), unit="D")).astype("datetime64[ns]"),
        "node_id": mm["node"].to_numpy().astype(np.int32),
        "a_n_out_rate": np.log1p(c_out_n / nh),
        "a_n_in_rate": np.log1p(c_in_n / nh),
        "a_mean_z_out": np.where(c_out_zn > 0, c_out_zs / np.maximum(c_out_zn, 1), np.nan),
        "a_mean_z_in": np.where(c_in_zn > 0, c_in_zs / np.maximum(c_in_zn, 1), np.nan),
        "a_max_z_out": zmax,
        "a_days_active_frac": col("c_active") / nh,
        "a_days_since_last": nh - mm["last_day"].to_numpy(dtype="float64"),
        "a_fmt_diversity": col("c_new_fmt"),
    })
    res[FEATURES] = res[FEATURES].astype(np.float32)
    res = res.sort_values(["feature_date", "node_id"]).reset_index(drop=True)
    if res.duplicated(["feature_date", "node_id"]).any():
        raise AssertionError("duplicate (feature_date, node_id) keys")
    return res