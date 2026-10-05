"""Step 4 - assemble the model matrices.

Reads (from data/processed):
  transactions_clean.parquet, txn_features.parquet, account_node_day.parquet
  and, once Person 2 delivers them, graph_node_day.parquet + graph_pair_txn.parquet.

Joins ONLY on txn_id or (date, node_id) - never by row position.
Warm-up rows (Sep 1) are dropped. Both matrices come from the SAME wide table, so
they have identical rows in identical order (sorted by ts, txn_id).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from aml.common import paths

ID_COLS = ["txn_id", "split", "is_laundering"]
SPLIT_LABELS = {"warmup", "train", "val", "test", "tail"}

# Locked split numbers (from the real file) - assemble stops if they differ.
EXPECTED_ROWS = {"train": 2_134_000, "val": 965_524, "test": 862_792, "tail": 1_108}
EXPECTED_POS = {"train": 2_208, "val": 1_036, "test": 956, "tail": 655}


# ---------------------------------------------------------------- config
def load_feature_sets(path=None) -> dict:
    path = path or (paths.CONFIGS / "feature_sets.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_columns(fs: dict):
    """Returns (m1_cols, m2_cols). M2 = M1 + account + pair + network."""
    m1 = list(fs["m1_transaction"])
    m2 = m1 + list(fs["group_account"]) + list(fs["group_pair"]) + list(fs["group_network"])
    if len(set(m2)) != len(m2):
        raise ValueError("feature_sets.yaml has duplicate column names")
    return m1, m2


# ---------------------------------------------------------------- helpers
def check_sorted(clean: pd.DataFrame) -> None:
    """transactions_clean must be sorted by (ts, txn_id) - the contract."""
    ts = clean["ts"].to_numpy()
    tid = clean["txn_id"].to_numpy()
    ok = (ts[1:] > ts[:-1]) | ((ts[1:] == ts[:-1]) & (tid[1:] > tid[:-1]))
    if not ok.all():
        raise AssertionError("transactions_clean is not sorted by (ts, txn_id)")


def _sided(col: str, side: str) -> str:
    """a_n_out_rate -> a_src_n_out_rate ; g_pagerank_log -> g_src_pagerank_log"""
    prefix, rest = col.split("_", 1)
    return f"{prefix}_{side}_{rest}"


def _needed(side: str, yaml_cols, prefix: str):
    """Unsided column names the YAML needs for one side, e.g. ('src', a) -> a_n_out_rate ..."""
    stem = f"{prefix}_{side}_"
    return [f"{prefix}_" + c[len(stem):] for c in yaml_cols if c.startswith(stem)]


def _join_node_side(base: pd.DataFrame, table: pd.DataFrame, node_col: str, side: str,
                    cols, what: str) -> pd.DataFrame:
    """Look up `cols` of table[(feature_date,node_id)] for base[(_date,node_col)].
    Returns only the new (sided) columns, aligned to base rows. Missing key -> error."""
    right = table[["feature_date", "node_id"] + list(cols)].copy()
    right["feature_date"] = pd.to_datetime(right["feature_date"]).astype("datetime64[ns]")
    right = right.rename(columns={c: _sided(c, side) for c in cols})
    left = pd.DataFrame({"feature_date": base["_date"].to_numpy(),
                         "node_id": base[node_col].to_numpy()})
    out = left.merge(right, on=["feature_date", "node_id"], how="left",
                     validate="many_to_one", indicator=True)
    if len(out) != len(base):
        raise AssertionError(f"{what}: join changed the row count")
    miss = out["_merge"] != "both"
    if miss.any():
        ex = left[miss.to_numpy()].head(3).to_dict("records")
        raise ValueError(f"{what}: {int(miss.sum())} rows ({side} side) have no (feature_date, node_id) "
                         f"match, e.g. {ex}")
    return out.drop(columns=["feature_date", "node_id", "_merge"])


def _join_pair(base: pd.DataFrame, pair: pd.DataFrame, cols) -> pd.DataFrame:
    right = pair[["txn_id"] + list(cols)]
    out = base[["txn_id"]].merge(right, on="txn_id", how="left",
                                 validate="many_to_one", indicator=True)
    if len(out) != len(base):
        raise AssertionError("graph_pair_txn: join changed the row count")
    miss = out["_merge"] != "both"
    if miss.any():
        raise ValueError(f"graph_pair_txn: {int(miss.sum())} prediction rows have no txn_id match, "
                         f"e.g. {out.loc[miss, 'txn_id'].head(3).tolist()}")
    return out.drop(columns=["txn_id", "_merge"])


def to_train_categories(wide: pd.DataFrame, cols) -> pd.DataFrame:
    """Make `cols` pandas categoricals whose category list comes from TRAIN rows only.
    A value in val/test/tail that never appears in train is an error."""
    is_train = (wide["split"] == "train").to_numpy()
    for c in cols:
        s = wide[c].astype(str)
        cats = sorted(s[is_train].unique())
        codes = pd.Index(cats).get_indexer(s)
        unseen = codes < 0
        if unseen.any():
            bad = sorted(s[unseen].unique())[:5]
            raise ValueError(f"{c}: values not in train categories: {bad}")
        wide[c] = pd.Categorical.from_codes(codes, categories=cats)
    return wide


def check_counts(wide: pd.DataFrame, expected_rows=EXPECTED_ROWS, expected_pos=EXPECTED_POS) -> None:
    got_rows = wide["split"].value_counts().to_dict()
    got_pos = wide.groupby("split")["is_laundering"].sum().astype(int).to_dict()
    bad = []
    for s in expected_rows:
        if got_rows.get(s, 0) != expected_rows[s]:
            bad.append(f"{s}: rows {got_rows.get(s, 0):,} != expected {expected_rows[s]:,}")
        if got_pos.get(s, 0) != expected_pos[s]:
            bad.append(f"{s}: positives {got_pos.get(s, 0):,} != expected {expected_pos[s]:,}")
    extra = set(got_rows) - set(expected_rows)
    if extra:
        bad.append(f"unexpected split labels: {sorted(extra)}")
    if bad:
        raise AssertionError("split counts do not match the locked numbers:\n  " + "\n  ".join(bad))


# ---------------------------------------------------------------- main build
def build_wide(clean, txf, acc, pair=None, gnd=None, fs=None) -> pd.DataFrame:
    """Wide table: ID_COLS + every feature column available (M1 + A, and P + C if graph tables given).
    Prediction rows only (warm-up dropped), in (ts, txn_id) order."""
    fs = fs or load_feature_sets()
    m1_cols, m2_cols = model_columns(fs)
    if (pair is None) != (gnd is None):
        raise ValueError("pass BOTH graph tables (pair and node-day) or neither")

    check_sorted(clean)
    labels = set(clean["split"].astype(str).unique())
    if labels != SPLIT_LABELS:
        raise ValueError(f"unexpected split labels {sorted(labels)}; expected {sorted(SPLIT_LABELS)}")

    keep = clean["split"].astype(str) != "warmup"
    base = clean.loc[keep, ["txn_id", "date", "src_node", "dst_node", "is_laundering", "split"]]
    base = base.reset_index(drop=True)
    base["split"] = base["split"].astype(str)

    # transaction features: join on txn_id only
    base = base.merge(txf[["txn_id"] + m1_cols], on="txn_id", how="left", validate="one_to_one")
    nan_cols = [c for c in m1_cols if base[c].isna().any()]
    if nan_cols:
        raise ValueError(f"transaction features missing/NaN after join: {nan_cols}")
    base["_date"] = pd.to_datetime(base["date"]).astype("datetime64[ns]")

    new = []
    # account features: (date, node)
    for side, node_col in (("src", "src_node"), ("dst", "dst_node")):
        cols = _needed(side, fs["group_account"], "a")
        new.append(_join_node_side(base, acc, node_col, side, cols, "account_node_day"))
    # graph features (only if delivered)
    if gnd is not None:
        new.append(_join_pair(base, pair, fs["group_pair"]))
        for side, node_col in (("src", "src_node"), ("dst", "dst_node")):
            cols = _needed(side, fs["group_network"], "g")
            new.append(_join_node_side(base, gnd, node_col, side, cols, "graph_node_day"))

    wide = pd.concat([base] + new, axis=1)
    want = m2_cols if gnd is not None else (m1_cols + list(fs["group_account"]))
    missing = [c for c in want if c not in wide.columns]
    if missing:
        raise KeyError(f"columns listed in feature_sets.yaml but not produced: {missing}")
    wide = wide[ID_COLS + want]
    return to_train_categories(wide, fs["categorical"])


# ---------------------------------------------------------------- drift
def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max gap between the two CDFs). 0 = same, 1 = disjoint."""
    a = np.sort(a)
    b = np.sort(b)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / a.size
    cdf_b = np.searchsorted(b, allv, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def drift_report(wide: pd.DataFrame, cols, a="train", b="val", n=200_000, seed=42):
    """Train-vs-val drift per numeric feature (KS statistic on a seeded sample, NaN share, means)."""
    rng = np.random.default_rng(seed)
    A = wide[wide["split"] == a]
    B = wide[wide["split"] == b]
    rows = []
    for c in cols:
        if str(wide[c].dtype) == "category":
            continue
        xa = A[c].to_numpy(dtype="float64")
        xb = B[c].to_numpy(dtype="float64")
        na, nb = float(np.isnan(xa).mean()), float(np.isnan(xb).mean())
        xa, xb = xa[~np.isnan(xa)], xb[~np.isnan(xb)]
        ks = np.nan
        if xa.size and xb.size:
            if xa.size > n:
                xa = xa[rng.choice(xa.size, n, replace=False)]
            if xb.size > n:
                xb = xb[rng.choice(xb.size, n, replace=False)]
            ks = ks_stat(xa, xb)
        rows.append({"feature": c, "ks": ks,
                     "nan_share_train": na, "nan_share_val": nb,
                     "mean_train": float(xa.mean()) if xa.size else None,
                     "mean_val": float(xb.mean()) if xb.size else None})
    rows.sort(key=lambda r: (-1 if np.isnan(r["ks"]) else r["ks"]), reverse=True)
    return rows