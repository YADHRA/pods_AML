import numpy as np
import pandas as pd
import pytest

from aml.features.account import build_account_node_day
from aml.features import assemble as A

DAY_SPLIT = {0: "warmup", 1: "train", 2: "train", 3: "train", 4: "val", 5: "test", 6: "tail", 7: "tail"}


def make_world(seed=0, n=3000, n_nodes=30, n_days=8):
    rng = np.random.default_rng(seed)
    day = rng.integers(0, n_days, n)
    ts = pd.Timestamp("2022-09-01") + pd.to_timedelta(day, unit="D") + pd.to_timedelta(rng.integers(0, 1440, n), unit="m")
    src = rng.integers(0, n_nodes, n).astype(np.int32)
    dst = rng.integers(0, n_nodes, n).astype(np.int32)
    dst = np.where(rng.random(n) < 0.12, src, dst).astype(np.int32)
    fmt = rng.choice(["ACH", "Cheque", "Wire"], n)
    clean = pd.DataFrame({
        "txn_id": np.arange(n, dtype=np.int32), "ts": ts, "date": ts.normalize(),
        "src_node": src, "dst_node": dst, "payment_format": fmt,
        "is_laundering": rng.random(n) < 0.05, "is_self_loop": src == dst,
        "split": [DAY_SPLIT[d] for d in day],
    })
    # sort by (ts, txn_id): txn_id is NOT monotonic afterwards, so a positional join would be wrong
    clean = clean.sort_values(["ts", "txn_id"]).reset_index(drop=True)

    txf = pd.DataFrame({
        "txn_id": np.arange(n, dtype=np.int32),
        "t_log_amount_paid": rng.normal(3, 1, n),
        "t_amount_z_in_currency": rng.normal(0, 1, n),
        "t_payment_format": fmt,
        "t_cur_paid": rng.choice(["USD", "EUR", "Yen"], n),
        "t_is_fx": rng.integers(0, 2, n).astype(float),
        "t_is_self_loop": (src == dst).astype(float),
        "t_same_bank": rng.integers(0, 2, n).astype(float),
        "t_hour": rng.integers(0, 24, n).astype(float),
    }).sample(frac=1.0, random_state=1).reset_index(drop=True)  # shuffled on purpose

    df = clean.merge(txf[["txn_id", "t_amount_z_in_currency"]], on="txn_id")
    acc = build_account_node_day(df)

    pred = clean[clean["split"] != "warmup"]
    keys = pd.concat([
        pd.DataFrame({"feature_date": pred["date"], "node_id": pred["src_node"]}),
        pd.DataFrame({"feature_date": pred["date"], "node_id": pred["dst_node"]}),
    ]).drop_duplicates().reset_index(drop=True)
    d = (keys["feature_date"] - pd.Timestamp("2022-09-01")).dt.days.to_numpy()
    gnd = keys.copy()
    gnd["g_pagerank_log"] = (gnd["node_id"] * 0.001 + d * 0.1).astype(np.float32)
    gnd["g_wcc_size_log"] = (gnd["node_id"] * 0.01 + d).astype(np.float32)
    gnd["g_n_unique_out_log"] = (gnd["node_id"] % 7).astype(np.float32)
    gnd["g_n_unique_in_log"] = (gnd["node_id"] % 5).astype(np.float32)
    gnd["g_has_sent_before"] = (gnd["node_id"] % 2).astype(np.int8)
    gnd["g_has_received_before"] = ((gnd["node_id"] + 1) % 2).astype(np.int8)

    pair = pd.DataFrame({
        "txn_id": pred["txn_id"].to_numpy(),
        "g_pair_seen_before": (pred["txn_id"].to_numpy() % 2).astype(np.int8),
        "g_pair_prior_count_log": (pred["txn_id"].to_numpy() % 9).astype(np.float32),
        "g_reverse_pair_seen": (pred["txn_id"].to_numpy() % 3 == 0).astype(np.int8),
    }).sample(frac=1.0, random_state=2).reset_index(drop=True)
    return clean, txf, acc, pair, gnd


@pytest.fixture(scope="module")
def world():
    return make_world()


A_COLS = ["n_out_rate", "n_in_rate", "mean_z_out", "mean_z_in", "max_z_out",
          "days_active_frac", "days_since_last", "fmt_diversity"]


def test_columns_match_yaml_and_warmup_dropped(world):
    clean, txf, acc, pair, gnd = world
    fs = A.load_feature_sets()
    m1, m2 = A.model_columns(fs)
    wide = A.build_wide(clean, txf, acc, pair, gnd)
    assert list(wide.columns) == A.ID_COLS + m2
    assert len(m1) == 8 and len(m2) == 8 + 16 + 3 + 8
    assert "warmup" not in set(wide["split"])
    exp = clean[clean["split"] != "warmup"]["txn_id"].to_numpy()
    assert (wide["txn_id"].to_numpy() == exp).all()  # same rows, same (ts, txn_id) order


def test_account_join_values_are_by_date_and_node(world):
    clean, txf, acc, pair, gnd = world
    wide = A.build_wide(clean, txf, acc, pair, gnd)
    c = clean.set_index("txn_id")
    a = acc.set_index(["feature_date", "node_id"])
    for r in wide.sample(300, random_state=0).itertuples(index=False):
        row = c.loc[r.txn_id]
        for side, node in (("src", row["src_node"]), ("dst", row["dst_node"])):
            exp = a.loc[(row["date"], node)]
            for f in A_COLS:
                got = getattr(r, f"a_{side}_{f}")
                e = exp[f"a_{f}"]
                assert (np.isnan(got) and np.isnan(e)) or got == pytest.approx(e)


def test_graph_join_sides_and_selection(world):
    clean, txf, acc, pair, gnd = world
    wide = A.build_wide(clean, txf, acc, pair, gnd)
    c = clean.set_index("txn_id")
    assert "g_src_has_sent_before" not in wide.columns  # not in the YAML -> not in the matrix
    assert "g_dst_has_received_before" not in wide.columns
    for r in wide.sample(300, random_state=1).itertuples(index=False):
        row = c.loc[r.txn_id]
        d = (row["date"] - pd.Timestamp("2022-09-01")).days
        s, t = int(row["src_node"]), int(row["dst_node"])
        assert r.g_src_pagerank_log == pytest.approx(s * 0.001 + d * 0.1, abs=1e-4)
        assert r.g_dst_pagerank_log == pytest.approx(t * 0.001 + d * 0.1, abs=1e-4)
        assert r.g_src_n_unique_out_log == s % 7
        assert r.g_dst_n_unique_in_log == t % 5
        assert r.g_dst_has_sent_before == t % 2
        assert r.g_src_has_received_before == (s + 1) % 2
        assert r.g_pair_seen_before == r.txn_id % 2
        assert r.g_pair_prior_count_log == r.txn_id % 9


def test_m1_only_without_graph_tables(world):
    clean, txf, acc, pair, gnd = world
    wide = A.build_wide(clean, txf, acc)
    assert not any(c.startswith("g_") for c in wide.columns)
    assert "a_src_n_out_rate" in wide.columns
    with pytest.raises(ValueError):
        A.build_wide(clean, txf, acc, pair=pair)  # only one graph table


def test_missing_account_key_raises(world):
    clean, txf, acc, pair, gnd = world
    with pytest.raises(ValueError, match="no .* match"):
        A.build_wide(clean, txf, acc.iloc[1:].reset_index(drop=True), pair, gnd)


def test_missing_graph_keys_raise(world):
    clean, txf, acc, pair, gnd = world
    with pytest.raises(ValueError, match="graph_node_day"):
        A.build_wide(clean, txf, acc, pair, gnd.iloc[1:].reset_index(drop=True))
    with pytest.raises(ValueError, match="graph_pair_txn"):
        A.build_wide(clean, txf, acc, pair.iloc[1:].reset_index(drop=True), gnd)


def test_unsorted_clean_raises(world):
    clean, txf, acc, pair, gnd = world
    with pytest.raises(AssertionError, match="sorted"):
        A.build_wide(clean.sample(frac=1.0, random_state=0).reset_index(drop=True), txf, acc, pair, gnd)


def test_categories_come_from_train_and_unseen_raises(world):
    clean, txf, acc, pair, gnd = world
    wide = A.build_wide(clean, txf, acc, pair, gnd)
    assert str(wide["t_cur_paid"].dtype) == "category"
    assert list(wide["t_cur_paid"].cat.categories) == sorted(wide["t_cur_paid"].cat.categories)
    txf2 = txf.copy()
    val_id = clean.loc[clean["split"] == "val", "txn_id"].iloc[0]
    txf2.loc[txf2["txn_id"] == val_id, "t_cur_paid"] = "Bitcoin"
    with pytest.raises(ValueError, match="not in train categories"):
        A.build_wide(clean, txf2, acc, pair, gnd)


def test_check_counts():
    w = pd.DataFrame({"split": ["train", "train", "val"], "is_laundering": [True, False, False]})
    A.check_counts(w, {"train": 2, "val": 1}, {"train": 1, "val": 0})
    with pytest.raises(AssertionError, match="locked numbers"):
        A.check_counts(w, {"train": 3, "val": 1}, {"train": 1, "val": 0})


def test_ks_stat():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert A.ks_stat(x, x.copy()) == 0.0
    assert A.ks_stat(x, x + 10) == pytest.approx(1.0)
    assert 0.0 < A.ks_stat(x, rng.normal(0.3, 1, 5000)) < 0.5


def test_drift_report_shape(world):
    clean, txf, acc, pair, gnd = world
    wide = A.build_wide(clean, txf, acc, pair, gnd)
    rows = A.drift_report(wide, list(wide.columns[3:]), n=500)
    feats = {r["feature"] for r in rows}
    assert "t_payment_format" not in feats and "a_src_n_out_rate" in feats