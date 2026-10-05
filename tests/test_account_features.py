import numpy as np
import pandas as pd
import pytest

from aml.features.account import FEATURES, build_account_node_day


def make_df(seed=0, n=4000, n_nodes=40, n_days=8):
    rng = np.random.default_rng(seed)
    day = rng.integers(0, n_days, n)
    ts = pd.Timestamp("2022-09-01") + pd.to_timedelta(day, unit="D") + pd.to_timedelta(rng.integers(0, 1440, n), unit="m")
    src = rng.integers(0, n_nodes, n)
    dst = rng.integers(0, n_nodes, n)
    sl = rng.random(n) < 0.12
    dst = np.where(sl, src, dst)
    df = pd.DataFrame({
        "ts": ts,
        "src_node": src.astype(np.int32),
        "dst_node": dst.astype(np.int32),
        "payment_format": rng.choice(["ACH", "Cheque", "Wire", "Credit Card"], n),
        "is_self_loop": src == dst,
        "split": np.where(day == 0, "warmup", "train"),
        "t_amount_z_in_currency": rng.normal(0, 1, n),
    })
    df["day"] = day
    return df


def brute(df, D, node):
    h = df[(~df["is_self_loop"]) & (df["day"] < D)]
    out = h[h["src_node"] == node]
    inn = h[h["dst_node"] == node]
    both = pd.concat([out, inn])
    z = "t_amount_z_in_currency"
    nan = np.nan
    return {
        "a_n_out_rate": np.log1p(len(out) / D),
        "a_n_in_rate": np.log1p(len(inn) / D),
        "a_mean_z_out": out[z].mean() if len(out) else nan,
        "a_mean_z_in": inn[z].mean() if len(inn) else nan,
        "a_max_z_out": out[z].max() if len(out) else nan,
        "a_days_active_frac": both["day"].nunique() / D,
        "a_days_since_last": (D - both["day"].max()) if len(both) else nan,
        "a_fmt_diversity": both["payment_format"].nunique(),
    }


def run(df):
    return build_account_node_day(df.drop(columns=["day"]))


def test_matches_brute_force_every_row():
    df = make_df()
    res = run(df)
    base = pd.Timestamp("2022-09-01")
    for r in res.itertuples(index=False):
        D = (r.feature_date - base).days
        exp = brute(df, D, r.node_id)
        for f in FEATURES:
            got = getattr(r, f)
            if np.isnan(exp[f]):
                assert np.isnan(got), (D, r.node_id, f)
            else:
                assert got == pytest.approx(exp[f], rel=1e-4, abs=1e-5), (D, r.node_id, f)


def test_same_day_and_future_rows_do_not_change_features():
    df = make_df(seed=1)
    res = run(df)
    base = pd.Timestamp("2022-09-01")
    D = 4
    df2 = df.copy()
    m = df2["day"] >= D
    rng = np.random.default_rng(99)
    df2.loc[m, "t_amount_z_in_currency"] = rng.normal(50, 5, m.sum())
    df2.loc[m, "payment_format"] = "Bitcoin"
    res2 = run(df2)
    a = res[res["feature_date"] == base + pd.Timedelta(days=D)].set_index("node_id")[FEATURES]
    b = res2[res2["feature_date"] == base + pd.Timedelta(days=D)].set_index("node_id")[FEATURES]
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_rows_cover_all_prediction_nodes_and_skip_warmup():
    df = make_df(seed=2)
    res = run(df)
    base = pd.Timestamp("2022-09-01")
    assert (res["feature_date"] > base).all()
    p = df[df["split"] != "warmup"]
    expected = set(zip(p["day"], p["src_node"])) | set(zip(p["day"], p["dst_node"]))
    got = set(zip(((res["feature_date"] - base).dt.days), res["node_id"]))
    assert expected == got
    assert not res.duplicated(["feature_date", "node_id"]).any()


def test_node_with_no_history_has_zero_counts_and_nan_means():
    df = make_df(seed=3)
    new = pd.DataFrame({
        "ts": [pd.Timestamp("2022-09-04 10:00")] * 2,
        "src_node": np.array([999, 999], dtype=np.int32),
        "dst_node": np.array([1, 999], dtype=np.int32),
        "payment_format": ["ACH", "ACH"],
        "is_self_loop": [False, True],
        "split": ["train", "train"],
        "t_amount_z_in_currency": [0.5, 0.5],
        "day": [3, 3],
    })
    res = run(pd.concat([df, new], ignore_index=True))
    r = res[res["node_id"] == 999].iloc[0]
    assert r["a_n_out_rate"] == 0 and r["a_n_in_rate"] == 0
    assert r["a_days_active_frac"] == 0 and r["a_fmt_diversity"] == 0
    assert np.isnan(r["a_mean_z_out"]) and np.isnan(r["a_max_z_out"]) and np.isnan(r["a_days_since_last"])


def test_self_loops_excluded_from_history():
    df = make_df(seed=4)
    res = run(df)
    df2 = df.copy()
    sl = df2["is_self_loop"]
    df2.loc[sl, "t_amount_z_in_currency"] = 1e6
    df2.loc[sl, "payment_format"] = "Reinvestment"
    res2 = run(df2)
    pd.testing.assert_frame_equal(res, res2)