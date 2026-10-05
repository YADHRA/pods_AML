import json

import numpy as np
import pandas as pd
import pytest

from aml.evaluation.threshold import load_threshold
from aml.features import assemble as A
from aml.models import xgb as X

FS = A.load_feature_sets()


def make_matrix(seed=0, n_train=6000, n_val=3000, n_test=1500):
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    fmt = rng.choice(["ACH", "Cheque", "Wire", "Cash"], n)
    cur = rng.choice(["USD", "EUR", "Yen"], n)
    amt = rng.normal(3, 1, n)
    z = rng.normal(0, 1, n)
    logit = -5.5 + 3.0 * (fmt == "ACH") + 0.5 * z
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "txn_id": np.arange(n, dtype=np.int32), "split": split, "is_laundering": y,
        "t_log_amount_paid": amt, "t_amount_z_in_currency": z,
        "t_payment_format": pd.Categorical(fmt, categories=sorted(set(fmt))),
        "t_cur_paid": pd.Categorical(cur, categories=sorted(set(cur))),
        "t_is_fx": rng.integers(0, 2, n).astype(float), "t_is_self_loop": rng.integers(0, 2, n).astype(float),
        "t_same_bank": rng.integers(0, 2, n).astype(float), "t_hour": rng.integers(0, 24, n).astype(float),
    })
    return df


FAST = dict(max_trees=40, early_stopping_rounds=10, n_jobs=2)


def test_feature_columns():
    full = X.feature_columns(FS, "m1", "full")
    ns = X.feature_columns(FS, "m1", "noshortcut")
    assert len(full) == 8
    assert ns == ["t_log_amount_paid", "t_amount_z_in_currency", "t_same_bank", "t_hour"]
    m2 = X.feature_columns(FS, "m2", "full")
    assert len(m2) == 35 and len(X.feature_columns(FS, "m2", "noshortcut")) == 35 - 6
    with pytest.raises(ValueError):
        X.feature_columns(FS, "m3", "full")


def test_sample_configs_deterministic_unique_same_for_all_models():
    a, b = X.sample_configs(12), X.sample_configs(12)
    assert a == b
    assert len({tuple(sorted(c.items())) for c in a}) == 12
    assert X.sample_configs(5) != X.sample_configs(5, search_seed=7)
    with pytest.raises(ValueError):
        X.sample_configs(10_000)
    for c in a:
        assert c["max_depth"] in X.MAX_DEPTH and c["learning_rate"] in X.LEARNING_RATE
        assert c["min_child_weight"] in X.MIN_CHILD_WEIGHT and c["spw_mode"] in X.SPW_MODES


def test_scale_pos_weight_uses_train_only_ratio():
    y = np.array([1] * 2 + [0] * 198)
    o = X.spw_options(y)
    assert o["none"] == 1.0 and o["full"] == pytest.approx(99.0) and o["sqrt"] == pytest.approx(99 ** 0.5)
    with pytest.raises(ValueError):
        X.spw_options(np.zeros(10))


def test_same_seed_reproducible_different_seed_differs():
    df = make_matrix()
    cols = X.feature_columns(FS, "m1", "full")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    spw = X.spw_options(tr.is_laundering)
    cfg = dict(max_depth=5, learning_rate=0.1, min_child_weight=2, spw_mode="sqrt")
    kw = dict(max_trees=30, early_stopping_rounds=10, n_jobs=2)
    m1 = X.fit_one(tr[cols], tr.is_laundering, va[cols], va.is_laundering, cfg, spw, 42, **kw)
    m1b = X.fit_one(tr[cols], tr.is_laundering, va[cols], va.is_laundering, cfg, spw, 42, **kw)
    m2 = X.fit_one(tr[cols], tr.is_laundering, va[cols], va.is_laundering, cfg, spw, 43, **kw)
    s1, s1b, s2 = X.score(m1, va[cols]), X.score(m1b, va[cols]), X.score(m2, va[cols])
    assert np.array_equal(s1, s1b)          # same seed -> identical scores
    assert not np.allclose(s1, s2)          # different seed -> genuinely different model


def test_run_model_end_to_end_uses_only_train_and_val(tmp_path):
    df = make_matrix()
    cols = X.feature_columns(FS, "m1", "full")
    # poison the test rows: if they were ever used, results would change
    poisoned = df.copy()
    poisoned.loc[poisoned.split == "test", cols[:2]] = np.nan
    thr_path = tmp_path / "thresholds.json"
    out = X.run_model(poisoned, cols, "m1_full", [42, 43], n_trials=3, results_dir=tmp_path / "res",
                      models_dir=tmp_path / "models", thresholds_path=thr_path, log=lambda *_: None, **FAST)
    clean = X.run_model(df, cols, "m1_full", [42, 43], n_trials=3, results_dir=tmp_path / "res2",
                        models_dir=tmp_path / "models2", thresholds_path=tmp_path / "t2.json", log=lambda *_: None, **FAST)
    assert out["val"]["42"]["pr_auc"] == pytest.approx(clean["val"]["42"]["pr_auc"])  # test rows have no influence
    assert set(out["val"]) == {"42", "43"}
    assert len(out["search"]) == 3 and out["best_config"]["max_depth"] in X.MAX_DEPTH
    # thresholds frozen per seed, model files and result json written
    assert load_threshold(thr_path, "m1_full_seed42") > 0 and load_threshold(thr_path, "m1_full_seed43") > 0
    assert (tmp_path / "models" / "m1_full_seed42.json").exists() and (tmp_path / "res" / "m1_full_val.json").exists()
    # model is better than chance on val (the planted ACH signal)
    assert out["val"]["42"]["pr_auc"] > 2 * out["val"]["42"]["prevalence"]


def test_rerun_refuses_to_overwrite_frozen_thresholds_unless_refreeze(tmp_path):
    df = make_matrix(seed=1)
    cols = X.feature_columns(FS, "m1", "noshortcut")
    args = dict(df=df, cols=cols, name="m1_noshortcut", seeds=[42], n_trials=2, results_dir=tmp_path / "r",
                models_dir=tmp_path / "m", thresholds_path=tmp_path / "t.json", log=lambda *_: None, **FAST)
    X.run_model(**args)
    with pytest.raises(FileExistsError):
        X.run_model(**args)
    X.run_model(refreeze=True, **args)