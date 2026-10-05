import json

import numpy as np
import pytest

from aml.evaluation import metrics as M
from aml.evaluation import threshold as T


def make_scores(seed=0, n=5000, prev=0.05, decimals=2):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < prev
    s = np.round(np.clip(rng.normal(0.3, 0.2, n) + 0.35 * y, 0, 1), decimals)  # rounding creates ties
    return y, s


def test_hand_computed_examples():
    y, s = [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]
    assert M.average_precision(y, s) == pytest.approx(0.5 + (2 / 3) * 0.5)
    assert M.roc_auc(y, s) == pytest.approx(0.75)
    # all scores tied -> a single threshold: precision = prevalence
    assert M.average_precision([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert M.roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)


def test_perfect_and_no_positive_cases():
    y = np.array([1, 1, 0, 0, 0])
    assert M.average_precision(y, [0.9, 0.8, 0.3, 0.2, 0.1]) == pytest.approx(1.0)
    assert M.roc_auc(y, [0.9, 0.8, 0.3, 0.2, 0.1]) == pytest.approx(1.0)
    assert np.isnan(M.average_precision([0, 0], [0.1, 0.2]))
    with pytest.raises(ValueError):
        M.average_precision([1, 0], [np.nan, 0.2])


def test_matches_sklearn_with_ties():
    sk = pytest.importorskip("sklearn.metrics")
    for seed in range(3):
        y, s = make_scores(seed)
        assert M.average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-9)
        assert M.roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-9)
        thr = 0.45
        p, r, f, _ = sk.precision_recall_fscore_support(y, s >= thr, average="binary", zero_division=0)
        e = M.evaluate(y, s, thr)
        assert (e["precision"], e["recall"], e["f1"]) == pytest.approx((p, r, f), abs=1e-9)
        tn, fp, fn, tp = sk.confusion_matrix(y, s >= thr).ravel()
        assert (e["tn"], e["fp"], e["fn"], e["tp"]) == (tn, fp, fn, tp)


def test_recall_at_budget():
    y = np.zeros(1000, dtype=bool)
    y[[0, 1, 500]] = True
    s = np.linspace(1, 0, 1000)  # row 0 has the highest score
    r, p, k = M.recall_at_budget(y, s, 0.005)
    assert k == 5 and r == pytest.approx(2 / 3) and p == pytest.approx(2 / 5)


def test_evaluate_keys_and_lift():
    y, s = make_scores(1)
    e = M.evaluate(y, s, 0.5)
    for k in ["pr_auc", "prevalence", "lift", "roc_auc", "precision", "recall", "f1", "tn", "fp", "fn", "tp",
              "accuracy", "recall_at_budget", "precision_at_budget", "n_alerts"]:
        assert k in e
    assert e["lift"] == pytest.approx(e["pr_auc"] / e["prevalence"])
    assert e["tp"] + e["fn"] == e["n_pos"] and e["n_alerts"] == e["tp"] + e["fp"]
    json.dumps(e)  # must be JSON-safe


def test_pr_curve_points_downsample():
    y, s = make_scores(2, n=20000, decimals=6)
    c = M.pr_curve_points(y, s, max_points=100)
    assert len(c["recall"]) <= 100
    assert c["recall"][-1] == pytest.approx(1.0)
    assert all(a <= b + 1e-12 for a, b in zip(c["recall"], c["recall"][1:]))


def test_best_f1_matches_brute_force():
    y, s = make_scores(3)
    got = T.best_f1_threshold(y, s)
    best = (-1, None)
    for t in np.unique(s)[::-1]:  # high -> low, keep the first (highest) on ties
        e = M.evaluate(y, s, t)
        if e["f1"] > best[0] + 1e-12:
            best = (e["f1"], t)
    assert got["val_f1"] == pytest.approx(best[0]) and got["threshold"] == pytest.approx(best[1])
    assert got["rule"] == "max_f1_on_validation"
    assert M.evaluate(y, s, got["threshold"])["f1"] == pytest.approx(got["val_f1"])


def test_best_f1_needs_positives():
    with pytest.raises(ValueError):
        T.best_f1_threshold([0, 0, 0], [0.1, 0.2, 0.3])


def test_freeze_and_load(tmp_path):
    p = tmp_path / "thresholds.json"
    info = {"threshold": 0.42, "val_f1": 0.1, "rule": T.RULE}
    T.freeze_threshold(p, "m1_full_seed42", info)
    assert T.load_threshold(p, "m1_full_seed42") == pytest.approx(0.42)
    with pytest.raises(FileExistsError):
        T.freeze_threshold(p, "m1_full_seed42", {"threshold": 0.9})
    T.freeze_threshold(p, "m1_full_seed42", {"threshold": 0.9}, overwrite=True)
    assert T.load_threshold(p, "m1_full_seed42") == pytest.approx(0.9)
    T.freeze_threshold(p, "m1_full_seed43", info)  # other keys are untouched
    assert T.load_threshold(p, "m1_full_seed42") == pytest.approx(0.9)
    with pytest.raises(KeyError):
        T.load_threshold(p, "m2_full_seed42")