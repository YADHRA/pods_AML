"""Evaluation metrics (numpy + pandas only - no sklearn needed).

Scores are RANKING scores, not calibrated probabilities. A "positive" prediction means
score >= threshold. PR-AUC is "average precision" (step-wise area under the PR curve),
always reported next to prevalence and lift (= PR-AUC / prevalence).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _prep(y, s):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype="float64")
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("y and scores must be 1-D arrays of the same length")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    return y, s


def curve_points(y, s):
    """Cumulative counts at each DISTINCT score, from highest score to lowest.
    Returns (thresholds, tp, fp): predicting score >= thresholds[i] gives tp[i] true and fp[i] false alerts."""
    y, s = _prep(y, s)
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    last_of_each_score = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    return s_sorted[last_of_each_score], tp[last_of_each_score], fp[last_of_each_score]


def average_precision(y, s) -> float:
    """PR-AUC: sum over thresholds of (recall step) x precision."""
    y, s = _prep(y, s)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    _, tp, fp = curve_points(y, s)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def roc_auc(y, s) -> float:
    """Probability a random positive scores higher than a random negative (ties count half)."""
    y, s = _prep(y, s)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confusion_at(y, s, threshold):
    y, s = _prep(y, s)
    pred = s >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    return tn, fp, fn, tp


def recall_at_budget(y, s, frac=0.005):
    """Alert the top `frac` of rows by score. Returns (recall, precision, k).
    Ties at the cut-off are broken by input order (stable sort)."""
    y, s = _prep(y, s)
    n = len(y)
    k = max(1, math.ceil(frac * n))
    order = np.argsort(-s, kind="mergesort")
    hits = int(y[order[:k]].sum())
    n_pos = int(y.sum())
    return (hits / n_pos if n_pos else float("nan")), hits / k, k


def evaluate(y, s, threshold, budget_frac=0.005) -> dict:
    """All metrics for one set of rows at a frozen threshold. JSON-safe."""
    y, s = _prep(y, s)
    n, n_pos = len(y), int(y.sum())
    prevalence = n_pos / n
    pr_auc = average_precision(y, s)
    tn, fp, fn, tp = confusion_at(y, s, threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    r_b, p_b, k = recall_at_budget(y, s, budget_frac)
    return {
        "n": n, "n_pos": n_pos, "prevalence": prevalence,
        "pr_auc": pr_auc,
        "lift": pr_auc / prevalence if prevalence else float("nan"),
        "roc_auc": roc_auc(y, s),
        "threshold": float(threshold),
        "precision": precision, "recall": recall, "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "n_alerts": tp + fp,
        "accuracy": (tp + tn) / n,  # supplementary only - misleading at 0.1% prevalence
        "budget_frac": budget_frac, "budget_k": k,
        "recall_at_budget": r_b, "precision_at_budget": p_b,
    }


def pr_curve_points(y, s, max_points=400) -> dict:
    """Down-sampled PR curve for the dashboard (always keeps the first and last point)."""
    y, s = _prep(y, s)
    n_pos = int(y.sum())
    thr, tp, fp = curve_points(y, s)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    idx = np.unique(np.linspace(0, len(thr) - 1, min(max_points, len(thr))).round().astype(int))
    return {"threshold": thr[idx].tolist(), "precision": precision[idx].tolist(), "recall": recall[idx].tolist()}