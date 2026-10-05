"""Threshold selection on VALIDATION and freezing.

Rule (confirmed): maximize F1 on validation. The result is written to results/thresholds.json
BEFORE the test set is touched. A frozen key cannot be overwritten by accident.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aml.evaluation.metrics import curve_points

RULE = "max_f1_on_validation"


def best_f1_threshold(y, s) -> dict:
    """Threshold (predict positive if score >= threshold) with the highest F1.
    If several thresholds tie on F1, the highest one wins (fewest alerts)."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("no positives in the validation rows - cannot choose a threshold")
    thr, tp, fp = curve_points(y, s)
    fn = n_pos - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    best = f1.max()
    i = int(np.nonzero(f1 >= best - 1e-12)[0][0])  # thresholds run high -> low, so first = highest
    return {
        "threshold": float(thr[i]),
        "val_f1": float(f1[i]),
        "val_precision": float(tp[i] / (tp[i] + fp[i])),
        "val_recall": float(tp[i] / n_pos),
        "val_n_alerts": int(tp[i] + fp[i]),
        "rule": RULE,
    }


def freeze_threshold(path, key: str, info: dict, overwrite: bool = False) -> None:
    """Write info under `key` in a JSON file. Refuses to replace an existing (frozen) key."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    if key in data and not overwrite:
        raise FileExistsError(f"threshold '{key}' is already frozen in {path} (pass overwrite=True to replace)")
    data[key] = info
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def load_threshold(path, key: str) -> float:
    data = json.loads(Path(path).read_text())
    if key not in data:
        raise KeyError(f"no frozen threshold for '{key}' in {path} - freeze it on validation first")
    return float(data[key]["threshold"])