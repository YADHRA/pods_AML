"""Step 5b - XGBoost training harness (used for Model 1 now, Model 2 later with the same code).

Protocol (locked):
  * scale_pos_weight comes from TRAIN only (neg/pos), tried as 1, sqrt(neg/pos) or neg/pos.
  * Random search: the SAME N configs for every model/variant (same budget), seed 42 for fitting.
  * Early stopping on VALIDATION aucpr; the best config is picked by validation PR-AUC.
  * The best config is then refit with every seed (42, 43, 44).
  * For each fit the threshold is chosen on validation (max F1) and frozen in results/thresholds.json.
  * TEST AND TAIL ROWS ARE NEVER USED OR SCORED HERE. Test is evaluated once, later, in its own script.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from aml.evaluation.metrics import average_precision, evaluate
from aml.evaluation.threshold import best_f1_threshold, freeze_threshold

# ---- search space (from the locked methodology) ----
MAX_DEPTH = [4, 5, 6, 7]
LEARNING_RATE = [0.05, 0.075, 0.1]
MIN_CHILD_WEIGHT = [1, 2, 3, 5, 7, 10]
SPW_MODES = ["none", "sqrt", "full"]  # scale_pos_weight = 1, sqrt(neg/pos), neg/pos (all from TRAIN)

# Fixed on purpose: with subsample = colsample = 1.0 XGBoost has no randomness, so seeds 42/43/44
# would give identical models. 0.8 makes the three seeds genuinely different.
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

MAX_TREES = 1000
EARLY_STOPPING_ROUNDS = 50
SEARCH_SEED = 42


def feature_columns(fs: dict, model: str, variant: str):
    """Columns for ('m1'|'m2', 'full'|'noshortcut') from configs/feature_sets.yaml."""
    m1 = list(fs["m1_transaction"])
    if model == "m1":
        cols = m1
    elif model == "m2":
        cols = m1 + list(fs["group_account"]) + list(fs["group_pair"]) + list(fs["group_network"])
    else:
        raise ValueError("model must be 'm1' or 'm2'")
    if variant == "full":
        return cols
    if variant == "noshortcut":
        drop = set(fs["shortcut_features"])
        return [c for c in cols if c not in drop]
    raise ValueError("variant must be 'full' or 'noshortcut'")


def sample_configs(n_trials: int, search_seed: int = SEARCH_SEED):
    """Deterministic random sample (no repeats) from the grid. Same call -> same list for every model."""
    grid = [dict(max_depth=d, learning_rate=lr, min_child_weight=m, spw_mode=s)
            for d, lr, m, s in itertools.product(MAX_DEPTH, LEARNING_RATE, MIN_CHILD_WEIGHT, SPW_MODES)]
    if n_trials > len(grid):
        raise ValueError(f"n_trials={n_trials} is more than the {len(grid)} configs in the grid")
    idx = np.random.default_rng(search_seed).choice(len(grid), size=n_trials, replace=False)
    return [grid[i] for i in idx]


def spw_options(y_train) -> dict:
    y = np.asarray(y_train).astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("no positives in train")
    ratio = (len(y) - n_pos) / n_pos
    return {"none": 1.0, "sqrt": math.sqrt(ratio), "full": ratio}


def fit_one(X_tr, y_tr, X_va, y_va, cfg: dict, spw: dict, seed: int,
            max_trees=MAX_TREES, early_stopping_rounds=EARLY_STOPPING_ROUNDS, n_jobs=None):
    model = XGBClassifier(
        n_estimators=max_trees,
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"],
        min_child_weight=cfg["min_child_weight"],
        scale_pos_weight=spw[cfg["spw_mode"]],
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=early_stopping_rounds,
        enable_categorical=True,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=0,
    )
    model.fit(X_tr, np.asarray(y_tr).astype(int), eval_set=[(X_va, np.asarray(y_va).astype(int))], verbose=False)
    return model


def score(model, X) -> np.ndarray:
    """Probability-like ranking score at the early-stopping iteration."""
    return model.predict_proba(X, iteration_range=(0, int(model.best_iteration) + 1))[:, 1]


def run_model(df: pd.DataFrame, cols, name: str, seeds, n_trials: int,
              results_dir, models_dir, thresholds_path,
              refreeze=False, max_trees=MAX_TREES, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
              n_jobs=None, log=print) -> dict:
    """Search + refit + freeze thresholds for one model/variant. Uses ONLY train and val rows."""
    results_dir, models_dir = Path(results_dir), Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    X_tr, y_tr = tr[cols], tr["is_laundering"].astype(int).to_numpy()
    X_va, y_va = va[cols], va["is_laundering"].astype(int).to_numpy()
    spw = spw_options(y_tr)
    log(f"[{name}] {len(cols)} features | train {len(tr):,} rows / {int(y_tr.sum()):,} pos | "
        f"val {len(va):,} rows / {int(y_va.sum()):,} pos | scale_pos_weight options "
        f"{ {k: round(v, 1) for k, v in spw.items()} }")

    kw = dict(max_trees=max_trees, early_stopping_rounds=early_stopping_rounds, n_jobs=n_jobs)
    trials, best_model, best_ap = [], None, -1.0
    for i, cfg in enumerate(sample_configs(n_trials), 1):
        t = time.time()
        m = fit_one(X_tr, y_tr, X_va, y_va, cfg, spw, seeds[0], **kw)
        ap = average_precision(y_va, score(m, X_va))
        trials.append({**cfg, "scale_pos_weight": spw[cfg["spw_mode"]], "best_iteration": int(m.best_iteration),
                       "val_pr_auc": ap, "seconds": round(time.time() - t, 1)})
        log(f"[{name}] trial {i}/{n_trials}: depth={cfg['max_depth']} lr={cfg['learning_rate']} "
            f"mcw={cfg['min_child_weight']} spw={cfg['spw_mode']} -> val PR-AUC {ap:.4f} "
            f"({int(m.best_iteration) + 1} trees, {trials[-1]['seconds']}s)")
        if ap > best_ap:
            best_ap, best_model, best_i = ap, m, i - 1
    best_cfg = {k: trials[best_i][k] for k in ("max_depth", "learning_rate", "min_child_weight", "spw_mode")}
    log(f"[{name}] best config: {best_cfg} (val PR-AUC {best_ap:.4f})")

    val_metrics = {}
    for seed in seeds:
        if seed == seeds[0]:
            m = best_model  # already fitted with this seed during the search
        else:
            m = fit_one(X_tr, y_tr, X_va, y_va, best_cfg, spw, seed, **kw)
        s_va = score(m, X_va)
        thr = best_f1_threshold(y_va, s_va)
        freeze_threshold(thresholds_path, f"{name}_seed{seed}", thr, overwrite=refreeze)
        val_metrics[str(seed)] = evaluate(y_va, s_va, thr["threshold"])
        m.save_model(str(models_dir / f"{name}_seed{seed}.json"))
        e = val_metrics[str(seed)]
        log(f"[{name}] seed {seed}: val PR-AUC {e['pr_auc']:.4f} (prevalence {e['prevalence']:.4f}, lift {e['lift']:.0f}x) | "
            f"threshold {thr['threshold']:.4f} -> precision {e['precision']:.3f}, recall {e['recall']:.3f}, F1 {e['f1']:.3f}")

    aps = [v["pr_auc"] for v in val_metrics.values()]
    out = {
        "name": name, "features": list(cols), "seeds": list(seeds), "n_trials": n_trials,
        "scale_pos_weight_options": spw, "fixed": {"subsample": SUBSAMPLE, "colsample_bytree": COLSAMPLE_BYTREE,
                                                    "max_trees": max_trees, "early_stopping_rounds": early_stopping_rounds},
        "search": trials, "best_config": best_cfg, "val": val_metrics,
        "val_pr_auc_mean": float(np.mean(aps)), "val_pr_auc_std": float(np.std(aps)),
    }
    (results_dir / f"{name}_val.json").write_text(json.dumps(out, indent=2))
    return out