import json

import numpy as np

from aml.common import paths


RUN_DATE = "20261007"
SEEDS = [42, 43, 44]
VARIANTS = {"full": "full", "noflags": "noshortcut"}  # file name -> thresholds key
SPLIT_KEYS = {"val": "validation", "test": "test", "tail": "tail"}
METRICS = [
    "pr_auc", "roc_auc", "precision", "recall", "f1", "lift",
    "recall_at_budget", "precision_at_budget",
    "tp", "fp", "fn", "tn", "n_alerts",
]

thresholds = json.loads((paths.RESULTS / "thresholds.json").read_text())


def load(model, variant, seed):
    run_id = f"{RUN_DATE}_{model}_{variant}_seed{seed}"
    f = paths.RESULTS / "metrics" / f"{run_id}.json"
    if not f.exists():
        raise SystemExit(f"STOP: missing {f}")
    d = json.loads(f.read_text())
    tkey = f"{model}_{VARIANTS[variant]}_seed{seed}"
    if abs(d["threshold"]["value"] - thresholds[tkey]["threshold"]) > 1e-12:
        raise SystemExit(f"STOP: threshold mismatch for {run_id}")
    return d


def stats(vals):
    a = np.array(vals, dtype=float)
    return {"mean": round(float(a.mean()), 5), "std": round(float(a.std(ddof=1)), 5)}


out_variants = {}
for variant in VARIANTS:
    docs = {
        m: {s: load(m, variant, s) for s in SEEDS} for m in ["m1", "m2"]
    }
    out_splits = {}
    for split, mk in SPLIT_KEYS.items():
        block = {}
        for seed in SEEDS:
            a, b = docs["m1"][seed][mk], docs["m2"][seed][mk]
            if (a["n"], a["n_pos"]) != (b["n"], b["n_pos"]):
                raise SystemExit(f"STOP: n/n_pos differ M1 vs M2 ({variant}, {split}, {seed})")
        for metric in METRICS:
            try:
                v1 = {s: float(docs["m1"][s][mk][metric]) for s in SEEDS}
                v2 = {s: float(docs["m2"][s][mk][metric]) for s in SEEDS}
            except KeyError as e:
                raise SystemExit(f"STOP: key {e} missing in {variant}/{split}")
            dl = {s: v2[s] - v1[s] for s in SEEDS}
            block[metric] = {
                "m1": {**stats(list(v1.values())), "per_seed": {str(s): round(v1[s], 5) for s in SEEDS}},
                "m2": {**stats(list(v2.values())), "per_seed": {str(s): round(v2[s], 5) for s in SEEDS}},
                "delta": {**stats(list(dl.values())), "per_seed": {str(s): round(dl[s], 5) for s in SEEDS}},
            }
        a = docs["m1"][SEEDS[0]][mk]
        out_splits[split] = {"n": a["n"], "n_pos": a["n_pos"], "metrics": block}
    out_variants[variant] = out_splits

doc = {
    "note": (
        "delta = M2 minus M1, computed per seed and then averaged. "
        "std uses ddof=1 over 3 seeds, so treat it as rough. Thresholds are "
        "frozen on validation. Tail has 59% fraud, so its PR-AUC and lift are "
        "mostly a base-rate effect and are never pooled with val or test. "
        "noflags drops the payment-format shortcut features."
    ),
    "seeds": SEEDS,
    "variants": out_variants,
}

out = paths.RESULTS / "comparison.json"
out.write_text(json.dumps(doc, indent=2))
print(f"wrote {out}")

for variant, splits in out_variants.items():
    for split in ["val", "test"]:
        m = splits[split]["metrics"]
        print(
            f"{variant:8s} {split:4s} "
            f"pr_auc M1 {m['pr_auc']['m1']['mean']:.3f} -> M2 {m['pr_auc']['m2']['mean']:.3f} "
            f"(delta {m['pr_auc']['delta']['mean']:+.3f}) | "
            f"f1 {m['f1']['m1']['mean']:.3f} -> {m['f1']['m2']['mean']:.3f}"
        )