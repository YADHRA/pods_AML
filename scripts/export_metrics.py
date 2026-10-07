import json

from aml.common import paths

RUN_DATE = "20261007"  # fixed so re-running gives the same run_ids

src = json.loads((paths.RESULTS / "test_results.json").read_text())
thr = json.loads((paths.RESULTS / "thresholds.json").read_text())

out_dir = paths.RESULTS / "metrics"
out_dir.mkdir(parents=True, exist_ok=True)

for key, r in src.items():
    variant = "noflags" if r["variant"] == "noshortcut" else r["variant"]
    run_id = f"{RUN_DATE}_{r['model']}_{variant}_seed{r['seed']}"
    t = thr[key]
    doc = {
        "run_id": run_id,
        "model": r["model"],
        "variant": variant,
        "seed": r["seed"],
        "n_features": len(r["features"]),
        "features": r["features"],
        "threshold": {"value": t["threshold"], "rule": t["rule"], "frozen_on": "validation"},
        "validation": r["validation_check"],
        "test": r["test"],
        "tail": r.get("tail"),
    }
    (out_dir / f"{run_id}.json").write_text(json.dumps(doc, indent=2))
    print(f"{run_id}: test PR-AUC {r['test']['pr_auc']:.4f} | F1 {r['test']['f1']:.3f}")

print(f"\nwrote {len(src)} files to {out_dir}")