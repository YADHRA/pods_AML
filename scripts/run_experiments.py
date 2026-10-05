import argparse
import time

import pandas as pd

from aml.common import paths
from aml.features import assemble as A
from aml.models import xgb as X

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["m1", "m2"], default="m1")
ap.add_argument("--variant", choices=["full", "noshortcut", "both"], default="both")
ap.add_argument("--n-trials", type=int, default=12)
ap.add_argument("--seeds", default="42,43,44")
ap.add_argument("--refreeze", action="store_true",
                help="allow overwriting frozen thresholds (only before test has been evaluated!)")
args = ap.parse_args()

seeds = [int(s) for s in args.seeds.split(",")]
fs = A.load_feature_sets()
matrix = paths.PROCESSED / ("matrix_model1.parquet" if args.model == "m1" else "matrix_model2.parquet")
if not matrix.exists():
    raise SystemExit(f"{matrix} not found - run scripts/run_assemble.py first")

t0 = time.time()
# Only the rows we are allowed to look at: train + val. Test and tail are dropped right after loading.
df = pd.read_parquet(matrix)
df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
print(f"loaded {matrix.name}: {len(df):,} train+val rows ({time.time() - t0:.0f}s)")

variants = ["full", "noshortcut"] if args.variant == "both" else [args.variant]
for v in variants:
    cols = X.feature_columns(fs, args.model, v)
    X.run_model(df, cols, f"{args.model}_{v}", seeds, args.n_trials,
                results_dir=paths.RESULTS / "models", models_dir=paths.ROOT / "models",
                thresholds_path=paths.RESULTS / "thresholds.json", refreeze=args.refreeze)
print(f"\ndone in {(time.time() - t0) / 60:.1f} min. Thresholds frozen in results/thresholds.json; test NOT touched.")