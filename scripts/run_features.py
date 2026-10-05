"""P1. transactions_clean -> txn_features.parquet (more stages are added in later steps)."""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aml.common import paths
from aml.features import transaction


def main():
    t0 = time.time()
    clean = pd.read_parquet(paths.PROCESSED / "transactions_clean.parquet", columns=transaction.COLS)
    feats, stats = transaction.build_txn_features(clean)
    feats.to_parquet(paths.PROCESSED / "txn_features.parquet", index=False)
    transaction.save_stats(stats, paths.PROCESSED / "currency_stats.json")
    print(f"txn_features: {feats.shape}  ({time.time()-t0:.0f}s)")
    print(feats.drop(columns="txn_id").describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    print("NaN counts:", feats.isna().sum()[lambda s: s > 0].to_dict() or "none")


if __name__ == "__main__":
    main()