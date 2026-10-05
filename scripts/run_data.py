"""P1. RAW CSV -> audit -> txn_id -> node_map -> split -> data/processed/*.parquet + results/eda/*.json
Usage: python scripts/run_data.py [--nrows N] [--skip-sha]"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aml.common.config import load_config
from aml.common import paths
from aml.data import load, validate, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None, help="debug: only read first N rows (skips count checks)")
    ap.add_argument("--skip-sha", action="store_true")
    a = ap.parse_args()
    cfg, t0 = load_config(), time.time()
    csv = paths.RAW_CSV
    print(f"[1/5] reading {csv}")
    if not a.skip_sha:
        print("      sha256:", load.sha256_of(csv), " (paste into configs/config.yaml -> dataset.sha256)")
    raw = load.read_raw(csv, nrows=a.nrows)
    print(f"[2/5] audit ({len(raw):,} rows, {time.time()-t0:.0f}s)")
    rep = validate.audit(raw)
    problems = validate.check_expectations(rep, cfg)
    if problems and a.nrows is None:
        raise SystemExit("DATASET CHECK FAILED:\n  " + "\n  ".join(problems))
    validate.write_report(rep, paths.RESULTS / "eda" / "data_quality.json", paths.ROOT / "docs" / "data_quality.md")
    print("[3/5] build clean table + node map + split")
    clean, node_map = split.build_clean(raw, cfg)
    del raw
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    print("[4/5] writing parquet")
    clean.to_parquet(paths.PROCESSED / "transactions_clean.parquet", index=False)
    node_map.to_parquet(paths.PROCESSED / "node_map.parquet", index=False)
    print("[5/5] writing EDA summaries")
    eda = paths.RESULTS / "eda"
    (eda / "splits.json").write_text(json.dumps(split.split_summary(clean), indent=2))
    (eda / "daily.json").write_text(json.dumps(split.daily_summary(clean), indent=2))
    print(f"\nnodes: {len(node_map):,}   rows: {len(clean):,}   done in {time.time()-t0:.0f}s")
    for r in split.split_summary(clean):
        print(f"  {r['split']:7s} rows={r['n_rows']:>9,} pos={r['n_pos']:>5,} rate={r['pos_rate']*100:6.3f}%  {r['date_min'][:10]}..{r['date_max'][:10]}")


if __name__ == "__main__":
    main()
