import os
import time
from pathlib import Path

import pandas as pd

from aml.features.account import FEATURES, build_account_node_day

DATA = Path(os.environ.get("AML_DATA_DIR", "data")) / "processed"

t0 = time.time()
clean = pd.read_parquet(
    DATA / "transactions_clean.parquet",
    columns=["txn_id", "ts", "src_node", "dst_node", "payment_format", "is_self_loop", "split"],
)
tf = pd.read_parquet(DATA / "txn_features.parquet", columns=["txn_id", "t_amount_z_in_currency"])
df = clean.merge(tf, on="txn_id", how="left", validate="one_to_one")
assert df["t_amount_z_in_currency"].notna().all(), "z-score missing after join"
df["payment_format"] = df["payment_format"].astype(str)

res = build_account_node_day(df)

# sanity checks on real data
first_hist_day = res["feature_date"].min()
assert first_hist_day == pd.to_datetime(df["ts"]).min().normalize() + pd.Timedelta(days=1)
d1 = res[res["feature_date"] == first_hist_day]
assert d1["a_days_since_last"].dropna().eq(1).all(), "Sep 2 should only see Sep 1 history"
assert d1["a_days_active_frac"].isin([0.0, 1.0]).all()

out = DATA / "account_node_day.parquet"
res.to_parquet(out, index=False)
print("account_node_day:", res.shape, f"({time.time() - t0:.0f}s)")
print("dates:", res["feature_date"].min().date(), "->", res["feature_date"].max().date())
print("rows per date:")
print(res.groupby("feature_date").size().to_string())
print(res[FEATURES].describe().loc[["mean", "std", "min", "max"]].round(3).T)
print("NaN share per feature:")
print(res[FEATURES].isna().mean().round(3).to_string())