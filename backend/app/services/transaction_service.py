from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

TRANSACTIONS_FILE = DATA_DIR / "transactions_clean.parquet"
NODE_MAP_FILE = DATA_DIR / "node_map.parquet"
SHAP_TXN_FILE = RESULTS_DIR / "shap_by_txn_m2_full_seed42.parquet"
INSPECTOR_FILE = RESULTS_DIR / "transaction_inspector.parquet"

TRANSACTION_COLUMNS = [
    "txn_id",
    "ts",
    "date",
    "src_node",
    "dst_node",
    "src_bank",
    "dst_bank",
    "amount_paid",
    "amount_received",
    "cur_paid",
    "cur_recv",
    "payment_format",
    "is_laundering",
    "split",
    "is_self_loop",
]

_node_map_cache = None

def get_node_map():
    global _node_map_cache
    if _node_map_cache is None:
        _node_map_cache = pd.read_parquet(NODE_MAP_FILE)
    return _node_map_cache


def get_transaction(txn_id: int):
    transactions = pd.read_parquet(
        TRANSACTIONS_FILE,
        columns=TRANSACTION_COLUMNS,
        filters=[("txn_id", "==", txn_id)],
    )

    if transactions.empty:
        return None

    row = transactions.iloc[0]
    nm = get_node_map()

    src_node = int(row["src_node"])
    dst_node = int(row["dst_node"])

    src_acc_row = nm[nm["node_id"] == src_node]
    dst_acc_row = nm[nm["node_id"] == dst_node]

    src_account = str(src_acc_row["account"].iloc[0]) if not src_acc_row.empty else f"Node {src_node}"
    dst_account = str(dst_acc_row["account"].iloc[0]) if not dst_acc_row.empty else f"Node {dst_node}"

    return {
        "txn_id": int(row["txn_id"]),
        "timestamp": str(row["ts"]),
        "date": str(row["date"]),
        "src_node": src_node,
        "dst_node": dst_node,
        "src_bank": str(row["src_bank"]),
        "dst_bank": str(row["dst_bank"]),
        "src_account": src_account,
        "dst_account": dst_account,
        "amount_paid": float(row["amount_paid"]),
        "amount_received": float(row["amount_received"]),
        "currency_paid": str(row["cur_paid"]),
        "currency_received": str(row["cur_recv"]),
        "payment_format": str(row["payment_format"]),
        "is_laundering": int(row["is_laundering"]),
        "split": str(row["split"]),
        "is_self_loop": bool(row["is_self_loop"]),
    }


def get_explanation(txn_id: int, model: str = "m2"):
    model = model.lower()
    features = []

    # Try reading from detailed shap_by_txn file
    if SHAP_TXN_FILE.exists():
        shap_df = pd.read_parquet(
            SHAP_TXN_FILE,
            filters=[("txn_id", "==", txn_id)],
        )
        if not shap_df.empty:
            if model == "m1":
                # M1 uses strictly transaction-level features (t_*)
                shap_df = shap_df[shap_df["feature"].str.startswith("t_")]
            # Sort by absolute SHAP contribution
            shap_df = shap_df.reindex(shap_df["shap_value"].abs().sort_values(ascending=False).index)
            for _, r in shap_df.head(8).iterrows():
                val = r["feature_value_str"] if pd.notna(r["feature_value_str"]) else r["feature_value"]
                if pd.isna(val) or val is None or str(val) == "nan":
                    val = None
                elif isinstance(val, (int, float)):
                    f_val = float(val)
                    if pd.isna(f_val):
                        val = None
                    else:
                        val = round(f_val, 4)
                else:
                    val = str(val)

                s_val = float(r["shap_value"])
                if pd.isna(s_val):
                    s_val = 0.0
                abs_s = abs(s_val)
                impact = "High" if abs_s >= 0.5 else "Medium" if abs_s >= 0.1 else "Low"
                features.append({
                    "feature": str(r["feature"]),
                    "shap_value": round(s_val, 4),
                    "feature_value": val,
                    "impact": impact,
                })

    # Fallback to inspector parquet if shap_by_txn had no row
    if not features and INSPECTOR_FILE.exists():
        insp_df = pd.read_parquet(
            INSPECTOR_FILE,
            filters=[("txn_id", "==", txn_id)],
        )
        if not insp_df.empty:
            irow = insp_df.iloc[0]
            for idx in [1, 2, 3]:
                feat_name = str(irow.get(f"shap{idx}_feature", ""))
                if not feat_name or feat_name == "None":
                    continue
                if model == "m1" and not feat_name.startswith("t_"):
                    continue
                s_val = float(irow.get(f"shap{idx}_value", 0.0))
                abs_s = abs(s_val)
                impact = "High" if abs_s >= 0.5 else "Medium" if abs_s >= 0.1 else "Low"
                features.append({
                    "feature": feat_name,
                    "shap_value": s_val,
                    "feature_value": None,
                    "impact": impact,
                })

    return {
        "txn_id": txn_id,
        "model": model,
        "features": features,
    }