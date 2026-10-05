"""Historical pair-level features for AML transactions."""

import numpy as np
import pandas as pd


PAIR_FEATURE_COLS = [
    "g_pair_seen_before",
    "g_pair_prior_count_log",
    "g_reverse_pair_seen",
]


def build_pair_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build historical pair features using only strictly earlier days.

    For a transaction on day D, history contains only non-self
    transactions where date < D.

    Returns one row per input transaction with:
        txn_id
        g_pair_seen_before
        g_pair_prior_count_log
        g_reverse_pair_seen
    """
    required = {
        "txn_id",
        "date",
        "src_node",
        "dst_node",
        "is_self_loop",
    }

    missing = required - set(transactions.columns)

    if missing:
        raise ValueError(
            f"transactions missing columns: {sorted(missing)}"
        )

    if transactions["txn_id"].duplicated().any():
        raise ValueError("txn_id must be unique")

    if transactions.empty:
        return pd.DataFrame(columns=["txn_id"] + PAIR_FEATURE_COLS)

    tx = transactions[
        ["txn_id", "date", "src_node", "dst_node", "is_self_loop"]
    ].copy()

    tx["date"] = pd.to_datetime(tx["date"]).dt.normalize()

    # Self-loops stay as transaction rows but are excluded from
    # historical pair topology.
    history = tx.loc[
        ~tx["is_self_loop"],
        ["date", "src_node", "dst_node"],
    ].copy()

    # Number of transactions for each directed pair on each day.
    daily_pair_counts = (
        history.groupby(
            ["date", "src_node", "dst_node"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "pair_day_count"})
    )

    if daily_pair_counts.empty:
        result = tx[["txn_id"]].copy()
        result["g_pair_seen_before"] = 0
        result["g_pair_prior_count_log"] = 0.0
        result["g_reverse_pair_seen"] = 0
        return result[
            ["txn_id"] + PAIR_FEATURE_COLS
        ]

    daily_pair_counts = daily_pair_counts.sort_values(
        ["src_node", "dst_node", "date"]
    ).reset_index(drop=True)

    # Count only transactions from dates strictly before the
    # current date.
    daily_pair_counts["pair_prior_count"] = (
        daily_pair_counts
        .groupby(["src_node", "dst_node"])["pair_day_count"]
        .cumsum()
        .sub(daily_pair_counts["pair_day_count"])
    )

    daily_pair_counts["g_pair_seen_before"] = (
        daily_pair_counts["pair_prior_count"] > 0
    ).astype("int8")

    daily_pair_counts["g_pair_prior_count_log"] = np.log1p(
        daily_pair_counts["pair_prior_count"]
    )

    pair_history = daily_pair_counts[
        [
            "date",
            "src_node",
            "dst_node",
            "g_pair_seen_before",
            "g_pair_prior_count_log",
        ]
    ]

    # Build a table of pair occurrences that can be used to check
    # whether the reverse pair existed on an earlier day.
    #
    # Example:
    # current transaction A -> B
    # reverse historical pair = B -> A
    reverse_occurrences = history[
        ["date", "src_node", "dst_node"]
    ].rename(
        columns={
            "src_node": "dst_node",
            "dst_node": "src_node",
        }
    )

    reverse_occurrences = reverse_occurrences.drop_duplicates()

    # For each current pair/date, determine whether a reverse pair
    # existed on ANY strictly earlier date.
    current_pairs = daily_pair_counts[
        ["date", "src_node", "dst_node"]
    ].copy()

    reverse_lookup = reverse_occurrences.rename(
        columns={"date": "reverse_date"}
    )

    reverse_check = current_pairs.merge(
        reverse_lookup,
        on=["src_node", "dst_node"],
        how="left",
    )

    reverse_check["g_reverse_pair_seen"] = (
        reverse_check["reverse_date"]
        < reverse_check["date"]
    )

    reverse_flags = (
        reverse_check
        .groupby(
            ["date", "src_node", "dst_node"],
            as_index=False,
        )["g_reverse_pair_seen"]
        .any()
    )

    reverse_flags["g_reverse_pair_seen"] = (
        reverse_flags["g_reverse_pair_seen"]
        .astype("int8")
    )

    result = tx[
        ["txn_id", "date", "src_node", "dst_node"]
    ].copy()

    result = result.merge(
        pair_history,
        on=["date", "src_node", "dst_node"],
        how="left",
    )

    result = result.merge(
        reverse_flags,
        on=["date", "src_node", "dst_node"],
        how="left",
    )

    result["g_pair_seen_before"] = (
        result["g_pair_seen_before"]
        .fillna(0)
        .astype("int8")
    )

    result["g_pair_prior_count_log"] = (
        result["g_pair_prior_count_log"]
        .fillna(0.0)
        .astype("float64")
    )

    result["g_reverse_pair_seen"] = (
        result["g_reverse_pair_seen"]
        .fillna(0)
        .astype("int8")
    )

    return result[
        ["txn_id"] + PAIR_FEATURE_COLS
    ]