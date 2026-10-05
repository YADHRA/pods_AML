"""Historical directed edge table for the AML graph."""

import pandas as pd


def build_historical_edges(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build the first-seen day for every directed non-self transaction pair.

    Self-loops remain transaction rows but are excluded from graph topology.

    Returns:
        DataFrame with:
            src_node
            dst_node
            first_day
    """
    required = {"src_node", "dst_node", "date", "is_self_loop"}
    missing = required - set(transactions.columns)

    if missing:
        raise ValueError(f"transactions missing columns: {sorted(missing)}")

    edges = transactions.loc[
        ~transactions["is_self_loop"],
        ["src_node", "dst_node", "date"],
    ].copy()

    edges = (
        edges.groupby(
            ["src_node", "dst_node"],
            as_index=False,
            sort=False,
        )["date"]
        .min()
        .rename(columns={"date": "first_day"})
    )

    return edges