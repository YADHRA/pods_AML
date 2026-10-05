"""Historical daily graph snapshots for the AML graph."""

import pandas as pd


def build_snapshot_edges(
    historical_edges: pd.DataFrame,
    feature_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return graph edges available strictly before feature_date.

    The strict '< feature_date' rule prevents same-day information
    from leaking into features for that day.

    Args:
        historical_edges:
            DataFrame containing src_node, dst_node, first_day.
        feature_date:
            Date for which the historical graph is required.

    Returns:
        DataFrame containing only edges with first_day < feature_date.
    """
    required = {"src_node", "dst_node", "first_day"}
    missing = required - set(historical_edges.columns)

    if missing:
        raise ValueError(
            f"historical_edges missing columns: {sorted(missing)}"
        )

    feature_date = pd.Timestamp(feature_date).normalize()

    edges = historical_edges.loc[
        historical_edges["first_day"] < feature_date,
        ["src_node", "dst_node", "first_day"],
    ].copy()

    return edges.reset_index(drop=True)