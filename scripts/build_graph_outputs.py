
"""Build end-to-end graph feature outputs from the stub or processed data."""

from pathlib import Path
import sys

# Make the project's src/ package importable when this script
# is executed directly with:
# python scripts/build_graph_outputs.py
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from aml.common.schema import (
    assert_graph_node_day,
    assert_graph_pair_txn,
)
from aml.graph.edges import build_historical_edges
from aml.graph.network_features import build_network_features
from aml.graph.nodes import load_node_map
from aml.graph.pair_features import build_pair_features
from aml.graph.pagerank import build_pagerank_features
from aml.graph.snapshots import build_snapshot_edges
from aml.graph.writers import (
    write_graph_node_day,
    write_graph_pair_txn,
)


def build_graph_node_day_output(
    transactions: pd.DataFrame,
    node_map: pd.DataFrame,
) -> pd.DataFrame:
    """Build graph features for every transaction date and canonical node."""

    historical_edges = build_historical_edges(
        transactions
    )

    node_ids = node_map["node_id"]

    feature_dates = (
        pd.to_datetime(transactions["date"])
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    all_daily_features = []

    for feature_date in feature_dates:
        snapshot_edges = build_snapshot_edges(
            historical_edges,
            feature_date,
        )

        network_features = build_network_features(
            snapshot_edges,
            node_ids,
        )

        pagerank_features = build_pagerank_features(
            snapshot_edges,
            node_ids,
        )

        # Both feature tables contain exactly one row
        # per canonical node_id.
        daily = network_features.merge(
            pagerank_features,
            on="node_id",
            how="inner",
            validate="one_to_one",
        )

        daily["feature_date"] = pd.Timestamp(
            feature_date
        ).normalize()

        all_daily_features.append(
            daily[
                [
                    "feature_date",
                    "node_id",
                    "g_pagerank_log",
                    "g_wcc_size_log",
                    "g_n_unique_out_log",
                    "g_n_unique_in_log",
                    "g_has_sent_before",
                    "g_has_received_before",
                ]
            ]
        )

    if not all_daily_features:
        return pd.DataFrame(
            columns=[
                "feature_date",
                "node_id",
                "g_pagerank_log",
                "g_wcc_size_log",
                "g_n_unique_out_log",
                "g_n_unique_in_log",
                "g_has_sent_before",
                "g_has_received_before",
            ]
        )

    output = pd.concat(
        all_daily_features,
        ignore_index=True,
    )

    output = output.sort_values(
        ["feature_date", "node_id"]
    ).reset_index(drop=True)

    return output


def build_graph_outputs(
    input_dir: str | Path = "data_stub",
    output_dir: str | Path = "data_stub",
) -> None:
    """Build and validate both graph contract outputs."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    transactions_path = (
        input_dir / "transactions_clean.parquet"
    )

    node_map_path = (
        input_dir / "node_map.parquet"
    )

    # ---------------------------------------------------------
    # Load input data
    # ---------------------------------------------------------

    transactions = pd.read_parquet(
        transactions_path
    )

    node_map = load_node_map(
        node_map_path
    )

    # ---------------------------------------------------------
    # Build graph_node_day
    # ---------------------------------------------------------

    graph_node_day = build_graph_node_day_output(
        transactions,
        node_map,
    )

    assert_graph_node_day(
        graph_node_day
    )

    write_graph_node_day(
        graph_node_day,
        output_dir / "graph_node_day.parquet",
    )

    # ---------------------------------------------------------
    # Build graph_pair_txn
    # ---------------------------------------------------------

    pair_features = build_pair_features(
        transactions
    )

    write_graph_pair_txn(
        pair_features,
        transactions,
        output_dir / "graph_pair_txn.parquet",
    )

    # ---------------------------------------------------------
    # Read outputs back and validate again
    # ---------------------------------------------------------

    written_node_day = pd.read_parquet(
        output_dir / "graph_node_day.parquet"
    )

    written_pair_txn = pd.read_parquet(
        output_dir / "graph_pair_txn.parquet"
    )

    assert_graph_node_day(
        written_node_day
    )

    assert_graph_pair_txn(
        written_pair_txn
    )

    # ---------------------------------------------------------
    # Final integration checks
    # ---------------------------------------------------------

    warmup_ids = set(
        transactions.loc[
            transactions["split"] == "warmup",
            "txn_id",
        ]
    )

    warmup_pair_rows = written_pair_txn[
        "txn_id"
    ].isin(warmup_ids).sum()

    # The contract excludes warm-up rows from graph_pair_txn.
    if warmup_pair_rows != 0:
        raise ValueError(
            "graph_pair_txn contains warm-up transaction rows"
        )

    # Every transaction date must have one graph feature row
    # for every canonical node.
    expected_node_day_rows = (
        transactions["date"]
        .dt.normalize()
        .nunique()
        * len(node_map)
    )

    if len(written_node_day) != expected_node_day_rows:
        raise ValueError(
            "graph_node_day row count does not match "
            "feature_dates × canonical_nodes"
        )

    # ---------------------------------------------------------
    # Additional safety checks
    # ---------------------------------------------------------

    # graph_node_day must have exactly one row per
    # (feature_date, node_id).
    if written_node_day.duplicated(
        ["feature_date", "node_id"]
    ).any():
        raise ValueError(
            "graph_node_day contains duplicate "
            "(feature_date, node_id) keys"
        )

    # Every feature date must contain all canonical nodes.
    expected_dates = (
        transactions["date"]
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    for feature_date in expected_dates:
        daily_rows = written_node_day[
            written_node_day["feature_date"]
            == feature_date
        ]

        if len(daily_rows) != len(node_map):
            raise ValueError(
                "graph_node_day does not contain "
                "all canonical nodes for feature date "
                f"{feature_date}"
            )

    # All graph feature values must be finite.
    numeric_graph_cols = [
        "g_pagerank_log",
        "g_wcc_size_log",
        "g_n_unique_out_log",
        "g_n_unique_in_log",
        "g_has_sent_before",
        "g_has_received_before",
    ]

    if not written_node_day[
        numeric_graph_cols
    ].apply(
        lambda column: pd.api.types.is_numeric_dtype(column)
    ).all():
        raise ValueError(
            "graph_node_day contains non-numeric graph feature columns"
        )

    if not written_node_day[
        [
            "g_pagerank_log",
            "g_wcc_size_log",
            "g_n_unique_out_log",
            "g_n_unique_in_log",
        ]
    ].apply(
        lambda column: column.notna().all()
        and column.map(pd.api.types.is_number).all()
    ).all():
        raise ValueError(
            "graph_node_day contains invalid numeric feature values"
        )

    # graph_pair_txn must contain only prediction splits.
    prediction_splits = {
        "train",
        "val",
        "test",
        "tail",
    }

    pair_txn_ids = set(
        written_pair_txn["txn_id"]
    )

    pair_splits = set(
        transactions.loc[
            transactions["txn_id"].isin(pair_txn_ids),
            "split",
        ].unique()
    )

    if not pair_splits.issubset(
        prediction_splits
    ):
        raise ValueError(
            "graph_pair_txn contains transaction rows "
            f"outside prediction splits: "
            f"{sorted(pair_splits - prediction_splits)}"
        )

    # ---------------------------------------------------------
    # Success summary
    # ---------------------------------------------------------

    print("GRAPH OUTPUT BUILD OK")
    print(
        f"transactions = {len(transactions)}"
    )
    print(
        f"nodes = {len(node_map)}"
    )
    print(
        f"feature dates = "
        f"{written_node_day['feature_date'].nunique()}"
    )
    print(
        f"graph_node_day rows = "
        f"{len(written_node_day)}"
    )
    print(
        f"graph_pair_txn rows = "
        f"{len(written_pair_txn)}"
    )
    print(
        f"warmup pair rows = "
        f"{warmup_pair_rows}"
    )


if __name__ == "__main__":
    build_graph_outputs()
