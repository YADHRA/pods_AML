"""End-to-end integration tests for graph output generation."""

from pathlib import Path
import sys

# Make the project's src/ package importable when pytest
# collects this test file directly.
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
from aml.graph.snapshots import build_snapshot_edges


DATA_DIR = ROOT / "data_stub"

TRANSACTIONS_PATH = (
    DATA_DIR / "transactions_clean.parquet"
)

NODE_MAP_PATH = (
    DATA_DIR / "node_map.parquet"
)

GRAPH_NODE_DAY_PATH = (
    DATA_DIR / "graph_node_day.parquet"
)

GRAPH_PAIR_TXN_PATH = (
    DATA_DIR / "graph_pair_txn.parquet"
)


def load_stub_data():
    """Load the generated stub inputs and graph outputs."""

    transactions = pd.read_parquet(
        TRANSACTIONS_PATH
    )

    node_map = pd.read_parquet(
        NODE_MAP_PATH
    )

    graph_node_day = pd.read_parquet(
        GRAPH_NODE_DAY_PATH
    )

    graph_pair_txn = pd.read_parquet(
        GRAPH_PAIR_TXN_PATH
    )

    return (
        transactions,
        node_map,
        graph_node_day,
        graph_pair_txn,
    )


def test_graph_node_day_has_expected_shape():
    (
        transactions,
        node_map,
        graph_node_day,
        _,
    ) = load_stub_data()

    expected_dates = (
        transactions["date"]
        .dt.normalize()
        .nunique()
    )

    expected_rows = (
        expected_dates * len(node_map)
    )

    assert len(graph_node_day) == expected_rows


def test_graph_node_day_keys_are_unique():
    (
        _,
        _,
        graph_node_day,
        _,
    ) = load_stub_data()

    assert not graph_node_day.duplicated(
        ["feature_date", "node_id"]
    ).any()


def test_graph_node_day_covers_all_nodes_each_day():
    (
        transactions,
        node_map,
        graph_node_day,
        _,
    ) = load_stub_data()

    expected_nodes = set(
        node_map["node_id"]
    )

    feature_dates = (
        transactions["date"]
        .dt.normalize()
        .drop_duplicates()
    )

    for feature_date in feature_dates:
        actual_nodes = set(
            graph_node_day.loc[
                graph_node_day["feature_date"]
                == feature_date,
                "node_id",
            ]
        )

        assert actual_nodes == expected_nodes


def test_graph_node_day_matches_contract():
    (
        _,
        _,
        graph_node_day,
        _,
    ) = load_stub_data()

    assert_graph_node_day(
        graph_node_day
    )


def test_graph_node_day_has_no_future_edges():
    (
        transactions,
        _,
        graph_node_day,
        _,
    ) = load_stub_data()

    historical_edges = build_historical_edges(
        transactions
    )

    feature_dates = (
        graph_node_day["feature_date"]
        .drop_duplicates()
        .sort_values()
    )

    for feature_date in feature_dates:
        snapshot = build_snapshot_edges(
            historical_edges,
            feature_date,
        )

        if not snapshot.empty:
            assert (
                snapshot["first_day"]
                < feature_date
            ).all()


def test_pair_output_excludes_warmup():
    (
        transactions,
        _,
        _,
        graph_pair_txn,
    ) = load_stub_data()

    warmup_ids = set(
        transactions.loc[
            transactions["split"] == "warmup",
            "txn_id",
        ]
    )

    assert not graph_pair_txn[
        "txn_id"
    ].isin(warmup_ids).any()


def test_pair_output_matches_contract():
    (
        _,
        _,
        _,
        graph_pair_txn,
    ) = load_stub_data()

    assert_graph_pair_txn(
        graph_pair_txn
    )


def test_pair_output_has_unique_transaction_ids():
    (
        _,
        _,
        _,
        graph_pair_txn,
    ) = load_stub_data()

    assert graph_pair_txn[
        "txn_id"
    ].is_unique


def test_pair_output_contains_only_prediction_splits():
    (
        transactions,
        _,
        _,
        graph_pair_txn,
    ) = load_stub_data()

    prediction_splits = {
        "train",
        "val",
        "test",
        "tail",
    }

    pair_ids = set(
        graph_pair_txn["txn_id"]
    )

    pair_splits = set(
        transactions.loc[
            transactions["txn_id"].isin(pair_ids),
            "split",
        ].unique()
    )

    assert pair_splits.issubset(
        prediction_splits
    )