
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from aml.graph.writers import (
    prepare_graph_pair_txn,
    validate_graph_node_day_output,
    validate_graph_pair_txn_output,
    write_graph_node_day,
    write_graph_pair_txn,
)


def make_node_day():
    return pd.DataFrame({
        "feature_date": pd.to_datetime([
            "2022-09-02",
            "2022-09-02",
            "2022-09-03",
            "2022-09-03",
        ]),
        "node_id": [0, 1, 0, 1],
        "g_pagerank_log": [0.01, 0.02, 0.015, 0.025],
        "g_wcc_size_log": [1.1, 1.1, 1.3, 1.3],
        "g_n_unique_out_log": [0.0, 0.69, 0.69, 1.1],
        "g_n_unique_in_log": [0.69, 0.0, 1.1, 0.69],
        "g_has_sent_before": [0, 1, 1, 1],
        "g_has_received_before": [1, 0, 1, 1],
    })


def make_pair_features():
    return pd.DataFrame({
        "txn_id": [0, 1, 2, 3],
        "g_pair_seen_before": [0, 1, 0, 1],
        "g_pair_prior_count_log": [0.0, 0.69, 0.0, 1.1],
        "g_reverse_pair_seen": [0, 0, 1, 1],
    })


def make_transactions():
    return pd.DataFrame({
        "txn_id": [0, 1, 2, 3],
        "split": [
            "warmup",
            "train",
            "val",
            "test",
        ],
    })


def test_valid_node_day_passes():
    validate_graph_node_day_output(
        make_node_day()
    )


def test_duplicate_node_day_key_rejected():
    df = make_node_day()

    df.loc[1, "feature_date"] = df.loc[0, "feature_date"]
    df.loc[1, "node_id"] = df.loc[0, "node_id"]

    with pytest.raises(
        ValueError,
        match="keys must be unique",
    ):
        validate_graph_node_day_output(df)


def test_negative_node_feature_rejected():
    df = make_node_day()
    df.loc[0, "g_pagerank_log"] = -1.0

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        validate_graph_node_day_output(df)


def test_invalid_network_flag_rejected():
    df = make_node_day()
    df.loc[0, "g_has_sent_before"] = 2

    with pytest.raises(
        ValueError,
        match="only 0/1",
    ):
        validate_graph_node_day_output(df)


def test_nonfinite_node_feature_rejected():
    df = make_node_day()
    df.loc[0, "g_wcc_size_log"] = np.inf

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        validate_graph_node_day_output(df)


def test_valid_pair_output_passes():
    validate_graph_pair_txn_output(
        make_pair_features()
    )


def test_duplicate_pair_txn_rejected():
    df = make_pair_features()

    df.loc[1, "txn_id"] = df.loc[0, "txn_id"]

    with pytest.raises(
        ValueError,
        match="txn_id must be unique",
    ):
        validate_graph_pair_txn_output(df)


def test_invalid_pair_flag_rejected():
    df = make_pair_features()
    df.loc[0, "g_reverse_pair_seen"] = 3

    with pytest.raises(
        ValueError,
        match="only 0/1",
    ):
        validate_graph_pair_txn_output(df)


def test_negative_pair_count_rejected():
    df = make_pair_features()
    df.loc[0, "g_pair_prior_count_log"] = -0.5

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        validate_graph_pair_txn_output(df)


def test_warmup_pair_rows_are_removed():
    output = prepare_graph_pair_txn(
        make_pair_features(),
        make_transactions(),
    )

    assert list(output["txn_id"]) == [1, 2, 3]
    assert len(output) == 3


def test_pair_output_column_order():
    output = prepare_graph_pair_txn(
        make_pair_features(),
        make_transactions(),
    )

    assert list(output.columns) == [
        "txn_id",
        "g_pair_seen_before",
        "g_pair_prior_count_log",
        "g_reverse_pair_seen",
    ]


def test_missing_transaction_split_rejected():
    tx = make_transactions().drop(columns=["split"])

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        prepare_graph_pair_txn(
            make_pair_features(),
            tx,
        )


def test_unknown_split_rejected():
    tx = make_transactions()
    tx.loc[1, "split"] = "unknown"

    with pytest.raises(
        ValueError,
        match="unknown split",
    ):
        prepare_graph_pair_txn(
            make_pair_features(),
            tx,
        )


def test_missing_pair_txn_rejected():
    pair = make_pair_features()
    pair.loc[0, "txn_id"] = 999

    with pytest.raises(
        ValueError,
        match="missing from transactions",
    ):
        prepare_graph_pair_txn(
            pair,
            make_transactions(),
        )


def test_node_day_writer_roundtrip(tmp_path):
    path = tmp_path / "graph_node_day.parquet"

    write_graph_node_day(
        make_node_day(),
        path,
    )

    written = pd.read_parquet(path)

    assert list(written.columns) == [
        "feature_date",
        "node_id",
        "g_pagerank_log",
        "g_wcc_size_log",
        "g_n_unique_out_log",
        "g_n_unique_in_log",
        "g_has_sent_before",
        "g_has_received_before",
    ]

    assert len(written) == 4


def test_pair_writer_roundtrip(tmp_path):
    path = tmp_path / "graph_pair_txn.parquet"

    write_graph_pair_txn(
        make_pair_features(),
        make_transactions(),
        path,
    )

    written = pd.read_parquet(path)

    assert list(written.columns) == [
        "txn_id",
        "g_pair_seen_before",
        "g_pair_prior_count_log",
        "g_reverse_pair_seen",
    ]

    assert list(written["txn_id"]) == [1, 2, 3]

