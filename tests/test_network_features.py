import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml.graph.network_features import build_network_features


def make_edges():
    return pd.DataFrame({
        "src_node": [0, 0, 1, 2],
        "dst_node": [1, 2, 2, 3],
    })


def test_unique_in_and_out_counts():
    result = build_network_features(
        make_edges(),
        [0, 1, 2, 3, 4],
    )

    row = result.loc[result.node_id == 0].iloc[0]

    # Node 0 -> 1 and 0 -> 2
    assert row["g_n_unique_out_log"] == np.log1p(2)

    # Node 0 has received nothing
    assert row["g_n_unique_in_log"] == 0.0


def test_node_with_both_in_and_out():
    result = build_network_features(
        make_edges(),
        [0, 1, 2, 3, 4],
    )

    row = result.loc[result.node_id == 2].iloc[0]

    # 2 -> 3
    assert row["g_n_unique_out_log"] == np.log1p(1)

    # 0 -> 2 and 1 -> 2
    assert row["g_n_unique_in_log"] == np.log1p(2)


def test_sent_and_received_flags():
    result = build_network_features(
        make_edges(),
        [0, 1, 2, 3, 4],
    )

    row0 = result.loc[result.node_id == 0].iloc[0]
    row3 = result.loc[result.node_id == 3].iloc[0]

    assert row0["g_has_sent_before"] == 1
    assert row0["g_has_received_before"] == 0

    assert row3["g_has_sent_before"] == 0
    assert row3["g_has_received_before"] == 1


def test_wcc_size():
    result = build_network_features(
        make_edges(),
        [0, 1, 2, 3, 4],
    )

    # 0 -> 1, 0 -> 2, 1 -> 2, 2 -> 3
    # All four nodes belong to one weakly connected component.
    for node_id in [0, 1, 2, 3]:
        row = result.loc[result.node_id == node_id].iloc[0]
        assert row["g_wcc_size_log"] == np.log1p(4)


def test_isolated_node():
    result = build_network_features(
        make_edges(),
        [0, 1, 2, 3, 4],
    )

    row = result.loc[result.node_id == 4].iloc[0]

    assert row["g_n_unique_out_log"] == 0.0
    assert row["g_n_unique_in_log"] == 0.0
    assert row["g_has_sent_before"] == 0
    assert row["g_has_received_before"] == 0
    assert row["g_wcc_size_log"] == np.log1p(1)


def test_empty_history():
    result = build_network_features(
        pd.DataFrame(columns=["src_node", "dst_node"]),
        [0, 1, 2],
    )

    assert len(result) == 3

    assert (result["g_n_unique_out_log"] == 0.0).all()
    assert (result["g_n_unique_in_log"] == 0.0).all()
    assert (result["g_has_sent_before"] == 0).all()
    assert (result["g_has_received_before"] == 0).all()
    assert (result["g_wcc_size_log"] == np.log1p(1)).all()


def test_duplicate_edges_are_rejected():
    edges = pd.DataFrame({
        "src_node": [0, 0],
        "dst_node": [1, 1],
    })

    with pytest.raises(
        ValueError,
        match="unique directed edges",
    ):
        build_network_features(edges, [0, 1])


def test_duplicate_node_ids_are_rejected():
    with pytest.raises(
        ValueError,
        match="node_ids must be unique",
    ):
        build_network_features(
            make_edges(),
            [0, 1, 1, 2],
        )


def test_missing_columns_are_rejected():
    edges = pd.DataFrame({
        "src_node": [0],
    })

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        build_network_features(edges, [0, 1])


def test_self_loop_does_not_create_topology():
    edges = pd.DataFrame({
        "src_node": [0, 0],
        "dst_node": [0, 1],
    })

    # The self-loop should NOT be treated as graph topology.
    # The function expects historical_edges to already exclude
    # self-loops, so this test confirms that accidental self-loops
    # are rejected rather than silently included.
    with pytest.raises(ValueError):
        build_network_features(edges, [0, 1])