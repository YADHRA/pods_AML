
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from aml.graph.pagerank import build_pagerank_features


def make_edges():
    return pd.DataFrame({
        "src_node": [0, 1, 2],
        "dst_node": [1, 2, 0],
        "first_day": pd.to_datetime([
            "2022-09-01",
            "2022-09-01",
            "2022-09-02",
        ]),
    })


def test_output_contains_all_nodes():
    result = build_pagerank_features(
        make_edges(),
        [0, 1, 2, 3],
    )

    assert len(result) == 4
    assert result["node_id"].nunique() == 4
    assert set(result["node_id"]) == {0, 1, 2, 3}


def test_pagerank_is_positive_and_log_transformed():
    result = build_pagerank_features(
        make_edges(),
        [0, 1, 2],
    )

    assert (result["g_pagerank_log"] > 0).all()
    assert np.isfinite(
        result["g_pagerank_log"]
    ).all()


def test_cycle_has_equal_pagerank():
    edges = pd.DataFrame({
        "src_node": [0, 1, 2],
        "dst_node": [1, 2, 0],
        "first_day": pd.to_datetime([
            "2022-09-01",
            "2022-09-01",
            "2022-09-01",
        ]),
    })

    result = build_pagerank_features(
        edges,
        [0, 1, 2],
    )

    values = result["g_pagerank_log"].to_numpy()

    assert np.allclose(
        values,
        values[0],
        atol=1e-10,
    )


def test_isolated_node_is_included():
    result = build_pagerank_features(
        make_edges(),
        [0, 1, 2, 3],
    )

    row = result.loc[
        result.node_id == 3
    ].iloc[0]

    assert np.isfinite(
        row["g_pagerank_log"]
    )

    assert row["g_pagerank_log"] > 0


def test_empty_history_gives_uniform_pagerank():
    result = build_pagerank_features(
        pd.DataFrame(
            columns=[
                "src_node",
                "dst_node",
                "first_day",
            ]
        ),
        [0, 1, 2, 3],
    )

    expected = np.log1p(1.0 / 4)

    assert len(result) == 4

    assert np.allclose(
        result["g_pagerank_log"],
        expected,
    )


def test_duplicate_edges_are_rejected():
    edges = pd.DataFrame({
        "src_node": [0, 0],
        "dst_node": [1, 1],
        "first_day": pd.to_datetime([
            "2022-09-01",
            "2022-09-02",
        ]),
    })

    with pytest.raises(
        ValueError,
        match="unique directed edges",
    ):
        build_pagerank_features(
            edges,
            [0, 1],
        )


def test_duplicate_node_ids_are_rejected():
    with pytest.raises(
        ValueError,
        match="node_ids must be unique",
    ):
        build_pagerank_features(
            make_edges(),
            [0, 1, 1, 2],
        )


def test_missing_columns_are_rejected():
    edges = pd.DataFrame({
        "src_node": [0],
        "dst_node": [1],
    })

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        build_pagerank_features(
            edges,
            [0, 1],
        )


def test_self_loops_are_rejected():
    edges = pd.DataFrame({
        "src_node": [0],
        "dst_node": [0],
        "first_day": pd.to_datetime([
            "2022-09-01",
        ]),
    })

    with pytest.raises(
        ValueError,
        match="must not contain self-loops",
    ):
        build_pagerank_features(
            edges,
            [0],
        )


def test_unknown_nodes_are_rejected():
    edges = pd.DataFrame({
        "src_node": [0, 99],
        "dst_node": [1, 1],
        "first_day": pd.to_datetime([
            "2022-09-01",
            "2022-09-01",
        ]),
    })

    with pytest.raises(
        ValueError,
        match="not present",
    ):
        build_pagerank_features(
            edges,
            [0, 1, 2],
        )

