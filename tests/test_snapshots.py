import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml.graph.snapshots import build_snapshot_edges


def test_snapshot_uses_only_strictly_earlier_edges():
    historical_edges = pd.DataFrame({
        "src_node": [0, 1, 2],
        "dst_node": [1, 2, 3],
        "first_day": pd.to_datetime([
            "2022-09-01",
            "2022-09-02",
            "2022-09-03",
        ]),
    })

    result = build_snapshot_edges(
        historical_edges,
        pd.Timestamp("2022-09-03"),
    )

    expected = pd.DataFrame({
        "src_node": [0, 1],
        "dst_node": [1, 2],
        "first_day": pd.to_datetime([
            "2022-09-01",
            "2022-09-02",
        ]),
    })

    pd.testing.assert_frame_equal(result, expected)


def test_first_day_has_no_historical_edges():
    historical_edges = pd.DataFrame({
        "src_node": [0, 1],
        "dst_node": [1, 2],
        "first_day": pd.to_datetime([
            "2022-09-01",
            "2022-09-01",
        ]),
    })

    result = build_snapshot_edges(
        historical_edges,
        pd.Timestamp("2022-09-01"),
    )

    assert result.empty


def test_snapshot_does_not_modify_input():
    historical_edges = pd.DataFrame({
        "src_node": [0, 1],
        "dst_node": [1, 2],
        "first_day": pd.to_datetime([
            "2022-09-01",
            "2022-09-02",
        ]),
    })

    original = historical_edges.copy(deep=True)

    build_snapshot_edges(
        historical_edges,
        pd.Timestamp("2022-09-03"),
    )

    pd.testing.assert_frame_equal(
        historical_edges,
        original,
    )


def test_missing_columns_are_rejected():
    historical_edges = pd.DataFrame({
        "src_node": [0],
        "dst_node": [1],
    })

    with pytest.raises(ValueError, match="missing columns"):
        build_snapshot_edges(
            historical_edges,
            pd.Timestamp("2022-09-02"),
        )