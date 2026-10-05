import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml.graph.edges import build_historical_edges
def test_first_day_is_recorded_for_each_directed_pair():
    transactions = pd.DataFrame({
        "src_node": [0, 0, 0, 1],
        "dst_node": [1, 1, 2, 2],
        "date": pd.to_datetime([
            "2022-09-01",
            "2022-09-02",
            "2022-09-03",
            "2022-09-02",
        ]),
        "is_self_loop": [False, False, False, False],
    })

    result = build_historical_edges(transactions)

    result = result.sort_values(
        ["src_node", "dst_node"]
    ).reset_index(drop=True)

    expected = pd.DataFrame({
        "src_node": [0, 0, 1],
        "dst_node": [1, 2, 2],
        "first_day": pd.to_datetime([
            "2022-09-01",
            "2022-09-03",
            "2022-09-02",
        ]),
    })

    pd.testing.assert_frame_equal(result, expected)


def test_self_loops_are_excluded():
    transactions = pd.DataFrame({
        "src_node": [0, 0, 1],
        "dst_node": [0, 1, 1],
        "date": pd.to_datetime([
            "2022-09-01",
            "2022-09-01",
            "2022-09-02",
        ]),
        "is_self_loop": [True, False, True],
    })

    result = build_historical_edges(transactions)

    assert len(result) == 1
    assert result.iloc[0]["src_node"] == 0
    assert result.iloc[0]["dst_node"] == 1


def test_first_day_uses_earliest_occurrence():
    transactions = pd.DataFrame({
        "src_node": [2, 2, 2],
        "dst_node": [3, 3, 3],
        "date": pd.to_datetime([
            "2022-09-05",
            "2022-09-01",
            "2022-09-03",
        ]),
        "is_self_loop": [False, False, False],
    })

    result = build_historical_edges(transactions)

    assert result.iloc[0]["first_day"] == pd.Timestamp("2022-09-01")