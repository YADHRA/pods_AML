import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml.graph.pair_features import build_pair_features


def make_transactions():
    return pd.DataFrame({
        "txn_id": [1, 2, 3, 4, 5, 6, 7],
        "date": pd.to_datetime([
            "2022-09-01",  # 1: A -> B
            "2022-09-02",  # 2: A -> B
            "2022-09-03",  # 3: B -> A
            "2022-09-03",  # 4: A -> B (same day as reverse)
            "2022-09-04",  # 5: A -> B
            "2022-09-04",  # 6: B -> A
            "2022-09-05",  # 7: C -> C self-loop
        ]),
        "src_node": [10, 10, 20, 10, 10, 20, 30],
        "dst_node": [20, 20, 10, 20, 20, 10, 30],
        "is_self_loop": [False, False, False, False, False, False, True],
    })


def test_first_occurrence_has_no_pair_history():
    result = build_pair_features(make_transactions())

    row = result.loc[result.txn_id == 1].iloc[0]

    assert row["g_pair_seen_before"] == 0
    assert row["g_pair_prior_count_log"] == 0.0
    assert row["g_reverse_pair_seen"] == 0


def test_repeated_pair_uses_previous_days_only():
    result = build_pair_features(make_transactions())

    row = result.loc[result.txn_id == 2].iloc[0]

    assert row["g_pair_seen_before"] == 1
    assert row["g_pair_prior_count_log"] == np.log1p(1)
    assert row["g_reverse_pair_seen"] == 0


def test_reverse_pair_is_detected_from_previous_day():
    result = build_pair_features(make_transactions())

    row = result.loc[result.txn_id == 3].iloc[0]

    assert row["g_pair_seen_before"] == 0
    assert row["g_pair_prior_count_log"] == 0.0
    assert row["g_reverse_pair_seen"] == 1


def test_same_day_reverse_pair_does_not_leak():
    result = build_pair_features(make_transactions())

    row = result.loc[result.txn_id == 4].iloc[0]

    assert row["g_pair_seen_before"] == 1
    assert row["g_reverse_pair_seen"] == 0


def test_prior_count_uses_only_previous_days():
    result = build_pair_features(make_transactions())

    row = result.loc[result.txn_id == 5].iloc[0]

    assert row["g_pair_seen_before"] == 1
    assert row["g_pair_prior_count_log"] == np.log1p(3)


def test_self_loop_has_zero_pair_features():
    result = build_pair_features(make_transactions())

    row = result.loc[result.txn_id == 7].iloc[0]

    assert row["g_pair_seen_before"] == 0
    assert row["g_pair_prior_count_log"] == 0.0
    assert row["g_reverse_pair_seen"] == 0


def test_output_has_one_row_per_transaction():
    transactions = make_transactions()

    result = build_pair_features(transactions)

    assert len(result) == len(transactions)
    assert result["txn_id"].is_unique

    assert list(result.columns) == [
        "txn_id",
        "g_pair_seen_before",
        "g_pair_prior_count_log",
        "g_reverse_pair_seen",
    ]


def test_duplicate_txn_id_is_rejected():
    transactions = make_transactions()

    transactions.loc[1, "txn_id"] = transactions.loc[0, "txn_id"]

    with pytest.raises(ValueError, match="txn_id must be unique"):
        build_pair_features(transactions)


def test_missing_columns_are_rejected():
    transactions = make_transactions().drop(columns=["src_node"])

    with pytest.raises(ValueError, match="missing columns"):
        build_pair_features(transactions)


def test_empty_input_returns_contract_columns():
    transactions = make_transactions().iloc[0:0].copy()

    result = build_pair_features(transactions)

    assert list(result.columns) == [
        "txn_id",
        "g_pair_seen_before",
        "g_pair_prior_count_log",
        "g_reverse_pair_seen",
    ]

    assert result.empty