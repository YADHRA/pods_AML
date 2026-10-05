import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml.graph.nodes import validate_node_map


def test_valid_node_map():
    df = pd.DataFrame({
        "node_id": [0, 1, 2],
        "bank": ["002", "005", "037"],
        "account": ["A", "B", "C"],
    })

    validate_node_map(df)


def test_rejects_duplicate_node_id():
    df = pd.DataFrame({
        "node_id": [0, 0],
        "bank": ["002", "005"],
        "account": ["A", "B"],
    })

    with pytest.raises(ValueError, match="node_id values must be unique"):
        validate_node_map(df)


def test_rejects_duplicate_bank_account():
    df = pd.DataFrame({
        "node_id": [0, 1],
        "bank": ["002", "002"],
        "account": ["A", "A"],
    })

    with pytest.raises(ValueError, match=r"\(bank, account\) pairs"):
        validate_node_map(df)


def test_rejects_non_contiguous_ids():
    df = pd.DataFrame({
        "node_id": [0, 2],
        "bank": ["002", "005"],
        "account": ["A", "B"],
    })

    with pytest.raises(ValueError, match="contiguous"):
        validate_node_map(df)
