"""Node-map loading and validation for the AML graph."""

from pathlib import Path

import pandas as pd

from aml.common.schema import NODE_MAP_DTYPES


def validate_node_map(df: pd.DataFrame) -> None:
    """Validate P1's node map without modifying node IDs."""
    expected = list(NODE_MAP_DTYPES)

    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"node_map missing columns: {missing}")

    if df["node_id"].isna().any():
        raise ValueError("node_id contains missing values")

    if not df["node_id"].is_unique:
        raise ValueError("node_id values must be unique")

    if df[["bank", "account"]].duplicated().any():
        raise ValueError("(bank, account) pairs must be unique")

    ids = df["node_id"].to_numpy()
    expected_ids = range(len(df))

    if list(ids) != list(expected_ids):
        raise ValueError("node_id values must be contiguous starting at 0")


def load_node_map(path: str | Path) -> pd.DataFrame:
    """Load P1's canonical node map and validate it."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"node_map not found: {path}")

    df = pd.read_parquet(path)

    validate_node_map(df)

    return df
