
"""Validated writers for graph feature contract outputs."""

from pathlib import Path

import numpy as np
import pandas as pd

from aml.common.schema import (
    GRAPH_NODE_DAY_COLS,
    GRAPH_NODE_DAY_KEYS,
    GRAPH_PAIR_TXN_COLS,
    GRAPH_PAIR_TXN_KEYS,
    SPLITS,
)


NODE_DAY_OUTPUT_COLS = GRAPH_NODE_DAY_KEYS + GRAPH_NODE_DAY_COLS
PAIR_OUTPUT_COLS = GRAPH_PAIR_TXN_KEYS + GRAPH_PAIR_TXN_COLS


def validate_graph_node_day_output(df: pd.DataFrame) -> None:
    """Validate graph_node_day before writing."""

    missing = [
        col
        for col in NODE_DAY_OUTPUT_COLS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"graph_node_day missing columns: {missing}"
        )

    if df.duplicated(GRAPH_NODE_DAY_KEYS).any():
        raise ValueError(
            "graph_node_day keys must be unique"
        )

    if df["feature_date"].isna().any():
        raise ValueError(
            "graph_node_day feature_date contains missing values"
        )

    if df["node_id"].isna().any():
        raise ValueError(
            "graph_node_day node_id contains missing values"
        )

    numeric_cols = [
        "g_pagerank_log",
        "g_wcc_size_log",
        "g_n_unique_out_log",
        "g_n_unique_in_log",
    ]

    for col in numeric_cols:
        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"{col} contains non-numeric or missing values"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{col} contains non-finite values"
            )

        if (values < 0).any():
            raise ValueError(
                f"{col} contains negative values"
            )

    for col in [
        "g_has_sent_before",
        "g_has_received_before",
    ]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(
                f"{col} must contain only 0/1"
            )


def validate_graph_pair_txn_output(df: pd.DataFrame) -> None:
    """Validate graph_pair_txn before writing."""

    missing = [
        col
        for col in PAIR_OUTPUT_COLS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"graph_pair_txn missing columns: {missing}"
        )

    if df["txn_id"].isna().any():
        raise ValueError(
            "graph_pair_txn txn_id contains missing values"
        )

    if df["txn_id"].duplicated().any():
        raise ValueError(
            "graph_pair_txn txn_id must be unique"
        )

    if not pd.api.types.is_integer_dtype(df["txn_id"]):
        raise ValueError(
            "graph_pair_txn txn_id must be integer"
        )

    for col in [
        "g_pair_seen_before",
        "g_reverse_pair_seen",
    ]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(
                f"{col} must contain only 0/1"
            )

    values = pd.to_numeric(
        df["g_pair_prior_count_log"],
        errors="coerce",
    )

    if values.isna().any():
        raise ValueError(
            "g_pair_prior_count_log contains "
            "non-numeric or missing values"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "g_pair_prior_count_log contains "
            "non-finite values"
        )

    if (values < 0).any():
        raise ValueError(
            "g_pair_prior_count_log contains "
            "negative values"
        )


def _write_validated_parquet(
    df: pd.DataFrame,
    path: str | Path,
    columns: list[str],
) -> Path:
    """Write an already validated DataFrame in contract order."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = df[columns].copy()

    output.to_parquet(
        path,
        index=False,
    )

    return path


def write_graph_node_day(
    df: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Validate and write graph_node_day.parquet."""

    validate_graph_node_day_output(df)

    path = _write_validated_parquet(
        df,
        path,
        NODE_DAY_OUTPUT_COLS,
    )

    written = pd.read_parquet(path)

    validate_graph_node_day_output(written)

    return path


def prepare_graph_pair_txn(
    pair_features: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare graph_pair_txn for contract output.

    Warm-up transactions are excluded.
    """

    required_pair = {
        "txn_id",
        *GRAPH_PAIR_TXN_COLS,
    }

    missing_pair = (
        required_pair - set(pair_features.columns)
    )

    if missing_pair:
        raise ValueError(
            "pair_features missing columns: "
            f"{sorted(missing_pair)}"
        )

    required_tx = {
        "txn_id",
        "split",
    }

    missing_tx = (
        required_tx - set(transactions.columns)
    )

    if missing_tx:
        raise ValueError(
            "transactions missing columns: "
            f"{sorted(missing_tx)}"
        )

    if pair_features["txn_id"].duplicated().any():
        raise ValueError(
            "pair_features txn_id must be unique"
        )

    if transactions["txn_id"].duplicated().any():
        raise ValueError(
            "transactions txn_id must be unique"
        )

    split_lookup = transactions[
        ["txn_id", "split"]
    ].copy()

    output = pair_features.merge(
        split_lookup,
        on="txn_id",
        how="left",
        validate="one_to_one",
    )

    if output["split"].isna().any():
        raise ValueError(
            "some pair feature txn_ids are missing "
            "from transactions"
        )

    invalid_splits = (
        set(output["split"]) - set(SPLITS)
    )

    if invalid_splits:
        raise ValueError(
            f"unknown split labels: {sorted(invalid_splits)}"
        )

    output = output.loc[
        output["split"] != "warmup"
    ].copy()

    return output[PAIR_OUTPUT_COLS]


def write_graph_pair_txn(
    pair_features: pd.DataFrame,
    transactions: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Prepare, validate and write graph_pair_txn.parquet."""

    output = prepare_graph_pair_txn(
        pair_features,
        transactions,
    )

    validate_graph_pair_txn_output(output)

    path = _write_validated_parquet(
        output,
        path,
        PAIR_OUTPUT_COLS,
    )

    written = pd.read_parquet(path)

    validate_graph_pair_txn_output(written)

    return path
#Encoding UTF8