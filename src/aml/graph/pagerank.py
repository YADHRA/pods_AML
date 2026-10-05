
"""Historical PageRank features for the AML graph."""

import networkx as nx
import numpy as np
import pandas as pd


PAGERANK_FEATURE_COLS = [
    "g_pagerank_log",
]


def build_pagerank_features(
    historical_edges: pd.DataFrame,
    node_ids: pd.Series | list[int],
) -> pd.DataFrame:
    """
    Build historical directed, unweighted PageRank features.

    The input historical_edges MUST already represent only edges
    available strictly before the feature date.

    Self-loops are rejected because graph topology excludes them.

    Every canonical node is returned, including isolated nodes.

    PageRank configuration:
        - directed graph
        - unweighted edges
        - alpha = 0.85
        - uniform personalization
        - deterministic node set from node_ids

    Feature:
        g_pagerank_log
            log1p(PageRank)

    Args:
        historical_edges:
            Historical graph edges. Expected columns:
            src_node, dst_node, first_day.

        node_ids:
            Canonical node IDs from node_map.parquet.

    Returns:
        DataFrame with:
            node_id
            g_pagerank_log
    """

    # ---------------------------------------------------------
    # 1. Validate required edge columns
    # ---------------------------------------------------------
    required = {
        "src_node",
        "dst_node",
        "first_day",
    }

    missing = required - set(historical_edges.columns)

    if missing:
        raise ValueError(
            f"historical_edges missing columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # 2. Validate canonical node IDs
    # ---------------------------------------------------------
    node_ids = pd.Series(node_ids, dtype="int32")

    if node_ids.duplicated().any():
        raise ValueError("node_ids must be unique")

    nodes = pd.DataFrame({"node_id": node_ids})

    canonical_nodes = set(node_ids.tolist())

    # ---------------------------------------------------------
    # 3. Validate historical edge table
    # ---------------------------------------------------------
    if historical_edges.duplicated(
        ["src_node", "dst_node"]
    ).any():
        raise ValueError(
            "historical_edges must contain unique directed edges"
        )

    # Self-loops must never enter graph topology.
    if (
        historical_edges["src_node"]
        == historical_edges["dst_node"]
    ).any():
        raise ValueError(
            "historical_edges must not contain self-loops"
        )

    # Every graph endpoint must belong to the canonical node map.
    edge_nodes = set(
        historical_edges["src_node"]
    ).union(
        set(historical_edges["dst_node"])
    )

    unknown_nodes = edge_nodes - canonical_nodes

    if unknown_nodes:
        sample = sorted(unknown_nodes)[:10]

        raise ValueError(
            "historical_edges contains node IDs not present "
            f"in node_ids: {sample}"
        )

    # ---------------------------------------------------------
    # 4. Empty historical graph
    # ---------------------------------------------------------
    if historical_edges.empty:
        # With no edges, every node has equal PageRank:
        # 1 / number_of_nodes.
        n_nodes = len(nodes)

        if n_nodes == 0:
            return pd.DataFrame(
                columns=["node_id"] + PAGERANK_FEATURE_COLS
            )

        pagerank_value = 1.0 / n_nodes

        result = nodes.copy()

        result["g_pagerank_log"] = np.log1p(
            pagerank_value
        )

        return result[
            ["node_id"] + PAGERANK_FEATURE_COLS
        ]

    # ---------------------------------------------------------
    # 5. Build directed graph
    # ---------------------------------------------------------
    graph = nx.DiGraph()

    # Add every canonical node first so isolated nodes are kept.
    graph.add_nodes_from(node_ids.tolist())

    # Add historical edges.
    #
    # No transaction counts are attached as weights.
    # Therefore PageRank is explicitly unweighted.
    graph.add_edges_from(
        historical_edges[
            ["src_node", "dst_node"]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    # ---------------------------------------------------------
    # 6. Calculate PageRank
    # ---------------------------------------------------------
    #
    # weight=None means every edge has equal weight.
    #
    # alpha=0.85 is the standard damping value.
    #
    # A uniform personalization distribution is used by default,
    # including for dangling nodes.
    pagerank = nx.pagerank(
        graph,
        alpha=0.85,
        max_iter=200,
        tol=1e-10,
        weight=None,
    )

    # ---------------------------------------------------------
    # 7. Convert PageRank to log-transformed feature
    # ---------------------------------------------------------
    result = nodes.copy()

    result["g_pagerank"] = (
        result["node_id"]
        .map(pagerank)
        .astype("float64")
    )

    if result["g_pagerank"].isna().any():
        raise ValueError(
            "PageRank missing for one or more canonical nodes"
        )

    result["g_pagerank_log"] = np.log1p(
        result["g_pagerank"]
    )

    # ---------------------------------------------------------
    # 8. Return contract columns only
    # ---------------------------------------------------------
    return result[
        ["node_id"] + PAGERANK_FEATURE_COLS
    ]

