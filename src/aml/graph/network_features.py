"""Historical network-level features for the AML graph."""

import numpy as np
import networkx as nx
import pandas as pd


NETWORK_FEATURE_COLS = [
    "g_n_unique_out_log",
    "g_n_unique_in_log",
    "g_has_sent_before",
    "g_has_received_before",
    "g_wcc_size_log",
]


def build_network_features(
    historical_edges: pd.DataFrame,
    node_ids: pd.Series | list[int],
) -> pd.DataFrame:
    """
    Build network features from a historical graph snapshot.

    The input historical_edges MUST already represent only edges
    available strictly before the feature date.

    Self-loops must be excluded from historical_edges.
    This function rejects them as a defensive safety check.

    Every node is returned, including isolated nodes.

    Features:
        g_n_unique_out_log
            log1p(number of unique historical outgoing counterparties)

        g_n_unique_in_log
            log1p(number of unique historical incoming counterparties)

        g_has_sent_before
            1 if the node has sent at least one historical transaction

        g_has_received_before
            1 if the node has received at least one historical transaction

        g_wcc_size_log
            log1p(size of the node's weakly connected component)

    Returns:
        DataFrame keyed by node_id.
    """

    # ---------------------------------------------------------
    # 1. Validate required columns
    # ---------------------------------------------------------
    required = {"src_node", "dst_node"}

    missing = required - set(historical_edges.columns)

    if missing:
        raise ValueError(
            f"historical_edges missing columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # 2. Validate node IDs
    # ---------------------------------------------------------
    node_ids = pd.Series(node_ids, dtype="int32")

    if node_ids.duplicated().any():
        raise ValueError("node_ids must be unique")

    # Keep every canonical node, including isolated nodes.
    nodes = pd.DataFrame({"node_id": node_ids})

    # ---------------------------------------------------------
    # 3. Validate historical edges
    # ---------------------------------------------------------
    if historical_edges.duplicated(
        ["src_node", "dst_node"]
    ).any():
        raise ValueError(
            "historical_edges must contain unique directed edges"
        )

    # Self-loops must never enter graph topology.
    # Step 3 already removes them, but this is an additional
    # defensive check to prevent accidental leakage into graph
    # features.
    if (
        historical_edges["src_node"]
        == historical_edges["dst_node"]
    ).any():
        raise ValueError(
            "historical_edges must not contain self-loops"
        )

    # ---------------------------------------------------------
    # 4. Handle empty historical graph
    # ---------------------------------------------------------
    if historical_edges.empty:
        result = nodes.copy()

        result["g_n_unique_out_log"] = 0.0
        result["g_n_unique_in_log"] = 0.0
        result["g_has_sent_before"] = 0
        result["g_has_received_before"] = 0

        # An isolated node forms a component of size 1.
        result["g_wcc_size_log"] = np.log1p(1)

        return result[
            ["node_id"] + NETWORK_FEATURE_COLS
        ]

    # ---------------------------------------------------------
    # 5. Copy only the topology columns
    # ---------------------------------------------------------
    edges = historical_edges[
        ["src_node", "dst_node"]
    ].copy()

    # ---------------------------------------------------------
    # 6. Unique outgoing counterparties
    # ---------------------------------------------------------
    out_counts = (
        edges.groupby("src_node")["dst_node"]
        .nunique()
        .rename("n_unique_out")
    )

    # ---------------------------------------------------------
    # 7. Unique incoming counterparties
    # ---------------------------------------------------------
    in_counts = (
        edges.groupby("dst_node")["src_node"]
        .nunique()
        .rename("n_unique_in")
    )

    # ---------------------------------------------------------
    # 8. Historical sent/received flags
    # ---------------------------------------------------------
    sent_nodes = set(edges["src_node"])
    received_nodes = set(edges["dst_node"])

    # ---------------------------------------------------------
    # 9. Weakly Connected Components
    # ---------------------------------------------------------
    # The original graph is directed, but WCC ignores direction.
    # Therefore, we use an undirected NetworkX graph here.
    graph = nx.Graph()

    # Add every canonical node first so isolated nodes are included.
    graph.add_nodes_from(node_ids.tolist())

    # Add historical non-self edges.
    graph.add_edges_from(
        edges[
            ["src_node", "dst_node"]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    # Calculate component size for every node.
    wcc_size = {}

    for component in nx.connected_components(graph):
        size = len(component)

        for node_id in component:
            wcc_size[node_id] = size

    # ---------------------------------------------------------
    # 10. Build final result
    # ---------------------------------------------------------
    result = nodes.copy()

    # Unique outgoing counterparties.
    result["n_unique_out"] = (
        result["node_id"]
        .map(out_counts)
        .fillna(0)
    )

    # Unique incoming counterparties.
    result["n_unique_in"] = (
        result["node_id"]
        .map(in_counts)
        .fillna(0)
    )

    # Log transform to reduce skew.
    result["g_n_unique_out_log"] = np.log1p(
        result["n_unique_out"]
    )

    result["g_n_unique_in_log"] = np.log1p(
        result["n_unique_in"]
    )

    # Historical sent flag.
    result["g_has_sent_before"] = (
        result["node_id"]
        .isin(sent_nodes)
        .astype("int8")
    )

    # Historical received flag.
    result["g_has_received_before"] = (
        result["node_id"]
        .isin(received_nodes)
        .astype("int8")
    )

    # Weakly connected component size.
    # If a node is not found, it is isolated → component size 1.
    result["g_wcc_size_log"] = (
        result["node_id"]
        .map(wcc_size)
        .fillna(1)
        .pipe(np.log1p)
        .astype("float64")
    )

    # ---------------------------------------------------------
    # 11. Return only contract columns
    # ---------------------------------------------------------
    return result[
        ["node_id"] + NETWORK_FEATURE_COLS
    ]
