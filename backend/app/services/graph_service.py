from pathlib import Path

import pandas as pd

from aml.graph.edges import build_historical_edges
from aml.graph.snapshots import build_snapshot_edges


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

TRANSACTIONS_FILE = DATA_DIR / "transactions_clean.parquet"
NODE_MAP_FILE = DATA_DIR / "node_map.parquet"


# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------

_HISTORICAL_EDGES: pd.DataFrame | None = None
_NODE_MAP: pd.DataFrame | None = None


def _get_node_map() -> pd.DataFrame:
    """Load and cache the canonical node map."""
    global _NODE_MAP

    if _NODE_MAP is None:
        _NODE_MAP = pd.read_parquet(
            NODE_MAP_FILE,
            columns=[
                "node_id",
                "bank",
                "account",
            ],
        )

    return _NODE_MAP


def _get_all_historical_edges() -> pd.DataFrame:
    """
    Build the historical graph once and cache it.

    Graph methodology:
    - self-loops are excluded from topology
    - one directed edge is kept per (src_node, dst_node)
    - first_day is the first observed transaction date
    """
    global _HISTORICAL_EDGES

    if _HISTORICAL_EDGES is None:
        transactions = pd.read_parquet(
            TRANSACTIONS_FILE,
            columns=[
                "src_node",
                "dst_node",
                "date",
                "is_self_loop",
            ],
        )

        transactions["date"] = pd.to_datetime(
            transactions["date"]
        ).dt.normalize()

        _HISTORICAL_EDGES = build_historical_edges(
            transactions
        )

    return _HISTORICAL_EDGES


def _get_historical_edges(date: str) -> pd.DataFrame:
    """Return graph edges available strictly before the requested date."""
    requested_date = pd.to_datetime(date).normalize()

    historical_edges = _get_all_historical_edges()

    return build_snapshot_edges(
        historical_edges,
        requested_date,
    )


def _build_adjacency(
    edges: pd.DataFrame,
) -> dict[int, set[int]]:
    """Build an undirected adjacency map for investigation traversal."""

    adjacency: dict[int, set[int]] = {}

    for row in edges.itertuples(index=False):
        src = int(row.src_node)
        dst = int(row.dst_node)

        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    return adjacency


def _find_nodes_within_depth(
    node_id: int,
    adjacency: dict[int, set[int]],
    depth: int,
) -> set[int]:
    """Return nodes within the requested BFS depth."""

    visited = {node_id}
    frontier = {node_id}

    for _ in range(depth):
        next_frontier = set()

        for current in frontier:
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    next_frontier.add(neighbor)

        visited.update(next_frontier)
        frontier = next_frontier

        if not frontier:
            break

    return visited


def get_graph(
    node_id: int,
    date: str,
    depth: int = 1,
):
    """Return a historical investigation subgraph around a node."""

    if depth < 1 or depth > 3:
        raise ValueError(
            "depth must be between 1 and 3"
        )

    requested_date = pd.to_datetime(date).normalize()

    node_map = _get_node_map()

    selected_node = node_map[
        node_map["node_id"] == node_id
    ]

    if selected_node.empty:
        return None

    historical_edges = _get_historical_edges(
        requested_date
    )

    adjacency = _build_adjacency(
        historical_edges
    )

    graph_node_ids = _find_nodes_within_depth(
        node_id,
        adjacency,
        depth,
    )

    graph_nodes = node_map[
        node_map["node_id"].isin(graph_node_ids)
    ]

    nodes = [
        {
            "id": int(row.node_id),
            "bank": str(row.bank),
            "account": str(row.account),
            "is_center": int(row.node_id) == node_id,
        }
        for row in graph_nodes.itertuples(index=False)
    ]

    graph_edges = historical_edges[
        historical_edges["src_node"].isin(graph_node_ids)
        & historical_edges["dst_node"].isin(graph_node_ids)
    ]

    edges = [
        {
            "source": int(row.src_node),
            "target": int(row.dst_node),
        }
        for row in graph_edges.itertuples(index=False)
    ]

    return {
        "node_id": node_id,
        "date": str(requested_date.date()),
        "depth": depth,
        "nodes": nodes,
        "edges": edges,
    }