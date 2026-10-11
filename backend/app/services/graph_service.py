
from collections import defaultdict
from pathlib import Path

import pandas as pd

from aml.graph.edges import build_historical_edges
from aml.graph.snapshots import build_snapshot_edges


# ============================================================
# 1. FILE PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRANSACTIONS_FILE = DATA_DIR / "transactions_clean.parquet"
NODE_MAP_FILE = DATA_DIR / "node_map.parquet"

PREDICTIONS_FILE = (
    PROJECT_ROOT / "results" / "predictions_seed42.parquet"
)

DEFAULT_MAX_NODES = 60
MAX_NODES_LIMIT = 100
MAX_NEW_NODES_PER_LEVEL = 20

# Risk thresholds are display categories, not model thresholds.
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40

# Cached datasets
_HISTORICAL_EDGES: pd.DataFrame | None = None
_NODE_MAP: pd.DataFrame | None = None
_TRANSACTIONS: pd.DataFrame | None = None
_PREDICTIONS: pd.DataFrame | None = None

# Bounded per-date caches
_ACCOUNT_RISK_CACHE: dict[str, dict[int, float]] = {}
_EDGE_STATS_CACHE: dict[str, pd.DataFrame] = {}


# ============================================================
# 2. LOAD NODE MAP
# ============================================================

def _get_node_map() -> pd.DataFrame:
    """Load account IDs, bank names, and account identifiers."""

    global _NODE_MAP

    if _NODE_MAP is None:
        if not NODE_MAP_FILE.exists():
            raise FileNotFoundError(
                f"Node map file not found: {NODE_MAP_FILE}"
            )

        node_map = pd.read_parquet(
            NODE_MAP_FILE,
            columns=["node_id", "bank", "account"],
        )

        node_map["node_id"] = pd.to_numeric(
            node_map["node_id"],
            errors="coerce",
        )

        node_map = node_map.dropna(subset=["node_id"]).copy()
        node_map["node_id"] = node_map["node_id"].astype("int64")

        _NODE_MAP = (
            node_map.drop_duplicates(subset=["node_id"])
            .reset_index(drop=True)
        )

    return _NODE_MAP


# ============================================================
# 3. LOAD TRANSACTIONS
# ============================================================

def _get_transactions() -> pd.DataFrame:
    """Load transactions needed for graph and historical risk."""

    global _TRANSACTIONS

    if _TRANSACTIONS is None:
        if not TRANSACTIONS_FILE.exists():
            raise FileNotFoundError(
                f"Transactions file not found: {TRANSACTIONS_FILE}"
            )

        transactions = pd.read_parquet(
            TRANSACTIONS_FILE,
            columns=[
                "txn_id",
                "src_node",
                "dst_node",
                "date",
                "amount_paid",
                "is_self_loop",
            ],
        )

        transactions["date"] = pd.to_datetime(
            transactions["date"],
            errors="coerce",
        ).dt.normalize()

        transactions["src_node"] = pd.to_numeric(
            transactions["src_node"],
            errors="coerce",
        )

        transactions["dst_node"] = pd.to_numeric(
            transactions["dst_node"],
            errors="coerce",
        )

        transactions["amount_paid"] = pd.to_numeric(
            transactions["amount_paid"],
            errors="coerce",
        )

        transactions = transactions.dropna(
            subset=["txn_id", "src_node", "dst_node", "date"]
        ).copy()

        transactions["src_node"] = (
            transactions["src_node"].astype("int64")
        )

        transactions["dst_node"] = (
            transactions["dst_node"].astype("int64")
        )

        transactions["is_self_loop"] = (
            transactions["is_self_loop"]
            .fillna(False)
            .astype(bool)
        )

        _TRANSACTIONS = transactions

    return _TRANSACTIONS


# ============================================================
# 4. LOAD M2 MODEL PREDICTIONS
# ============================================================

def _get_predictions() -> pd.DataFrame:
    """Load genuine M2 prediction scores by transaction ID."""

    global _PREDICTIONS

    if _PREDICTIONS is None:
        if not PREDICTIONS_FILE.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {PREDICTIONS_FILE}"
            )

        predictions = pd.read_parquet(
            PREDICTIONS_FILE,
            columns=["txn_id", "score_m2_full"],
        )

        predictions["score_m2_full"] = pd.to_numeric(
            predictions["score_m2_full"],
            errors="coerce",
        )

        predictions = predictions.dropna(
            subset=["txn_id", "score_m2_full"]
        ).copy()

        # The prediction dataset should contain one row per txn_id.
        # Keep one row if duplicate IDs unexpectedly occur.
        predictions = predictions.drop_duplicates(
            subset=["txn_id"],
            keep="first",
        )

        _PREDICTIONS = predictions

    return _PREDICTIONS


# ============================================================
# 5. BUILD HISTORICAL GRAPH EDGES
# ============================================================

def _get_all_historical_edges() -> pd.DataFrame:
    """Build graph edges using the project's locked rules."""

    global _HISTORICAL_EDGES

    if _HISTORICAL_EDGES is None:
        transactions = _get_transactions()

        edge_input = transactions[
            [
                "src_node",
                "dst_node",
                "date",
                "is_self_loop",
            ]
        ]

        _HISTORICAL_EDGES = build_historical_edges(
            edge_input
        )

    return _HISTORICAL_EDGES


def _get_historical_edges(date: str) -> pd.DataFrame:
    """Return edges available strictly before the snapshot date."""

    requested_date = pd.to_datetime(date).normalize()

    return build_snapshot_edges(
        _get_all_historical_edges(),
        requested_date,
    )


# ============================================================
# 6. HISTORICAL EDGE STATISTICS
# ============================================================

def _get_edge_statistics(date: str) -> pd.DataFrame:
    """
    Aggregate real transactions strictly before the snapshot date.
    Self-loops are excluded.
    """

    if date in _EDGE_STATS_CACHE:
        return _EDGE_STATS_CACHE[date]

    requested_date = pd.to_datetime(date).normalize()
    transactions = _get_transactions()

    historical = transactions[
        (transactions["date"] < requested_date)
        & (~transactions["is_self_loop"])
    ]

    if historical.empty:
        result = pd.DataFrame(
            columns=[
                "src_node",
                "dst_node",
                "txn_count",
                "total_amount",
                "last_date",
            ]
        )
    else:
        result = (
            historical.groupby(
                ["src_node", "dst_node"],
                as_index=False,
                sort=False,
            )
            .agg(
                txn_count=("txn_id", "count"),
                total_amount=("amount_paid", "sum"),
                last_date=("date", "max"),
            )
        )

    if len(_EDGE_STATS_CACHE) >= 4:
        _EDGE_STATS_CACHE.pop(next(iter(_EDGE_STATS_CACHE)))

    _EDGE_STATS_CACHE[date] = result

    return result


# ============================================================
# 7. HISTORICAL ACCOUNT RISK
# ============================================================

def _get_account_risk(date: str) -> dict[int, float]:
    """
    Assign each account its highest available M2 score from
    transactions before the requested snapshot date.

    Only actual predictions are used. Ground-truth labels are
    not used to calculate or display risk.

    Accounts without a valid historical prediction are omitted
    from this dictionary and become 'Not scored' in the API.
    """

    if date in _ACCOUNT_RISK_CACHE:
        return _ACCOUNT_RISK_CACHE[date]

    transactions = _get_transactions()
    predictions = _get_predictions()

    requested_date = pd.to_datetime(date).normalize()

    historical = transactions[
        (transactions["date"] < requested_date)
        & (~transactions["is_self_loop"])
    ][
        ["txn_id", "src_node", "dst_node"]
    ]

    # Only transactions that have an actual model score qualify.
    scored_transactions = historical.merge(
        predictions,
        on="txn_id",
        how="inner",
        validate="one_to_one",
    )

    if scored_transactions.empty:
        account_risk = {}

    else:
        sender_scores = scored_transactions[
            ["src_node", "score_m2_full"]
        ].rename(
            columns={"src_node": "node_id"}
        )

        receiver_scores = scored_transactions[
            ["dst_node", "score_m2_full"]
        ].rename(
            columns={"dst_node": "node_id"}
        )

        account_scores = pd.concat(
            [sender_scores, receiver_scores],
            ignore_index=True,
        )

        account_scores = account_scores.dropna(
            subset=["node_id", "score_m2_full"]
        )

        # Maximum historical transaction risk score per account.
        account_scores = account_scores.groupby(
            "node_id"
        )["score_m2_full"].max()

        account_risk = {
            int(node_id): float(score)
            for node_id, score in account_scores.items()
        }

    if len(_ACCOUNT_RISK_CACHE) >= 4:
        _ACCOUNT_RISK_CACHE.pop(
            next(iter(_ACCOUNT_RISK_CACHE))
        )

    _ACCOUNT_RISK_CACHE[date] = account_risk

    return account_risk


# ============================================================
# 8. RISK LEVEL CLASSIFICATION
# ============================================================

def _get_risk_level(score: float | None) -> str:
    """Convert a model score into a frontend display category."""

    if score is None:
        return "Not scored"

    if score >= HIGH_RISK_THRESHOLD:
        return "High"

    if score >= MEDIUM_RISK_THRESHOLD:
        return "Medium"

    return "Low"


# ============================================================
# 9. BUILD UNDIRECTED ADJACENCY
# ============================================================

def _build_adjacency(
    edges: pd.DataFrame,
) -> dict[int, set[int]]:
    """Allow graph exploration in either transaction direction."""

    adjacency: dict[int, set[int]] = defaultdict(set)

    for row in edges.itertuples(index=False):
        src = int(row.src_node)
        dst = int(row.dst_node)

        if src == dst:
            continue

        adjacency[src].add(dst)
        adjacency[dst].add(src)

    return dict(adjacency)


# ============================================================
# 10. RANK NEIGHBOURS
# ============================================================

def _rank_neighbors(
    neighbors: set[int],
    current: int,
    adjacency: dict[int, set[int]],
    edge_weight: dict[tuple[int, int], int],
) -> list[int]:
    """Rank neighbours by transaction activity and connectivity."""

    def rank(node: int):
        link_count = max(
            edge_weight.get((current, node), 0),
            edge_weight.get((node, current), 0),
        )

        degree = len(adjacency.get(node, set()))

        return (-link_count, -degree, node)

    return sorted(neighbors, key=rank)


# ============================================================
# 11. SIZE-LIMITED BREADTH-FIRST NODE SELECTION
# ============================================================

def _select_limited_nodes(
    node_id: int,
    adjacency: dict[int, set[int]],
    depth: int,
    max_nodes: int,
    edge_weight: dict[tuple[int, int], int],
) -> tuple[set[int], dict[int, int], int]:
    """
    Expand the graph level by level.

    The centre is always retained. Each level adds at most
    MAX_NEW_NODES_PER_LEVEL nodes, respecting max_nodes.
    """

    distances = {node_id: 0}
    selected = {node_id}
    frontier = [node_id]

    for level in range(1, depth + 1):
        candidates = set()

        for current in frontier:
            for neighbor in adjacency.get(current, set()):
                if neighbor not in distances:
                    candidates.add(neighbor)

        if not candidates:
            break

        remaining_slots = max_nodes - len(selected)

        if remaining_slots <= 0:
            break

        level_limit = min(
            MAX_NEW_NODES_PER_LEVEL,
            remaining_slots,
        )

        best_parent = {}

        for candidate in candidates:
            parents = [
                parent
                for parent in frontier
                if candidate in adjacency.get(parent, set())
            ]

            best_parent[candidate] = max(
                (
                    edge_weight.get((parent, candidate), 0),
                    edge_weight.get((candidate, parent), 0),
                    len(adjacency.get(candidate, set())),
                )
                for parent in parents
            ) if parents else (0, 0, 0)

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -max(best_parent[candidate][0:2]),
                -best_parent[candidate][2],
                candidate,
            ),
        )

        chosen = ranked[:level_limit]

        for candidate in chosen:
            selected.add(candidate)
            distances[candidate] = level

        frontier = chosen

        if len(selected) >= max_nodes:
            break

    return selected, distances, len(distances)


# ============================================================
# 12. COUNT ALL REACHABLE NODES
# ============================================================

def _find_all_nodes_within_depth(
    node_id: int,
    adjacency: dict[int, set[int]],
    depth: int,
) -> tuple[set[int], dict[int, int]]:
    """Count reachable nodes before applying the display cap."""

    visited = {node_id}
    distances = {node_id: 0}
    frontier = {node_id}

    for level in range(1, depth + 1):
        next_frontier = set()

        for current in frontier:
            next_frontier.update(
                neighbor
                for neighbor in adjacency.get(current, set())
                if neighbor not in visited
            )

        for neighbor in next_frontier:
            distances[neighbor] = level

        visited.update(next_frontier)
        frontier = next_frontier

        if not frontier:
            break

    return visited, distances


# ============================================================
# 13. MAIN GRAPH API SERVICE
# ============================================================

def get_graph(
    node_id: int,
    date: str,
    depth: int = 1,
    max_nodes: int = DEFAULT_MAX_NODES,
):
    """
    Return a historical graph with:
      - Account and bank details
      - BFS depth
      - Historical M2 risk score and risk level
      - Real edge transaction statistics
      - Node truncation metadata
    """

    if depth < 1 or depth > 3:
        raise ValueError("depth must be between 1 and 3")

    if max_nodes < 1 or max_nodes > MAX_NODES_LIMIT:
        raise ValueError(
            f"max_nodes must be between 1 and {MAX_NODES_LIMIT}"
        )

    requested_date = pd.to_datetime(
        date,
        errors="raise",
    ).normalize()

    snapshot_date = str(requested_date.date())

    node_map = _get_node_map()

    selected_node = node_map[
        node_map["node_id"] == node_id
    ]

    if selected_node.empty:
        return None

    # Build graph topology using the historical snapshot.
    historical_edges = _get_historical_edges(snapshot_date)

    edge_stats = _get_edge_statistics(snapshot_date)

    adjacency = _build_adjacency(historical_edges)

    # Rank candidate links using real historical transaction counts.
    edge_weight = {}

    if not edge_stats.empty:
        for row in edge_stats.itertuples(index=False):
            src = int(row.src_node)
            dst = int(row.dst_node)
            count = int(row.txn_count)

            edge_weight[(src, dst)] = count
            edge_weight[(dst, src)] = count

    # Count the full reachable graph before display truncation.
    all_reachable, _ = _find_all_nodes_within_depth(
        node_id,
        adjacency,
        depth,
    )

    graph_node_ids, distances, _ = _select_limited_nodes(
        node_id=node_id,
        adjacency=adjacency,
        depth=depth,
        max_nodes=max_nodes,
        edge_weight=edge_weight,
    )

    graph_nodes = node_map[
        node_map["node_id"].isin(graph_node_ids)
    ]

    # Calculate account risk only from predictions before the date.
    account_risk = _get_account_risk(snapshot_date)

    nodes = []

    for row in graph_nodes.itertuples(index=False):
        current_id = int(row.node_id)
        score = account_risk.get(current_id)

        nodes.append(
            {
                "id": current_id,
                "bank": str(row.bank),
                "account": str(row.account),
                "is_center": current_id == node_id,
                "depth": int(distances.get(current_id, 0)),
                "risk_score": score,
                "risk_level": _get_risk_level(score),
            }
        )

    # Keep only edges whose two endpoints are displayed.
    graph_edges = historical_edges[
        historical_edges["src_node"].isin(graph_node_ids)
        & historical_edges["dst_node"].isin(graph_node_ids)
    ].copy()

    if not graph_edges.empty and not edge_stats.empty:
        graph_edges = graph_edges.merge(
            edge_stats,
            on=["src_node", "dst_node"],
            how="left",
        )
    else:
        graph_edges["txn_count"] = 0
        graph_edges["total_amount"] = 0.0
        graph_edges["last_date"] = pd.NaT

    edges = []

    for row in graph_edges.itertuples(index=False):
        txn_count = getattr(row, "txn_count", 0)
        total_amount = getattr(row, "total_amount", 0.0)
        last_date = getattr(row, "last_date", pd.NaT)

        edges.append(
            {
                "source": int(row.src_node),
                "target": int(row.dst_node),
                "txn_count": (
                    int(txn_count)
                    if pd.notna(txn_count)
                    else 0
                ),
                "total_amount": (
                    float(total_amount)
                    if pd.notna(total_amount)
                    else 0.0
                ),
                "last_date": (
                    str(pd.Timestamp(last_date).date())
                    if pd.notna(last_date)
                    else None
                ),
            }
        )

    return {
        "node_id": node_id,
        "date": snapshot_date,
        "depth": depth,
        "max_nodes": max_nodes,
        "total_nodes_found": len(all_reachable),
        "truncated": len(all_reachable) > len(graph_node_ids),
        "nodes": nodes,
        "edges": edges,
    }