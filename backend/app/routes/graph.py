
from fastapi import APIRouter, HTTPException, Query

from app.schemas.api import GraphResponse
from app.schemas.validators import validate_date
from app.services.graph_service import get_graph


router = APIRouter(
    prefix="/api/graph",
    tags=["Graph Investigation"],
)


@router.get("/{node_id}", response_model=GraphResponse)
def graph_details(
    node_id: int,
    date: str = Query(..., description="Snapshot date in YYYY-MM-DD format"),
    depth: int = Query(2, ge=1, le=3),
    max_nodes: int = Query(60, ge=1, le=100),
):
    try:
        validated_date = validate_date(date)
        graph = get_graph(
            node_id=node_id,
            date=validated_date,
            depth=depth,
            max_nodes=max_nodes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if graph is None:
        raise HTTPException(
            status_code=404,
            detail=f"Graph node {node_id} not found",
        )

    return graph