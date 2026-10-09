from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.api import GraphResponse
from backend.app.schemas.validators import validate_date
from backend.app.services.graph_service import get_graph


router = APIRouter(
    prefix="/api/graph",
    tags=["Graph Investigation"],
)


@router.get("/{node_id}", response_model=GraphResponse)
def graph_details(
    node_id: int,
    date: str = Query(
        ...,
        description="Historical graph date in YYYY-MM-DD format",
    ),
    depth: int = Query(
        1,
        ge=1,
        le=3,
        description="Graph investigation depth",
    ),
):
    try:
        validated_date = validate_date(date)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    try:
        graph = get_graph(
            node_id=node_id,
            date=validated_date,
            depth=depth,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if graph is None:
        raise HTTPException(
            status_code=404,
            detail=f"Graph node {node_id} not found",
        )

    return graph