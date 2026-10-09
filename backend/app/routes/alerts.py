
from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.api import AlertsResponse
from backend.app.services.alerts_service import get_alerts

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("", response_model=AlertsResponse)
def list_alerts(
    split: str = Query(default="test"),
    model: str = Query(default="m2"),
    variant: str = Query(default="full"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    try:
        return get_alerts(
            split=split,
            model=model,
            variant=variant,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

