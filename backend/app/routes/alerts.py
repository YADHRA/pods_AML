from fastapi import APIRouter, HTTPException, Query

from app.schemas.api import AlertsResponse
from app.services.alerts_service import get_account_alerts, get_alerts

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("", response_model=AlertsResponse)
def list_alerts(
    split: str = Query(default="test"),
    model: str = Query(default="m2"),
    variant: str = Query(default="full"),
    risk_level: str | None = Query(default=None),
    payment_format: str | None = Query(default=None),
    search: str | None = Query(default=None),
    sort_by: str = Query(default="score"),
    sort_order: str = Query(default="desc"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    try:
        return get_alerts(
            split=split,
            model=model,
            variant=variant,
            risk_level=risk_level,
            payment_format=payment_format,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@router.get("/accounts")
def list_account_alerts(
    split: str = Query(default="test"),
    model: str = Query(default="m2"),
    risk_level: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    try:
        return get_account_alerts(
            split=split,
            model=model,
            risk_level=risk_level,
            search=search,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
