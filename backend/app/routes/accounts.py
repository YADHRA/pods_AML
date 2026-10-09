from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.api import AccountResponse
from backend.app.schemas.validators import validate_date
from backend.app.services.account_service import get_account


router = APIRouter(
    prefix="/api/accounts",
    tags=["Accounts"],
)


@router.get("/{node_id}", response_model=AccountResponse)
def account_details(
    node_id: int,
    date: str = Query(
        ...,
        description="Historical feature date in YYYY-MM-DD format",
    ),
):
    try:
        validated_date = validate_date(date)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    account = get_account(node_id, validated_date)

    if account is None:
        raise HTTPException(
            status_code=404,
            detail=f"Account/node {node_id} not found for date {validated_date}",
        )

    return account