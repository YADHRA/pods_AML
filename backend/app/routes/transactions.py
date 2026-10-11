from fastapi import APIRouter, HTTPException, Query

from app.schemas.api import TransactionResponse
from app.services.transaction_service import get_explanation, get_transaction


router = APIRouter(
    prefix="/api/transactions",
    tags=["Transactions"],
)


@router.get("/{txn_id}", response_model=TransactionResponse)
def transaction_details(txn_id: int):
    transaction = get_transaction(txn_id)

    if transaction is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {txn_id} not found",
        )

    return transaction


@router.get("/{txn_id}/explanation")
def transaction_explanation(
    txn_id: int,
    model: str = Query(default="m2", description="Model: m1 or m2"),
):
    explanation = get_explanation(txn_id, model=model)
    return explanation
