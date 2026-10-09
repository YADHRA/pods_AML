from fastapi import APIRouter, HTTPException

from backend.app.schemas.api import TransactionResponse
from backend.app.services.transaction_service import get_transaction


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