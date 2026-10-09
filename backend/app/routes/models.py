from fastapi import APIRouter, HTTPException

from backend.app.schemas.api import PredictionResponse
from backend.app.services.model_service import get_prediction


router = APIRouter(
    prefix="/api/models",
    tags=["Models"],
)


@router.get(
    "/predictions/{txn_id}",
    response_model=PredictionResponse,
)
def prediction_details(txn_id: int):
    prediction = get_prediction(txn_id)

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction for transaction {txn_id} not found",
        )

    return prediction