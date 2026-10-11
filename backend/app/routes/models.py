
from fastapi import APIRouter, HTTPException

from app.schemas.api import PredictionResponse
from app.services.model_service import get_evaluation, get_prediction


router = APIRouter(
    prefix="/api/models",
    tags=["Models"],
)


@router.get("/evaluation")
def model_evaluation():
    return get_evaluation()


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