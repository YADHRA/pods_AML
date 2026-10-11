
from fastapi import APIRouter, HTTPException, Query

from app.schemas.api import PredictionResponse
from app.services.model_service import (
    get_comparison,
    get_curves,
    get_evaluation,
    get_prediction,
    get_shap,
)


router = APIRouter(
    prefix="/api/models",
    tags=["Models"],
)


@router.get("/evaluation")
def model_evaluation():
    return get_evaluation()


@router.get("/curves")
def model_curves(
    model: str = Query(default="m2", description="Model: m1 or m2"),
    split: str = Query(default="test", description="Split: test or val"),
):
    try:
        return get_curves(model=model, split=split)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/shap")
def model_shap(
    model: str = Query(default="m2", description="Model: m1 or m2"),
):
    try:
        return get_shap(model=model)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/comparison")
def model_comparison():
    try:
        return get_comparison()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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