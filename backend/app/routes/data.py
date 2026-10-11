import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/data", tags=["Data"])

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
EDA_DIR = RESULTS_DIR / "eda"

@router.get("/daily")
def get_daily():
    path = EDA_DIR / "daily.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="daily.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/splits")
def get_splits():
    path = EDA_DIR / "splits.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="splits.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/quality")
def get_quality():
    path = EDA_DIR / "data_quality.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="data_quality.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
