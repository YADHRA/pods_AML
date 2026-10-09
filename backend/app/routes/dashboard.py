from fastapi import APIRouter

from backend.app.schemas.api import DashboardResponse
from backend.app.services.dashboard_service import get_dashboard_summary


router = APIRouter(
    prefix="/api/dashboard",
    tags=["Dashboard"],
)


@router.get("/summary", response_model=DashboardResponse)
def dashboard_summary():
    return get_dashboard_summary()