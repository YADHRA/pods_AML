from fastapi import FastAPI

from backend.app.routes.accounts import router as account_router
from backend.app.routes.dashboard import router as dashboard_router
from backend.app.routes.transactions import router as transaction_router
from backend.app.routes.graph import router as graph_router
from backend.app.routes.models import router as model_router
from backend.app.routes.alerts import router as alerts_router



app = FastAPI(
    title="PODS AML API",
    description="Graph-Based Anti-Money Laundering System using XGBoost",
    version="1.0.0",
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "PODS AML API",
        "message": "Backend is running",
    }


app.include_router(dashboard_router)
app.include_router(transaction_router)
app.include_router(account_router)
app.include_router(graph_router)
app.include_router(model_router)
app.include_router(alerts_router)