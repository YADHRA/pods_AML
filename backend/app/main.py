
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.accounts import router as account_router
from app.routes.dashboard import router as dashboard_router
from app.routes.transactions import router as transaction_router
from app.routes.graph import router as graph_router
from app.routes.models import router as model_router
from app.routes.alerts import router as alerts_router

app = FastAPI(
    title="PODS AML API",
    description="Graph-Based Anti-Money Laundering System using XGBoost",
    version="1.0.0",
)

# Allow the local React/Vite frontend to access the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
        "http://localhost:5177",
        "http://127.0.0.1:5177",
        "http://localhost:5178",
        "http://127.0.0.1:5178",
        
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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