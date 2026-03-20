from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from app.api.schemas import AuthRequest, BatchPredictRequest, PredictRequest, SimulationRequest, TrainRequest
from app.api.service import PlatformService
from app.configs.settings import settings
from app.security.auth import authenticate_user, create_access_token, require_roles
from app.security.rate_limit import RateLimitMiddleware


app = FastAPI(title="Barings Fraud & OpRisk Platform", version="0.1.0")
app.add_middleware(RateLimitMiddleware, limit_per_minute=settings.rate_limit_per_minute)
service = PlatformService()


@app.post("/auth/token")
def issue_token(payload: AuthRequest):
    user = authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_access_token(user), "token_type": "bearer", "role": user["role"]}


@app.get("/health")
def health():
    return service.health()


@app.get("/metrics")
def metrics():
    return service.metrics()


@app.get("/sources")
def sources():
    return service.sources()


@app.get("/timeline")
def timeline():
    return service.timeline()


@app.post("/simulate")
def simulate(payload: SimulationRequest, user=Depends(require_roles("admin", "analyst"))):
    return service.simulate(payload, user)


@app.post("/train")
def train(payload: TrainRequest, user=Depends(require_roles("admin"))):
    return service.train(payload, user)


@app.post("/predict")
def predict(payload: PredictRequest, user=Depends(require_roles("admin", "analyst"))):
    return service.predict(payload, user)


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictRequest, user=Depends(require_roles("admin", "analyst"))):
    return service.predict_batch(payload.items, user)


@app.get("/explain/{alert_id}")
def explain(alert_id: str, user=Depends(require_roles("admin", "analyst"))):
    try:
        return service.explain(alert_id, user)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Alert not found") from exc


@app.get("/audit/{object_id}")
def audit(object_id: str, user=Depends(require_roles("admin", "analyst"))):
    return service.audit(object_id, user)

