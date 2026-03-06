# app/api/v1/endpoints/health.py

import time
from fastapi import APIRouter
from app.schemas.response_schemas import HealthResponse, HealthData
from app.services.tumor_classifier import tumor_classifier

router = APIRouter(prefix="/health", tags=["Health"])

_start_time = time.time()

@router.get(
    "/",
    response_model=HealthResponse,
    summary="Check API health status",
    description="Endpoint to check the health status of the API and model readiness"
)
def health_check() -> HealthResponse:
    model_info = tumor_classifier.get_model_info() if tumor_classifier.is_loaded else None
    return HealthResponse(
        success=True,
        message="API is healthy",
        data=HealthData(
            status="healthy" if tumor_classifier.is_loaded else "model not loaded",
            model_loaded=tumor_classifier.is_loaded,
            model_version=model_info.get("model_version") if model_info else None,
            device=model_info.get("device") if model_info else "unknown",
            uptime_seconds=round(time.time() - _start_time, 2)

        )    
    )  