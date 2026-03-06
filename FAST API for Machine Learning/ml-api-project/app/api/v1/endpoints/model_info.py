# app/api/v1/endpoints/model_info.py

from fastapi import APIRouter, Depends
from app.schemas.response_schemas import ModelInfoResponse
from app.services.tumor_classifier import tumor_classifier
from app.core.security import require_api_key

router = APIRouter(prefix="/model", tags=["Model"])

@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Get model information",
    description="Endpoint to retrieve information about the loaded model, including version and device"
)
def get_model_info(api_key: dict = Depends(require_api_key)) -> ModelInfoResponse:
    info = tumor_classifier.get_model_info() if tumor_classifier.is_loaded else {}
    return ModelInfoResponse(
        success=True,
        message="Model information retrieved successfully" if info else "No model loaded",
        data=info
    )