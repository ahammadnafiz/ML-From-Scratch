# app/schemas/response_schemas.py

from pydantic import BaseModel, Field
from typing import Any, Optional

class BaseResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the request was successful")
    message: str = Field(..., description="A human-readable message providing more details about the response")

class TumorProbabilities(BaseModel):
    glioma: float = Field(..., ge=0.0, le=1.0, description="Probability of glioma")
    meningioma: float = Field(..., ge=0.0, le=1.0, description="Probability of meningioma")
    notumor: float = Field(..., ge=0.0, le=1.0, description="Probability of no tumor detected")
    pituitary: float = Field(..., ge=0.0, le=1.0, description="Probability of pituitary tumor")

class PredictionResult(BaseModel):
    predicted_class: str = Field(..., description="The class with the highest predicted probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the predicted class")
    probabilities: TumorProbabilities = Field(..., description="Probabilities for each tumor class")
    description: str = Field(..., description="Description of the predicted class")
    inference_time_ms: float = Field(..., ge=0.0, description="Time taken for inference in milliseconds")
    model_version: str = Field(..., description="Version of the model used for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": "glioma",
                "confidence": 0.9423,
                "probabilities": {
                    "glioma": 0.9423,
                    "meningioma": 0.0312,
                    "notumor": 0.0189,
                    "pituitary": 0.0076,
                },
                "description": "Tumor arising from glial cells in the brain or spinal cord",
                "inference_time_ms": 42.3,
                "model_version": "1.0.0",
            }
        }


class PredictionResponse(BaseResponse):
    data: PredictionResult = Field(..., description="The result of the tumor classification prediction")

class BatchPredictionItem(BaseModel):
    """Single item in a batch prediction response."""
    filename: str = Field(..., description="Name of the input image file")
    result: Optional[PredictionResult] = Field(None, description="The result of the tumor classification prediction for this file")
    error: Optional[str] = Field(None, description="Error message if prediction failed for this file")


class BatchPredictionData(BaseModel):
    total: int = Field(..., description="Total number of images in the batch")
    successful: int = Field(..., description="Number of successful predictions")
    failed: int = Field(..., description="Number of failed predictions")
    results: list[BatchPredictionItem] = Field(..., description="Per-file prediction results or errors")


class BatchPredictionResponse(BaseResponse):
    data: BatchPredictionData = Field(..., description="Batch prediction summary and per-file results")

class ModelInfoResponse(BaseResponse):
    data: dict[str, Any] = Field(..., description="Information about the model, such as name, version, and path")

class HealthData(BaseModel):
    status: str = Field(..., description="Health status of the API")
    model_loaded: bool = Field(..., description="Indicates if the model is loaded and ready for inference")
    model_version: Optional[str] = Field(None, description="Version of the loaded model, if available")
    device: str = Field(..., description="Device being used for inference (e.g., 'cpu' or 'cuda')")
    uptime_seconds: float = Field(..., ge=0.0, description="Uptime of the API in seconds")

class HealthResponse(BaseResponse):
    data: HealthData = Field(..., description="Health status and information about the API")

class ErrorDetail(BaseModel):
    field: Optional[str] = Field(None, description="The field that caused the error, if applicable")
    message: str = Field(..., description="A detailed error message")

class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Indicates that the request was not successful")
    error: ErrorDetail = Field(..., description="Details about the error that occurred")
    details: Optional[list[ErrorDetail]] = Field(None, description="Additional error details, if applicable")