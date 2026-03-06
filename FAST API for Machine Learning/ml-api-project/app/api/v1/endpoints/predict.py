# app/api/v1/endpoints/predict.py

import io
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from PIL import Image
from loguru import logger

from app.schemas.response_schemas import (
    PredictionResponse,
    PredictionResult,
    TumorProbabilities,
    BatchPredictionResponse,
    BatchPredictionData,
    BatchPredictionItem,
)

from app.services.tumor_classifier import tumor_classifier
from app.core.security import require_api_key
from app.config import settings

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Allowed image MIME types
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/tiff",
}


def validate_and_open_image(file: UploadFile, file_bytes: bytes) -> Image.Image:
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=(
                f"Unsupported file type: {file.content_type}"
                f" - allowed types are: {', '.join(ALLOWED_CONTENT_TYPES)}"
            )
        )
    
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > settings.max_image_mb:
        raise HTTPException(
            status_code=400, 
            detail=(
                f"File size exceeds the maximum limit of {settings.max_image_mb} MB"
                f" - uploaded file size is {file_size_mb:.2f} MB"
            )
        )
    
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image.verify()  # Verify that it's a valid image
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")  # Reopen after verify
        
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid image file: {str(e)}")
    
    return image


def dict_to_prediction_result(result_dict: dict) -> PredictionResult:
    probs = result_dict.get("probabilities", {})
    return PredictionResult(
        predicted_class=result_dict.get("predicted_class", "unknown"),
        confidence=result_dict.get("confidence", 0.0),
        probabilities=TumorProbabilities(**probs),
        description=result_dict.get("description", ""),
        inference_time_ms=result_dict.get("inference_time_ms", 0.0),
        model_version=result_dict.get("model_version", "unknown")
    )

@router.post(
    "/tumor",
    response_model=PredictionResponse,
    summary="Classify Brain Tumor from MRI Image",
    description=(
        "Upload a single brain MRI image (JPEG or PNG) "
        "to classify it as glioma, meningioma, notumor, or pituitary."
    ),
    responses={
        200: {"description": "Successful classification"},
        401: {"description": "Invalid or missing API key"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
        422: {"description": "Invalid or corrupted image"},
        503: {"description": "Model not loaded"},
    }
)
async def predict_single(
    file: UploadFile = File(
        ...,
        description=(
            "Brain MRI image file to classify. "
            "Allowed types: JPEG, PNG, BMP, TIFF. "
            f"Max size: {settings.max_image_mb} MB."
        )
    ),
    api_key: dict = Depends(require_api_key)     
) -> PredictionResponse:
    if not tumor_classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please try again later.")

    logger.info(f"Single prediction request | file={file.filename} | type={file.content_type}")

    file_bytes = await file.read()
    image = validate_and_open_image(file, file_bytes)

    try:
        result_dict = tumor_classifier.predict(image)
    except Exception as e:
        logger.error(f"Prediction error for file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
    result = dict_to_prediction_result(result_dict)
    logger.info(
        f"Prediction successful | file={file.filename} | "
        f"class={result.predicted_class} | confidence={result.confidence:.4f}"
    )
    return PredictionResponse(
        success=True,
        message=f"Successfully classified {file.filename} as {result.predicted_class}",
        data=result
    )

@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Batch classify brain tumors from multiple MRI images",
    description=(
        "Upload multiple brain MRI images (JPEG or PNG) in a single request "
        "to classify each image as glioma, meningioma, notumor, or pituitary."
    )
)
async def predict_batch(
    files: list[UploadFile] = File(
        ...,
        description=(
            "List of brain MRI image files to classify. "
            "Allowed types: JPEG, PNG, BMP, TIFF. "
            f"Max size per file: {settings.max_image_mb} MB."
        )
    ),
    api_key: dict = Depends(require_api_key)     
) -> BatchPredictionResponse:
    if not tumor_classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please try again later.")


    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Number of files exceeds the maximum batch size of {settings.max_batch_size}"
        )
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded for batch prediction")
    
    logger.info(f"Batch prediction request | num_files={len(files)}")

    valid_pil_images: list[Image.Image] = []
    valid_indices: list[int] = []
    results: list[BatchPredictionItem] = []

    for i, file in enumerate(files):
        try:
            file_bytes = await file.read()
            image = validate_and_open_image(file, file_bytes)
            valid_pil_images.append(image)
            valid_indices.append(i)
            results.append(BatchPredictionItem(filename=file.filename or f"file_{i}"))
        except HTTPException as e:
            # Don't fail the whole batch for one bad image
            results.append(
                BatchPredictionItem(
                    filename=file.filename or f"file_{i}",
                    error=e.detail
                )
            )

    if valid_pil_images:
        try:
            batch_results = tumor_classifier.predict_batch(valid_pil_images)
            # Map results back to their original positions
            for batch_idx, original_idx in enumerate(valid_indices):
                results[original_idx].result = dict_to_prediction_result(
                    batch_results[batch_idx]
                )
        except Exception as e:
            logger.error(f"Batch inference failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")

    successful = sum(1 for r in results if r.result is not None)
    failed = len(results) - successful

    logger.info(f"Batch complete | successful={successful} | failed={failed}")

    return BatchPredictionResponse(
        success=True,
        message=f"Processed {len(files)} images: {successful} successful, {failed} failed",
        data=BatchPredictionData(
            total=len(files),
            successful=successful,
            failed=failed,
            results=results,
        ),
    )

