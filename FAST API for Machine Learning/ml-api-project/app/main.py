# app/main.py

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.v1.router import v1_router
from app.services.tumor_classifier import tumor_classifier
from app.core.middleware import RequestLoggingMiddleware
from app.config import settings, setup_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("logs", exist_ok=True)
    setup_logging(settings.log_level)
    
    logger.info(f"Starting {settings.app_name} v{settings.app_version}...")
    logger.info(f"Environment: {settings.app_env}")
    
    try:
        tumor_classifier.load(
            model_path=settings.model_path,
            model_version=settings.model_version,
        )
        logger.success("✅ Model loaded. API is ready to serve predictions.")
    except FileNotFoundError as e:
        logger.critical(f"❌ Model load failed: {e}")
        logger.critical(f"Place model at {settings.model_path} and restart.")
        raise
    yield
    
    logger.info("Shutting down gracefully...")
    
    
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    ## 🧠 Brain Tumor Classification API
    
    Endpoints for ML inference, authentication, and health monitoring.
    """,
    lifespan=lifespan,
    contact={"name": "API Support", "email": "support@yourapi.com"},
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

# Mount API router
app.include_router(v1_router)