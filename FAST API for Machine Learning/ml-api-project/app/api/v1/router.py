# app/api/v1/router.py

from fastapi import APIRouter
from app.api.v1.endpoints import health, model_info, predict

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(health.router)
v1_router.include_router(model_info.router)
v1_router.include_router(predict.router)