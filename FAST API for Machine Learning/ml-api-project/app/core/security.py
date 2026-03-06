# app/core/security.py

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from loguru import logger
from app.config import settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(api_key: str = Security(API_KEY_HEADER)) -> dict:
    valid_keys = settings.get_api_keys()

    if not api_key:
        logger.warning("Missing API key in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": "API key"},
        )
    
    if api_key not in valid_keys:
        key_prefix = api_key[:12] + "..." if len(api_key) > 12 else api_key
        logger.warning(f"Invalid API key used: {key_prefix}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "API key"},
        )
    

    return {"api_key": api_key, "key_prefix": api_key[:8] + "..."}