# app/core/middleware.py

import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start_time = time.time()
        is_health = "/health" in request.url.path
        if not is_health:
            logger.info(
                f"→ REQUEST  | id={request_id} | "
                f"method={request.method} | "
                f"path={request.url.path} | "
                f"client={request.client.host if request.client else 'unknown'}"
            )
        response = await call_next(request)
        process_time_ms = (time.time() - start_time) * 1000
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Process-Time-ms"] = f"{process_time_ms:.1f}"

        if not is_health:
            log_fn = logger.info if response.status_code < 400 else logger.warning
            log_fn(
                f"← RESPONSE | id={request_id} | "
                f"status={response.status_code} | "
                f"time={process_time_ms:.1f}ms"
            )
        return response
