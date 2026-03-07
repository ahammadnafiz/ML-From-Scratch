# FastAPI — The Senior Engineer's Playbook
> From "it works" to "it scales" — everything a new API engineer needs to think like a senior

---

## Table of Contents

1. [What FastAPI Actually Is](#1-what-fastapi-actually-is)
2. [Project Directory Structure](#2-project-directory-structure)
3. [The Request Lifecycle](#3-the-request-lifecycle)
4. [Routing — The Right Way](#4-routing--the-right-way)
5. [Pydantic Models — Your Contract](#5-pydantic-models--your-contract)
6. [Dependency Injection — The Senior's Secret Weapon](#6-dependency-injection--the-seniors-secret-weapon)
7. [Database Patterns (Async)](#7-database-patterns-async)
8. [Authentication & Authorization](#8-authentication--authorization)
9. [Error Handling — Be Explicit, Be Kind](#9-error-handling--be-explicit-be-kind)
10. [Background Tasks & Celery](#10-background-tasks--celery)
11. [Caching with Redis](#11-caching-with-redis)
12. [Middleware — Cross-Cutting Concerns](#12-middleware--cross-cutting-concerns)
13. [Configuration Management](#13-configuration-management)
14. [Logging & Observability](#14-logging--observability)
15. [Testing Strategy](#15-testing-strategy)
16. [ML Model Serving Patterns](#16-ml-model-serving-patterns)
17. [Docker & Deployment](#17-docker--deployment)
18. [Senior Engineer's Checklist](#18-senior-engineers-checklist)
19. [Common Beginner Mistakes](#19-common-beginner-mistakes)

---

## 1. What FastAPI Actually Is

FastAPI is an **ASGI** (Asynchronous Server Gateway Interface) web framework. Unlike Flask (WSGI), it is built for async-first Python. The two critical things to understand:

- **It generates OpenAPI docs automatically** from your type hints — no extra work needed
- **It validates all input/output through Pydantic** — if the data is wrong, it rejects it before your code even runs

```
Client → (HTTP Request) → Uvicorn (ASGI server) → FastAPI app → Your route handler → Response
```

Uvicorn is the server. FastAPI is the framework. You always run them together:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 2. Project Directory Structure

This is the structure a senior engineer would use for a real production API. **Structure is opinion made permanent** — get it right early.

### Small-to-Medium Project (Recommended for ML APIs)

```
my_api/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app factory — entrypoint
│   ├── config.py                # All settings via pydantic-settings
│   ├── dependencies.py          # Shared DI dependencies (db session, auth, etc.)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py        # Aggregates all v1 routes
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── health.py
│   │           ├── users.py
│   │           ├── inference.py # ML inference endpoints
│   │           └── jobs.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py          # JWT, password hashing
│   │   ├── logging.py           # Structured logging setup
│   │   └── exceptions.py        # Custom exception classes
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── session.py           # Async SQLAlchemy engine + session factory
│   │   ├── base.py              # Base model class
│   │   └── migrations/          # Alembic folder
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py              # SQLAlchemy ORM models (DB tables)
│   │   └── job.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py              # Pydantic request/response schemas
│   │   ├── job.py
│   │   └── inference.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── user_service.py      # Business logic lives here — NOT in endpoints
│   │   ├── inference_service.py # Model loading, prediction logic
│   │   └── cache_service.py     # Redis operations
│   │
│   └── workers/
│       ├── __init__.py
│       └── tasks.py             # Celery tasks
│
├── tests/
│   ├── conftest.py              # Shared fixtures (test client, db, etc.)
│   ├── unit/
│   │   └── test_services.py
│   └── integration/
│       └── test_endpoints.py
│
├── scripts/
│   └── seed_db.py
│
├── .env                         # Local secrets — NEVER commit
├── .env.example                 # Template — ALWAYS commit
├── pyproject.toml               # Dependencies (use uv or poetry)
├── Dockerfile
├── docker-compose.yml
└── alembic.ini
```

### Why This Structure?

| Folder | Rule | Reason |
|--------|------|--------|
| `api/endpoints/` | Only HTTP logic | No business logic, no DB calls |
| `services/` | All business logic | Testable, reusable, framework-agnostic |
| `models/` | ORM only | Database shape — not API shape |
| `schemas/` | Pydantic only | API contract — not DB shape |
| `core/` | Infrastructure | Things used everywhere |

**Golden Rule:** Your endpoints should be thin. If your route handler is more than ~20 lines, move logic to a service.

---

## 3. The Request Lifecycle

Understanding this sequence is critical. This is what happens when a request hits your API:

```
1. Request arrives at Uvicorn
2. Middleware stack runs (logging, CORS, auth check, etc.)
3. FastAPI matches URL to a route
4. Dependency Injection resolves all `Depends(...)` — bottom-up
5. Request body is parsed and validated by Pydantic
6. Your route function runs
7. Response is validated against the response_model
8. Middleware stack runs in reverse (response middleware)
9. Response sent to client
```

If **any** step fails (validation error, unresolved dependency, exception), FastAPI returns an error response automatically. This is why you spend less time writing error handling code.

---

## 4. Routing — The Right Way

### ❌ Beginner Way (everything in main.py)

```python
# main.py — DON'T DO THIS
app = FastAPI()

@app.get("/users")
def get_users(): ...

@app.post("/users")
def create_user(): ...

@app.get("/inference/predict")
def predict(): ...
```

### ✅ Senior Way (routers + versioning)

```python
# app/api/v1/endpoints/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.user import UserCreate, UserResponse
from app.services.user_service import UserService
from app.dependencies import get_db, get_current_user

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    return await UserService.get_all(db, skip=skip, limit=limit)


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    db=Depends(get_db),
):
    return await UserService.create(db, payload)
```

```python
# app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1.endpoints import users, inference, health, jobs

router = APIRouter(prefix="/api/v1")
router.include_router(health.router)
router.include_router(users.router)
router.include_router(inference.router)
router.include_router(jobs.router)
```

```python
# app/main.py — clean and minimal
from fastapi import FastAPI
from app.api.v1.router import router
from app.core.logging import setup_logging
from app.middleware import add_middlewares

def create_app() -> FastAPI:
    app = FastAPI(
        title="My ML API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    setup_logging()
    add_middlewares(app)
    app.include_router(router)
    return app

app = create_app()
```

### Route Naming Conventions

```
GET    /users          → list all users
GET    /users/{id}     → get one user
POST   /users          → create a user
PUT    /users/{id}     → full update
PATCH  /users/{id}     → partial update
DELETE /users/{id}     → delete

GET    /inference/predict    → ML prediction
POST   /jobs                 → submit async job
GET    /jobs/{id}/status     → poll job status
```

---

## 5. Pydantic Models — Your Contract

Pydantic models are **the most important concept** in FastAPI. They define what data looks like going in and out. Treat them as API contracts.

### Separate Input from Output

```python
# app/schemas/user.py
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from uuid import UUID

# What the CLIENT sends to CREATE a user
class UserCreate(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=8)

# What the CLIENT sends to UPDATE a user
class UserUpdate(BaseModel):
    name: str | None = Field(None, min_length=2, max_length=100)
    email: EmailStr | None = None

# What the API RETURNS — never expose passwords
class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}  # Allows reading from ORM objects
```

### ML-Specific Schema Pattern

```python
# app/schemas/inference.py
from pydantic import BaseModel, Field
from typing import Any

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    model_id: str = Field(default="default", description="Which model version to use")
    top_k: int = Field(default=5, ge=1, le=100)

class PredictionResult(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    request_id: str
    results: list[PredictionResult]
    model_version: str
    latency_ms: float
```

### Field Validation Tips

```python
from pydantic import BaseModel, Field, field_validator

class JobRequest(BaseModel):
    batch_size: int = Field(default=32, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    tags: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("tags")
    @classmethod
    def lowercase_tags(cls, v):
        return [tag.lower().strip() for tag in v]
```

---

## 6. Dependency Injection — The Senior's Secret Weapon

Dependency Injection (DI) is how FastAPI handles **shared resources** — database sessions, auth, config, services. Once you truly understand this, your code becomes clean automatically.

### Basic Pattern

```python
# app/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import async_session_factory
from app.core.security import decode_token

bearer_scheme = HTTPBearer()

# ---- Database ----
async def get_db() -> AsyncSession:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

# ---- Auth ----
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
):
    token = credentials.credentials
    user_id = decode_token(token)  # Raises if invalid
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user

# ---- Role check ----
def require_role(role: str):
    """Factory function — creates a dependency dynamically"""
    async def _check(current_user=Depends(get_current_user)):
        if current_user.role != role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        return current_user
    return _check
```

```python
# Usage in endpoint
@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: UUID,
    db=Depends(get_db),
    _admin=Depends(require_role("admin")),  # Only admins can delete
):
    await UserService.delete(db, user_id)
```

### Dependency Chaining

FastAPI resolves dependencies as a DAG (directed acyclic graph). Each dependency can itself depend on other dependencies. The same instance is shared within one request:

```
get_current_active_user
    └── get_current_user
            └── get_db
                    └── async_session_factory
```

This means your `db` session is created **once** per request and reused everywhere — which is exactly what you want.

---

## 7. Database Patterns (Async)

### Setup — SQLAlchemy Async Engine

```python
# app/db/session.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,          # postgresql+asyncpg://...
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,             # Auto-reconnect on stale connections
    echo=settings.DEBUG,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,          # IMPORTANT: prevents lazy-load errors after commit
)
```

```python
# app/db/base.py
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import func
from datetime import datetime
from uuid import UUID, uuid4

class Base(DeclarativeBase):
    pass

class TimestampMixin:
    """Add to every model — always track when rows were created/updated"""
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )
```

### Repository Pattern (Senior Approach)

```python
# app/services/user_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.schemas.user import UserCreate
from app.core.security import hash_password

class UserService:

    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: UUID) -> User | None:
        return await db.get(User, user_id)

    @staticmethod
    async def get_by_email(db: AsyncSession, email: str) -> User | None:
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all(db: AsyncSession, skip: int = 0, limit: int = 100) -> list[User]:
        result = await db.execute(select(User).offset(skip).limit(limit))
        return list(result.scalars().all())

    @staticmethod
    async def create(db: AsyncSession, payload: UserCreate) -> User:
        existing = await UserService.get_by_email(db, payload.email)
        if existing:
            raise ValueError("Email already registered")

        user = User(
            email=payload.email,
            name=payload.name,
            hashed_password=hash_password(payload.password),
        )
        db.add(user)
        await db.flush()   # Gets the DB-generated ID without committing
        await db.refresh(user)
        return user
```

**Why `flush()` instead of `commit()`?** The service doesn't own the transaction — the dependency (`get_db`) does. The service just does work; the dependency commits or rolls back at the end of the request.

---

## 8. Authentication & Authorization

### JWT Flow

```
1. User sends email + password → POST /auth/login
2. Server verifies credentials, creates JWT token
3. Server returns token to client
4. Client stores token, sends it in every request header:
   Authorization: Bearer <token>
5. Server verifies token signature on every protected endpoint
```

```python
# app/core/security.py
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(user_id: str, expires_minutes: int = 60) -> str:
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    payload = {"sub": user_id, "exp": expire, "type": "access"}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            raise ValueError
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
```

```python
# app/api/v1/endpoints/auth.py
@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db=Depends(get_db)):
    user = await UserService.get_by_email(db, payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    token = create_access_token(str(user.id))
    return {"access_token": token, "token_type": "bearer"}
```

---

## 9. Error Handling — Be Explicit, Be Kind

Never return generic 500 errors to clients. Define your error vocabulary clearly.

### Custom Exception Classes

```python
# app/core/exceptions.py
from fastapi import HTTPException, status

class NotFoundError(HTTPException):
    def __init__(self, resource: str, id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with id '{id}' not found",
        )

class ConflictError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)

class UnauthorizedError(HTTPException):
    def __init__(self, detail: str = "Authentication required"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)

class ForbiddenError(HTTPException):
    def __init__(self, detail: str = "Permission denied"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

class ValidationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)
```

### Global Exception Handler

```python
# app/main.py additions
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request.state.request_id},
    )
```

### Good Error Response Shape

Every error should be consistent:

```json
{
  "detail": "User with id '123' not found",
  "code": "USER_NOT_FOUND",
  "request_id": "req_abc123"
}
```

---

## 10. Background Tasks & Celery

### FastAPI Built-in Background Tasks (Simple, small jobs)

```python
from fastapi import BackgroundTasks

@router.post("/users/", response_model=UserResponse)
async def create_user(
    payload: UserCreate,
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
):
    user = await UserService.create(db, payload)
    background_tasks.add_task(send_welcome_email, user.email)  # Runs after response
    return user
```

**Use this for:** sending emails, logging, cache invalidation — things that are fast and don't need to be retried if they fail.

### Celery (Heavy, retry-able, distributed jobs)

**Use this for:** ML inference jobs, data processing, anything >100ms, anything that must be retried on failure.

```python
# app/workers/celery_app.py
from celery import Celery
from app.config import settings

celery_app = Celery(
    "worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    task_track_started=True,
    task_acks_late=True,            # Only ack after task completes (safer)
    worker_prefetch_multiplier=1,   # One task at a time per worker (fair)
)
```

```python
# app/workers/tasks.py
from app.workers.celery_app import celery_app

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_ml_inference(self, job_id: str, input_data: dict):
    try:
        # Load model, run inference
        result = model.predict(input_data)
        # Save result to DB
        return result
    except Exception as exc:
        raise self.retry(exc=exc)
```

```python
# Endpoint that submits a job
@router.post("/jobs/", response_model=JobSubmitResponse)
async def submit_job(payload: JobRequest, db=Depends(get_db)):
    job = await JobService.create(db, payload)
    run_ml_inference.delay(str(job.id), payload.model_dump())
    return {"job_id": str(job.id), "status": "queued"}

# Endpoint that polls job status
@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    task = run_ml_inference.AsyncResult(job_id)
    return {"status": task.state, "result": task.result if task.ready() else None}
```

---

## 11. Caching with Redis

```python
# app/services/cache_service.py
import redis.asyncio as redis
import json
from app.config import settings

class CacheService:
    def __init__(self):
        self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

    async def get(self, key: str) -> dict | None:
        value = await self.redis.get(key)
        return json.loads(value) if value else None

    async def set(self, key: str, value: dict, ttl_seconds: int = 300):
        await self.redis.setex(key, ttl_seconds, json.dumps(value))

    async def delete(self, key: str):
        await self.redis.delete(key)

cache = CacheService()
```

```python
# Cache-aside pattern in an endpoint
@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: UUID, db=Depends(get_db)):
    cache_key = f"user:{user_id}"

    # 1. Try cache first
    cached = await cache.get(cache_key)
    if cached:
        return cached

    # 2. Cache miss — hit DB
    user = await UserService.get_by_id(db, user_id)
    if not user:
        raise NotFoundError("User", str(user_id))

    # 3. Store in cache for next time
    await cache.set(cache_key, UserResponse.model_validate(user).model_dump(), ttl_seconds=600)
    return user
```

**Cache Invalidation Rule:** When you update or delete a user, always delete their cache key too.

```python
async def update_user(...):
    user = await UserService.update(db, user_id, payload)
    await cache.delete(f"user:{user_id}")  # Invalidate
    return user
```

---

## 12. Middleware — Cross-Cutting Concerns

Middleware runs on **every** request. Keep it fast and focused.

```python
# app/middleware.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
import logging

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return response


def add_middlewares(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourfrontend.com"],  # Never use ["*"] in production
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
```

---

## 13. Configuration Management

Never hardcode values. Never use `os.environ.get()` scattered through your code.

```python
# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # App
    APP_NAME: str = "My ML API"
    DEBUG: bool = False
    SECRET_KEY: str                  # Required — no default

    # Database
    DATABASE_URL: str                # postgresql+asyncpg://user:pass@host:5432/db

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # ML
    MODEL_PATH: str = "models/default"
    MODEL_BATCH_SIZE: int = 32
    INFERENCE_TIMEOUT_SECONDS: int = 30

    # External APIs
    OPENAI_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

@lru_cache()         # Singleton — settings loaded once, cached forever
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

```bash
# .env (never commit this)
SECRET_KEY=your-super-secret-key-min-32-chars
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/mydb
REDIS_URL=redis://localhost:6379/0
DEBUG=true
```

---

## 14. Logging & Observability

### Structured Logging (JSON in production)

```python
# app/core/logging.py
import logging
import sys
from app.config import settings

def setup_logging():
    log_format = (
        "%(asctime)s %(levelname)s %(name)s %(message)s"
        if settings.DEBUG
        else '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    )
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format=log_format,
        stream=sys.stdout,
    )
    # Silence noisy third-party loggers
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
```

### Health Check Endpoint (Required)

Every production API must have a `/health` endpoint. Load balancers and Kubernetes probe it.

```python
# app/api/v1/endpoints/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.dependencies import get_db

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "components": {
            "database": db_status,
        }
    }
```

---

## 15. Testing Strategy

### Setup — Async Test Client

```python
# tests/conftest.py
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.main import app
from app.db.base import Base
from app.dependencies import get_db

# Use a separate test database
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def db(test_engine):
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()  # Clean up after every test

@pytest_asyncio.fixture
async def client(db):
    app.dependency_overrides[get_db] = lambda: db  # Inject test DB
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
```

### Writing Tests

```python
# tests/integration/test_users.py
import pytest

@pytest.mark.asyncio
async def test_create_user(client):
    response = await client.post("/api/v1/users/", json={
        "email": "test@example.com",
        "name": "Test User",
        "password": "strongpassword123",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "password" not in data   # CRITICAL: never leak passwords

@pytest.mark.asyncio
async def test_create_user_duplicate_email(client):
    payload = {"email": "dup@example.com", "name": "User", "password": "pass12345"}
    await client.post("/api/v1/users/", json=payload)  # First creation
    response = await client.post("/api/v1/users/", json=payload)  # Duplicate
    assert response.status_code == 409
```

---

## 16. ML Model Serving Patterns

### Startup Lifecycle — Load Models Once

```python
# app/main.py
from contextlib import asynccontextmanager
from app.services.inference_service import InferenceService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    await InferenceService.load_model()
    yield
    # --- Shutdown ---
    await InferenceService.unload_model()

app = FastAPI(lifespan=lifespan)
```

```python
# app/services/inference_service.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch

class InferenceService:
    _model = None
    _executor = ThreadPoolExecutor(max_workers=4)

    @classmethod
    async def load_model(cls):
        # Run blocking model load in thread pool — don't block event loop
        loop = asyncio.get_event_loop()
        cls._model = await loop.run_in_executor(cls._executor, cls._load_blocking)

    @staticmethod
    def _load_blocking():
        # This is your heavy model loading code
        model = torch.load("models/model.pt")
        model.eval()
        return model

    @classmethod
    async def predict(cls, text: str) -> dict:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            cls._executor,
            lambda: cls._predict_blocking(text)
        )
        return result

    @classmethod
    def _predict_blocking(cls, text: str) -> dict:
        with torch.no_grad():
            # Your actual inference code
            output = cls._model(text)
        return {"label": output.label, "confidence": float(output.score)}
```

**Critical Rule:** ML inference (PyTorch, transformers) is CPU/GPU bound and **blocking**. Never run it directly in an async function. Always use `run_in_executor` to move it to a thread pool.

---

## 17. Docker & Deployment

### Dockerfile (Production-Grade)

```dockerfile
# Multi-stage build — keep final image small
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml .
RUN pip install uv && uv pip install --system -r pyproject.toml

FROM python:3.11-slim AS runtime
WORKDIR /app

# Non-root user — security best practice
RUN useradd --create-home appuser
USER appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --chown=appuser:appuser ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### docker-compose.yml (Local Dev)

```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./app:/app/app   # Hot-reload in dev

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s

  worker:
    build: .
    command: celery -A app.workers.celery_app worker --loglevel=info
    env_file: .env
    depends_on:
      - redis
      - db
```

---

## 18. Senior Engineer's Checklist

Before any code goes to production, a senior engineer mentally checks these:

### API Design
- [ ] All endpoints return consistent response shapes
- [ ] HTTP status codes are correct (201 for create, 204 for delete, 409 for conflicts)
- [ ] Pagination on all list endpoints (`skip`, `limit` or cursor-based)
- [ ] API is versioned (`/api/v1/`)
- [ ] Sensitive fields (passwords, tokens) never appear in responses
- [ ] `response_model` is set on every endpoint

### Security
- [ ] All secrets come from environment variables — zero hardcoded values
- [ ] Authentication required on all non-public endpoints
- [ ] CORS origins are explicit — never `*` in production
- [ ] Passwords hashed with bcrypt — never stored in plaintext
- [ ] Rate limiting in place (use `slowapi` or API gateway)
- [ ] Input validation via Pydantic with sensible max lengths

### Reliability
- [ ] Health check endpoint exists and checks DB connectivity
- [ ] Database connection pool configured and bounded
- [ ] All background jobs have retry logic
- [ ] Timeouts set on all external HTTP calls
- [ ] Graceful shutdown handled (lifespan context manager)

### Observability
- [ ] Structured logging on every request (method, path, status, duration)
- [ ] Request ID propagated through the entire request lifecycle
- [ ] Errors logged with full stack traces
- [ ] `/metrics` endpoint or Prometheus exporter in place

### Code Quality
- [ ] Business logic in services — not in endpoints
- [ ] No raw SQL strings — use ORM or parameterized queries
- [ ] All public functions have type hints
- [ ] Critical paths have integration tests
- [ ] `pyproject.toml` pins dependency versions

---

## 19. Common Beginner Mistakes

### ❌ Mistake 1: Blocking the event loop

```python
# BAD — time.sleep blocks the entire async event loop
@router.get("/predict")
async def predict():
    time.sleep(2)          # Blocks ALL other requests!
    return model.predict() # Blocking CPU work in async context
```

```python
# GOOD — offload blocking work to thread pool
@router.get("/predict")
async def predict():
    result = await asyncio.get_event_loop().run_in_executor(None, model.predict)
    return result
```

### ❌ Mistake 2: Creating a new DB session manually in every route

```python
# BAD — session never closed on error, no transaction management
@router.get("/users")
async def get_users():
    session = AsyncSession(engine)
    users = await session.execute(select(User))
    return users
```

```python
# GOOD — always use Depends(get_db)
@router.get("/users")
async def get_users(db=Depends(get_db)):
    result = await db.execute(select(User))
    return result.scalars().all()
```

### ❌ Mistake 3: Putting logic in routes

```python
# BAD — route does too much
@router.post("/users")
async def create_user(payload: UserCreate, db=Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar():
        raise HTTPException(409, "Email taken")
    hashed = bcrypt.hash(payload.password)
    user = User(email=payload.email, hashed_password=hashed)
    db.add(user)
    await db.commit()
    send_email(user.email)  # Even worse — blocking email in route
    return user
```

```python
# GOOD — thin route, fat service
@router.post("/users", status_code=201)
async def create_user(
    payload: UserCreate,
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
):
    user = await UserService.create(db, payload)
    background_tasks.add_task(send_welcome_email, user.email)
    return user
```

### ❌ Mistake 4: No response_model (data leaks)

```python
# BAD — might accidentally return hashed_password
@router.get("/users/{id}")
async def get_user(id: UUID, db=Depends(get_db)):
    return await db.get(User, id)
```

```python
# GOOD — response filtered through Pydantic schema
@router.get("/users/{id}", response_model=UserResponse)
async def get_user(id: UUID, db=Depends(get_db)):
    return await db.get(User, id)
```

### ❌ Mistake 5: Hardcoded config

```python
# BAD
DATABASE_URL = "postgresql://admin:password123@localhost/mydb"
SECRET_KEY = "mysecret"
```

```python
# GOOD
from app.config import settings
# Use settings.DATABASE_URL, settings.SECRET_KEY everywhere
```

---

## Quick Reference — HTTP Status Codes

| Code | Meaning | When to Use |
|------|---------|-------------|
| 200 | OK | Successful GET, PATCH |
| 201 | Created | Successful POST |
| 204 | No Content | Successful DELETE |
| 400 | Bad Request | Malformed request syntax |
| 401 | Unauthorized | Not authenticated |
| 403 | Forbidden | Authenticated but no permission |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Duplicate (e.g. email already exists) |
| 422 | Unprocessable Entity | Pydantic validation failed |
| 429 | Too Many Requests | Rate limit hit |
| 500 | Internal Server Error | Unexpected crash |
| 503 | Service Unavailable | Dependency down (DB, model server) |

---

## Quick Reference — When to Use What

| Need | Solution |
|------|----------|
| Shared resource per request | `Depends()` |
| Config values | `pydantic-settings` + `.env` |
| Input/output validation | Pydantic schemas |
| DB session | `async_sessionmaker` via `Depends(get_db)` |
| Heavy blocking work | `run_in_executor` |
| Simple async background job | `BackgroundTasks` |
| Heavy/retryable background job | Celery + Redis |
| Hot data / repeated reads | Redis cache-aside |
| Cross-request logic | Middleware |
| Protect a route | `Depends(get_current_user)` |
| Role-based access | `Depends(require_role("admin"))` |

---

*Study this structure. When you write code, ask: "Where does this belong?" — and the answer will almost always be clear. Endpoints are thin. Services are fat. Dependencies handle the plumbing.*
