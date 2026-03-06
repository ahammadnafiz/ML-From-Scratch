# 🐳 Docker & CI/CD for ML Engineers
## Learn by Doing — From Zero to Automated Deployments

> **Context:** You finished your FastAPI brain tumor API. Now you'll learn to containerize it properly and automate every test + deployment with GitHub Actions. Everything here is grounded in your actual project.

---

## Table of Contents

**Docker**
1. [What Docker Actually Is — The Mental Model](#1-what-docker-actually-is)
2. [Images vs Containers — The Core Distinction](#2-images-vs-containers)
3. [Writing a Dockerfile — Line by Line](#3-writing-a-dockerfile)
4. [Layer Caching — Why Order Matters](#4-layer-caching)
5. [Docker Compose — Running Multiple Services](#5-docker-compose)
6. [Volumes & Environment Variables](#6-volumes--environment-variables)
7. [Multi-Stage Builds — Smaller Production Images](#7-multi-stage-builds)
8. [Essential Docker Commands](#8-essential-docker-commands)

**CI/CD**

9. [What CI/CD Is and Why You Need It](#9-what-cicd-is-and-why-you-need-it)
10. [GitHub Actions — The Mental Model](#10-github-actions--the-mental-model)
11. [Your First Workflow — Run Tests on Every Push](#11-your-first-workflow)
12. [Build & Push Docker Image to Registry](#12-build--push-docker-image)
13. [Deploy on Merge to Main](#13-deploy-on-merge-to-main)
14. [Secrets — Never Hardcode Credentials](#14-secrets)
15. [Full Pipeline — The Complete Picture](#15-full-pipeline)

---

## 1. What Docker Actually Is

### The Problem It Solves

You've experienced this. You train a model on your laptop, it works. You push code to a server, it crashes. Error: wrong Python version, missing `libGL`, different `torch` version.

This is the **"works on my machine"** problem. The root cause is that software depends on its environment — the OS, system libraries, Python version, installed packages. Different machines have different environments.

### The Solution: Package the Environment Itself

Docker packages not just your code, but the **entire environment** your code needs to run:
- The OS base (Ubuntu 22.04)
- System libraries (libGL, libglib)
- Python version (3.11 exactly)
- All pip packages (exact versions)
- Your application code

This bundle is called an **image**. When you run it, it becomes a **container** — an isolated process that thinks it has its own OS, its own filesystem, its own network.

### The Mental Model

Think of it like this:

```
Without Docker:
  Your code runs on the host OS directly
  Host OS has whatever packages are installed
  Different machines = different results = bugs

With Docker:
  Your code runs inside a container
  Container has exactly the packages you specified
  Same container on any machine = identical results
```

Docker is not a virtual machine. A VM emulates entire hardware including a full OS kernel. Docker **shares the host OS kernel** but isolates everything above it. This makes containers start in milliseconds and use megabytes of memory, not minutes and gigabytes like VMs.

```
Virtual Machine:              Docker Container:
┌─────────────────┐           ┌─────────────────┐
│  Your App       │           │  Your App       │
│  Python 3.11    │           │  Python 3.11    │
│  Ubuntu 22.04   │           │  Ubuntu 22.04   │
│  Guest OS Kernel│           ├─────────────────┤
│  Hypervisor     │           │  Host OS Kernel │  ← shared
│  Host OS        │           │  Host Hardware  │
└─────────────────┘           └─────────────────┘
~2GB, starts in 60s           ~150MB, starts in <1s
```

---

## 2. Images vs Containers

This distinction trips up beginners constantly. Get it clear now.

**Image** — a read-only template. A blueprint. Like a class definition in Python. It sits on disk. You can have it without running it.

**Container** — a running instance of an image. Like an object instantiated from a class. It has a process ID, it uses CPU/memory, it can write to its own filesystem.

```
Image (on disk, read-only)     Container (running, writable)
brain-tumor-api:latest    →    container_abc123  (running)
                          →    container_def456  (running)
                          →    container_ghi789  (stopped)
```

You can run many containers from the same image. They're isolated — what happens in one container doesn't affect another.

**The lifecycle:**

```
Dockerfile
    ↓  docker build
Image
    ↓  docker run
Container (running)
    ↓  docker stop
Container (stopped)
    ↓  docker rm
(gone)
```

**Registry** — a remote storage for images. Like GitHub, but for Docker images. Docker Hub is the public one. AWS ECR, GCP Artifact Registry, and GitHub Container Registry (ghcr.io) are private options used in production.

```
Your machine          Registry              Server
Build image  →  push to registry  →  pull from registry  →  run container
```

---

## 3. Writing a Dockerfile — Line by Line

The `Dockerfile` is a script that tells Docker how to build your image. Each instruction creates a new layer.

Let's write the one for your brain tumor API and understand every single line:

```dockerfile
# ── INSTRUCTION 1: FROM ───────────────────────────────────────────────────────
# Every Dockerfile starts with FROM — the base image.
# This is the foundation everything else builds on.
#
# python:3.11-slim means:
#   - Official Python image (maintained by Python core team)
#   - Python version 3.11
#   - "slim" variant: stripped-down Debian, ~150MB vs ~900MB for full
#
# WHY slim and not alpine?
#   Alpine uses musl libc instead of glibc. Many Python packages (especially
#   PyTorch, Pillow) have binary wheels compiled for glibc. On Alpine you'd
#   have to compile from source — takes 20+ minutes and often fails.
#   Slim uses glibc, so binary wheels work perfectly.
FROM python:3.11-slim

# ── INSTRUCTION 2: WORKDIR ────────────────────────────────────────────────────
# Sets the working directory inside the container.
# All subsequent commands run from this directory.
# If it doesn't exist, Docker creates it.
#
# WHY /app?
#   Convention. Could be anything. /app is clean and universally understood.
#   Avoids running as root in the filesystem root (/).
WORKDIR /app

# ── INSTRUCTION 3: ENV ────────────────────────────────────────────────────────
# Set environment variables that persist inside the container.
#
# PYTHONDONTWRITEBYTECODE=1
#   Python normally creates .pyc cache files. Inside a container there's no
#   point — the container is ephemeral. Skip writing them → smaller image.
#
# PYTHONUNBUFFERED=1
#   Python buffers stdout/stderr by default. In a container, this means logs
#   might never appear because they sit in the buffer.
#   Setting this forces immediate output — critical for seeing logs in real time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── INSTRUCTION 4: RUN (system dependencies) ──────────────────────────────────
# RUN executes a shell command inside the image during build.
# Each RUN creates a new layer.
#
# WHY these packages?
#   libgl1-mesa-glx: OpenGL library. Pillow needs this to open certain image formats.
#   libglib2.0-0: GLib. Also needed by some image processing libraries.
#
# WHY chain with && into ONE RUN?
#   Each RUN = one layer. More layers = larger image, slower builds.
#   Chaining into one RUN = one layer.
#   See Section 4 for why this matters.
#
# WHY --no-install-recommends?
#   apt normally installs "recommended" packages (docs, extras, etc.).
#   We only want what we explicitly asked for.
#
# WHY rm -rf /var/lib/apt/lists/* at the end?
#   apt downloads package lists (~50MB) to /var/lib/apt/lists/.
#   After installing, we don't need them. Delete them to shrink the image.
#   CRITICAL: This must be in the SAME RUN command as apt-get install,
#   otherwise the lists are already committed to the layer and you can't
#   remove them in a later layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── INSTRUCTION 5: COPY requirements.txt (BEFORE code) ───────────────────────
# Copy ONLY the requirements file first — not the entire codebase.
#
# WHY? Layer caching (see Section 4).
# Short answer: if you copy all code first, then requirements.txt,
# every code change invalidates the pip install cache.
# Copying requirements.txt alone means pip install is only re-run
# when requirements.txt changes — not when you fix a typo in a comment.
COPY requirements.txt .

# ── INSTRUCTION 6: RUN pip install ────────────────────────────────────────────
# Install Python dependencies.
#
# --no-cache-dir: pip caches downloaded packages by default.
#   Inside a Docker build, you'll never reuse this cache across builds.
#   Skipping it saves ~100-200MB in the final image.
#
# --upgrade pip: Ensure we have the latest pip before installing.
#   Old pip can fail to resolve modern package metadata.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── INSTRUCTION 7: COPY application code ──────────────────────────────────────
# NOW copy the actual application code.
# This layer changes on every code change — and that's fine, because
# the expensive pip install layer above is already cached.
COPY app/ ./app/

# ── INSTRUCTION 8: Create non-root user ───────────────────────────────────────
# By default, Docker containers run as root.
# Running as root is a security risk — if someone exploits your app,
# they have root access inside the container (and potentially the host).
#
# Create a user with no password, no home dir, minimal privileges.
# Then transfer ownership of /app to that user.
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app

# Switch to the non-root user for all subsequent commands
USER appuser

# ── INSTRUCTION 9: EXPOSE ─────────────────────────────────────────────────────
# Documents which port the container listens on.
# This is DOCUMENTATION only — it doesn't actually open the port.
# The actual port binding happens at runtime: docker run -p 8000:8000
EXPOSE 8000

# ── INSTRUCTION 10: HEALTHCHECK ───────────────────────────────────────────────
# Tells Docker how to check if the container is healthy.
# Docker runs this command periodically.
# If it fails 3 times in a row → container marked "unhealthy"
# → load balancer stops sending it traffic
# → Kubernetes restarts it
#
# interval: check every 30 seconds
# timeout: fail if no response in 10 seconds
# start-period: wait 60s before starting health checks (model load time)
# retries: mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

# ── INSTRUCTION 11: CMD ───────────────────────────────────────────────────────
# The default command to run when the container starts.
# Unlike RUN (which runs during build), CMD runs when you `docker run`.
#
# Use JSON array format (exec form), not string form.
# String form: CMD "uvicorn app.main:app ..."
#   → runs through /bin/sh -c, signals don't reach uvicorn properly
# Exec form: CMD ["uvicorn", ...]
#   → uvicorn runs directly as PID 1, receives signals correctly
#   → SIGTERM from `docker stop` reaches uvicorn → graceful shutdown
#
# --workers 1: One process. If you need more throughput, run more containers
#   (horizontal scaling) rather than multiple workers in one container.
#   Multiple workers in one container means multiple model copies in memory.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
```

### Build and Run It

```bash
# Build the image
# -t = tag (name:version)
# . = build context (send this directory to Docker daemon)
docker build -t brain-tumor-api:latest .

# Run it
# -p 8000:8000 = host_port:container_port
# -d = detached (background)
# --name = give container a memorable name
docker run -d -p 8000:8000 --name tumor-api brain-tumor-api:latest

# Test it
curl http://localhost:8000/api/v1/health/

# See logs
docker logs tumor-api
docker logs -f tumor-api   # -f = follow, stream new logs

# Stop it
docker stop tumor-api

# Remove the stopped container
docker rm tumor-api
```

---

## 4. Layer Caching

This is the most important performance concept in Docker. Understanding it makes builds go from 5 minutes to 5 seconds.

### How Layers Work

Every instruction in a Dockerfile creates a layer. A layer is a diff — the changes made to the filesystem by that instruction.

```
FROM python:3.11-slim          → Layer 1: base OS + Python
RUN apt-get install ...        → Layer 2: system packages added
COPY requirements.txt .        → Layer 3: requirements.txt added
RUN pip install ...            → Layer 4: Python packages installed
COPY app/ ./app/               → Layer 5: application code added
```

Docker caches each layer. On rebuild, if a layer's input hasn't changed, Docker reuses the cached version — skips re-running that instruction entirely.

### The Cache Invalidation Rule

**If any layer changes, all subsequent layers are invalidated and rebuilt.**

This is why order matters enormously:

```dockerfile
# ❌ BAD ORDER — slow builds
COPY . .                    # Copy everything (code + requirements)
RUN pip install -r requirements.txt  # Install deps
# Problem: every time you change one line of code,
# COPY . . changes → pip install runs again → 2-3 minute wait

# ✅ GOOD ORDER — fast builds
COPY requirements.txt .      # Copy only requirements
RUN pip install -r requirements.txt  # Install deps (cached unless requirements change)
COPY app/ ./app/             # Copy code last
# Now: changing app code only rebuilds the last COPY layer → 2 seconds
```

### Visualizing Cache Behavior

```
First build (no cache):
Layer 1: FROM       → RUN (build)     ✓ cached for next time
Layer 2: apt-get    → RUN (build)     ✓ cached
Layer 3: req.txt    → RUN (build)     ✓ cached
Layer 4: pip        → RUN (build)     ✓ cached ← expensive!
Layer 5: app code   → RUN (build)     ✓ cached

You fix a bug in predict.py, rebuild:
Layer 1: FROM       → CACHE HIT ⚡
Layer 2: apt-get    → CACHE HIT ⚡
Layer 3: req.txt    → CACHE HIT ⚡
Layer 4: pip        → CACHE HIT ⚡ (requirements didn't change)
Layer 5: app code   → CHANGED → rebuild (fast, just copying files)

Total time: ~3 seconds instead of 3 minutes.
```

---

## 5. Docker Compose

Your brain tumor API is one service. But in production you'll often run multiple services together: your API + a database + Redis + a monitoring agent. Managing these individually with `docker run` becomes unwieldy.

Docker Compose defines all services, their relationships, and configuration in one `docker-compose.yml` file. One command starts everything.

```yaml
# docker-compose.yml
# version: "3.9" is the Compose file format version

version: "3.9"

services:

  # ── Your API ───────────────────────────────────────────────────────────────
  api:
    # build: tells Compose to build from local Dockerfile
    # (alternative: image: ghcr.io/yourname/brain-tumor-api:latest
    #  to pull from a registry instead of building)
    build:
      context: .
      dockerfile: Dockerfile
    
    # Port mapping: host:container
    ports:
      - "8000:8000"
    
    # Load env vars from .env file
    # These override defaults in config.py
    env_file:
      - .env
    
    # Override specific env vars (take priority over .env)
    environment:
      - APP_ENV=production
    
    # Volumes: mount host directories into the container
    # ./logs on host ↔ /app/logs in container
    # WHY? Logs written inside the container would disappear when container
    # restarts. Mounting to host makes them persistent.
    volumes:
      - ./logs:/app/logs
    
    # This service depends on redis being healthy before starting
    # Without this, api might start before redis is ready
    depends_on:
      redis:
        condition: service_healthy
    
    # Restart policy
    # unless-stopped: restart if it crashes, but not if you manually stop it
    restart: unless-stopped
    
    # Health check (same as in Dockerfile, but overridable here)
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # ── Redis (for future caching) ─────────────────────────────────────────────
  redis:
    # Use official Redis image — no need to build our own
    image: redis:7-alpine
    
    # Alpine variant: ~30MB vs ~100MB for full Redis image
    ports:
      - "6379:6379"    # Only needed for debugging from host
    
    # Named volume for persistence
    # Without this, Redis data vanishes when container restarts
    volumes:
      - redis_data:/data
    
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    
    restart: unless-stopped

# Named volumes are managed by Docker (not tied to a specific host path)
# They persist across container restarts and removals
volumes:
  redis_data:
```

### Docker Compose Commands

```bash
# Start all services (build if needed), detached
docker compose up -d

# Start and rebuild images (use after code changes)
docker compose up -d --build

# See all running services and their health
docker compose ps

# Stream logs from all services
docker compose logs -f

# Stream logs from one service only
docker compose logs -f api

# Stop all services (containers stopped but not removed)
docker compose stop

# Stop AND remove containers, networks
docker compose down

# Stop AND remove containers + volumes (DESTROYS DATA)
docker compose down -v

# Run a one-off command in a service
docker compose exec api pytest tests/ -v

# Open a shell inside the api container
docker compose exec api /bin/bash

# Scale a service to N replicas (needs load balancer to be useful)
docker compose up -d --scale api=3
```

---

## 6. Volumes & Environment Variables

### Volumes — Data That Survives Container Restarts

Containers have ephemeral filesystems. Anything written inside a container disappears when it's removed. Volumes solve this.

**Three types:**

**Bind mount** — mount a specific host directory into the container:
```yaml
volumes:
  - ./logs:/app/logs          # host path : container path
  - ./app/ml_models:/app/app/ml_models   # model weights from host
```
Good for: development (live code editing), log persistence.

**Named volume** — Docker manages the storage location:
```yaml
volumes:
  - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:   # declare it
```
Good for: databases, persistent state you don't need to browse directly.

**tmpfs mount** — in-memory, very fast, no persistence:
```yaml
tmpfs:
  - /tmp
```
Good for: temporary files during processing.

### Environment Variables — The Right Way

**Rule: never hardcode secrets or configuration in your Dockerfile or code.**

Three ways to pass env vars, in order of preference:

**1. `.env` file (development):**
```bash
# .env (never commit this to git)
API_KEYS=sk-tumor-dev-abc123,sk-tumor-prod-xyz789
MODEL_PATH=app/ml_models/best.pth
APP_ENV=development
```

```yaml
# docker-compose.yml
services:
  api:
    env_file:
      - .env
```

**2. `environment:` in docker-compose (for non-secret config):**
```yaml
environment:
  - APP_ENV=production
  - LOG_LEVEL=INFO
  # Don't put secrets here — they're visible in docker inspect
```

**3. Secret management systems (production):**
AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault. Your app fetches secrets at runtime from these services — they're never in any file. This is the production standard.

---

## 7. Multi-Stage Builds

Here's a pattern that dramatically shrinks production images. Important for ML because PyTorch alone is ~800MB.

The idea: use one "builder" stage with all build tools, then copy only the final artifacts into a lean "production" stage.

```dockerfile
# ── STAGE 1: Builder ──────────────────────────────────────────────────────────
# This stage has all build tools. It will NOT be in the final image.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies (gcc, etc.) needed to compile some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages into a specific directory
# --prefix=/install means packages go to /install, not system Python
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── STAGE 2: Production ───────────────────────────────────────────────────────
# This is the final image. Starts fresh from slim base.
# The build tools from Stage 1 are gone.
FROM python:3.11-slim AS production

WORKDIR /app

# Only copy the installed packages from the builder, not gcc/g++
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/

# System runtime deps (not build deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Result: the final image contains only runtime dependencies — no gcc, no build headers, no pip cache. Typically 20-40% smaller.

---

## 8. Essential Docker Commands

```bash
# ── Images ───────────────────────────────────────────────────────────────────
docker images                          # List all local images
docker pull python:3.11-slim           # Download image from registry
docker rmi brain-tumor-api:latest      # Remove an image
docker image prune                     # Remove all dangling images

# ── Containers ───────────────────────────────────────────────────────────────
docker ps                              # List running containers
docker ps -a                           # List all containers (incl. stopped)
docker stop <name/id>                  # Graceful stop (sends SIGTERM)
docker kill <name/id>                  # Force stop (sends SIGKILL)
docker rm <name/id>                    # Remove stopped container
docker rm -f <name/id>                 # Force remove running container

# ── Debugging ─────────────────────────────────────────────────────────────────
docker logs <name>                     # Print container logs
docker logs -f <name>                  # Stream logs (follow)
docker logs --tail 100 <name>          # Last 100 lines
docker exec -it <name> /bin/bash       # Interactive shell inside container
docker inspect <name>                  # Full config + state as JSON
docker stats                           # Live CPU/memory usage for all containers

# ── Build ─────────────────────────────────────────────────────────────────────
docker build -t name:tag .             # Build from Dockerfile in current dir
docker build -t name:tag -f path/to/Dockerfile .    # Custom Dockerfile path
docker build --no-cache -t name:tag .  # Force rebuild, ignore cache

# ── Registry ──────────────────────────────────────────────────────────────────
docker login ghcr.io                   # Login to GitHub Container Registry
docker tag brain-tumor-api:latest ghcr.io/yourname/brain-tumor-api:latest
docker push ghcr.io/yourname/brain-tumor-api:latest
docker pull ghcr.io/yourname/brain-tumor-api:latest

# ── Cleanup ───────────────────────────────────────────────────────────────────
docker system prune                    # Remove stopped containers, unused images
docker system prune -a                 # Remove everything not currently used
docker volume prune                    # Remove unused volumes
```

---

## 9. What CI/CD Is and Why You Need It

### The Problem

Right now, your workflow is:
1. Write code on laptop
2. Manually run `pytest` (sometimes)
3. Manually build Docker image
4. SSH into server and manually deploy

This works for one person. It breaks down immediately when:
- You have a team — someone merges broken code
- You forget to run tests before deploying
- You need to deploy 10 times a day
- You deploy on a Friday and break production

### CI — Continuous Integration

**CI** means: every time code is pushed to the repository, an automated system runs your tests. If tests fail, the push is flagged. Nothing broken ever reaches main branch.

The key word is **continuous** — not "run tests when I remember to."

### CD — Continuous Delivery / Deployment

**Continuous Delivery**: every code change that passes tests is automatically packaged and ready to deploy with one click.

**Continuous Deployment**: every code change that passes tests is automatically deployed to production — no human approval needed.

Most ML teams use Continuous Delivery (not full Deployment) because models need human review before going live.

### The Golden Rule

```
If it can be automated, it must be automated.
If humans do it manually, humans will eventually forget or make mistakes.
```

---

## 10. GitHub Actions — The Mental Model

GitHub Actions is a CI/CD platform built into GitHub. You define workflows in YAML files inside `.github/workflows/`. When certain events happen (push, PR, schedule), GitHub runs your workflow on their servers.

### Core Concepts

**Workflow** — the whole CI/CD pipeline. Defined in a YAML file. Triggered by events.

**Event** — what triggers the workflow. `push`, `pull_request`, `schedule`, `workflow_dispatch` (manual trigger).

**Job** — a group of steps. Jobs run in parallel by default (or sequentially if you define dependencies). Each job gets a fresh virtual machine.

**Step** — a single task inside a job. Either a shell command (`run:`) or a pre-built action (`uses:`).

**Action** — a reusable step, like a plugin. `actions/checkout@v4` is the action that clones your repository into the job's VM. The entire GitHub Marketplace has thousands of community actions.

**Runner** — the VM that executes the job. `ubuntu-latest` is a GitHub-hosted Ubuntu machine, free for public repos, 2000 minutes/month free for private.

```
Workflow (ci.yml)
    ├── triggered by: push to any branch
    │
    ├── Job: test
    │   ├── Step: checkout code
    │   ├── Step: setup Python 3.11
    │   ├── Step: pip install -r requirements.txt
    │   └── Step: pytest tests/ -v
    │
    └── Job: build-and-push (only if test passes, only on main branch)
        ├── Step: checkout code
        ├── Step: login to container registry
        ├── Step: build Docker image
        └── Step: push to registry
```

### The YAML Structure

```yaml
name: My Workflow           # Name shown in GitHub UI

on:                         # Trigger events
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

jobs:
  job-name:                 # Arbitrary job identifier
    runs-on: ubuntu-latest  # Runner type
    
    steps:
      - name: Step description     # Shown in GitHub UI
        uses: actions/checkout@v4  # Use a pre-built action
      
      - name: Another step
        run: |                     # Run shell commands
          echo "hello"
          pytest tests/
        
      - name: Step with env vars
        run: pytest tests/
        env:
          API_KEYS: ${{ secrets.API_KEYS }}   # Inject from secrets
```

---

## 11. Your First Workflow — Run Tests on Every Push

Create `.github/workflows/ci.yml`:

```yaml
# .github/workflows/ci.yml

name: CI — Test & Lint

# When does this run?
on:
  push:
    branches: ["*"]           # Every push to any branch
  pull_request:
    branches: [main]          # Every PR targeting main

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest

    steps:
      # Step 1: Check out your code
      # Without this, the runner has an empty VM with no files
      - name: Checkout code
        uses: actions/checkout@v4

      # Step 2: Set up Python
      # This installs the exact Python version, adds it to PATH
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          # Cache pip packages between runs
          # If requirements.txt hasn't changed, pip install skips downloading
          cache: "pip"
          cache-dependency-path: requirements.txt

      # Step 3: Install dependencies
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt

      # Step 4: Create required directories and mock model file
      # Tests need these to exist. The model file is mocked — tests use
      # TestClient which triggers lifespan, which tries to load the model.
      # We create a fake .pth that will fail gracefully (health = degraded)
      # but won't crash the test setup.
      - name: Prepare test environment
        run: |
          mkdir -p app/ml_models logs
          # Create a dummy model file so the path exists
          # Real model tests would download from S3/GCS
          python -c "import torch; from torchvision import models; import torch.nn as nn; m = models.efficientnet_v2_s(weights=None); m.classifier[1] = nn.Linear(1280, 4); torch.save(m.state_dict(), 'app/ml_models/best.pth')"
        
      # Step 5: Set environment variables for tests
      # Use GitHub's env file to set env vars for subsequent steps
      - name: Set test environment variables
        run: |
          echo "API_KEYS=sk-tumor-test-abc123" >> $GITHUB_ENV
          echo "MODEL_PATH=app/ml_models/best.pth" >> $GITHUB_ENV
          echo "APP_ENV=testing" >> $GITHUB_ENV

      # Step 6: Run tests with coverage
      - name: Run pytest
        run: |
          pytest tests/ -v --tb=short
        
      # Step 7: Upload test results (visible in GitHub UI)
      # Even if tests fail, upload the results so you can see what failed
      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()     # Run this step even if previous steps failed
        with:
          name: test-results
          path: |
            .pytest_cache/
          retention-days: 7
```

### How to Read This in GitHub

After pushing this file:
1. Go to your repo on GitHub
2. Click the **Actions** tab
3. You'll see your workflow running (yellow spinner = in progress, green check = passed, red X = failed)
4. Click a run → click a job → see each step's output

If a test fails, you'll see exactly which test, with the full error output — before anything gets deployed.

---

## 12. Build & Push Docker Image to Registry

This workflow builds your Docker image and pushes it to GitHub Container Registry (ghcr.io). Triggered only on pushes to `main`.

Create `.github/workflows/docker.yml`:

```yaml
# .github/workflows/docker.yml

name: Build & Push Docker Image

on:
  push:
    branches: [main]    # Only build production image when merging to main
  workflow_dispatch:    # Allow manual trigger from GitHub UI

jobs:
  build-and-push:
    name: Build Docker Image
    runs-on: ubuntu-latest
    
    # Permissions needed to push to GitHub Container Registry
    permissions:
      contents: read
      packages: write    # Required to push to ghcr.io

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      # Extract metadata for tagging the image
      # This action automatically generates sensible tags:
      #   - branch name: brain-tumor-api:main
      #   - commit SHA: brain-tumor-api:sha-a1b2c3d
      #   - semver tags if you push git tags: brain-tumor-api:1.0.0
      - name: Extract Docker metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository_owner }}/brain-tumor-api
          tags: |
            type=ref,event=branch
            type=sha,prefix=sha-
            type=raw,value=latest,enable={{is_default_branch}}

      # Set up Docker Buildx — advanced builder with better caching support
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      # Log in to GitHub Container Registry
      # GITHUB_TOKEN is automatically provided — no setup needed
      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      # Build and push the image
      # cache-from/cache-to: use GitHub Actions cache to speed up builds
      # If requirements.txt didn't change, the pip install layer is cached
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha        # Pull cache from GitHub Actions cache
          cache-to: type=gha,mode=max # Push cache to GitHub Actions cache
          
          # Build arguments (can be passed to Dockerfile ARG instructions)
          build-args: |
            BUILD_DATE=${{ github.event.head_commit.timestamp }}
            GIT_SHA=${{ github.sha }}
```

After this runs, your image is available at:
```
ghcr.io/yourusername/brain-tumor-api:latest
ghcr.io/yourusername/brain-tumor-api:sha-a1b2c3d
```

Anyone with access can pull it: `docker pull ghcr.io/yourusername/brain-tumor-api:latest`

---

## 13. Deploy on Merge to Main

Now the full loop: code merged → tests pass → image built → deployed to server.

This example deploys to any Linux server via SSH. Same pattern works for AWS EC2, GCP VM, DigitalOcean droplet.

Create `.github/workflows/deploy.yml`:

```yaml
# .github/workflows/deploy.yml

name: Deploy to Production

on:
  # Only run after the docker workflow succeeds
  # This creates a dependency chain: CI → Docker → Deploy
  workflow_run:
    workflows: ["Build & Push Docker Image"]
    types: [completed]
    branches: [main]

jobs:
  deploy:
    name: Deploy to Server
    runs-on: ubuntu-latest
    
    # Only deploy if the docker build succeeded (not if it failed)
    if: ${{ github.event.workflow_run.conclusion == 'success' }}

    steps:
      - name: Deploy via SSH
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.SERVER_HOST }}        # Your server IP
          username: ${{ secrets.SERVER_USER }}    # SSH username (e.g., ubuntu)
          key: ${{ secrets.SERVER_SSH_KEY }}      # Private SSH key
          
          script: |
            # Pull the latest image
            docker pull ghcr.io/${{ github.repository_owner }}/brain-tumor-api:latest
            
            # Stop and remove old container
            docker stop tumor-api || true    # || true: don't fail if not running
            docker rm tumor-api || true
            
            # Start new container
            docker run -d \
              --name tumor-api \
              --restart unless-stopped \
              -p 8000:8000 \
              --env-file /home/ubuntu/brain-tumor-api/.env \
              -v /home/ubuntu/brain-tumor-api/logs:/app/logs \
              ghcr.io/${{ github.repository_owner }}/brain-tumor-api:latest
            
            # Wait for health check to pass
            sleep 10
            curl -f http://localhost:8000/api/v1/health/ || exit 1
            
            # Clean up old images
            docker image prune -f
            
            echo "✅ Deployment successful"
```

### Setting Up Secrets in GitHub

Go to your repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret Name | Value |
|-------------|-------|
| `SERVER_HOST` | Your server's IP address |
| `SERVER_USER` | SSH username (e.g., `ubuntu`) |
| `SERVER_SSH_KEY` | Contents of your `~/.ssh/id_rsa` private key |
| `API_KEYS` | Your production API keys |

**Never put these in your code or YAML files.** Always reference as `${{ secrets.SECRET_NAME }}`.

---

## 14. Secrets — Never Hardcode Credentials

This deserves its own section because it's the most common security mistake.

### What Counts as a Secret

- API keys (`sk-tumor-prod-xyz789`)
- Database passwords
- SSH private keys
- JWT secret keys
- Cloud provider credentials (AWS_ACCESS_KEY_ID)

### The Rules

**Rule 1:** Secrets never appear in code. Not even in comments.

**Rule 2:** Secrets never appear in Docker images. The `docker history` command shows every layer — if you put a secret in a `RUN` command, it's visible even if you delete it in a later layer.

**Rule 3:** `.env` files are never committed. Add to `.gitignore` immediately:
```bash
echo ".env" >> .gitignore
echo "*.pth" >> .gitignore   # Model weights are large, not secrets but shouldn't be in git
```

**Rule 4:** In workflows, always use `${{ secrets.SECRET_NAME }}`. GitHub automatically masks these values in logs — they show as `***`.

### Secret Rotation

Secrets should be rotated (replaced with new values) periodically. If a key is ever leaked:
1. Immediately revoke/delete the old key
2. Generate a new key
3. Update it in GitHub Secrets (or your secret manager)
4. Redeploy

---

## 15. Full Pipeline — The Complete Picture

Here's how everything connects for your brain tumor API:

### The Three Workflow Files

```
.github/
└── workflows/
    ├── ci.yml         ← runs on every push, every branch
    ├── docker.yml     ← runs on push to main only
    └── deploy.yml     ← runs after docker.yml succeeds
```

### The Full Flow

```
You write code and push to feature branch
            ↓
ci.yml triggers
  → checkout code
  → install dependencies
  → build test model weights
  → run pytest
  → ✅ or ❌ (reported on the commit in GitHub)

You open a Pull Request to main
            ↓
ci.yml triggers again on PR
  → same tests run on the PR
  → GitHub blocks merge if tests fail
  → Code review happens
  → PR merged to main

Push to main detected
            ↓
docker.yml triggers
  → build Docker image
  → push to ghcr.io with tags: latest, sha-abc123

docker.yml succeeds
            ↓
deploy.yml triggers
  → SSH into production server
  → pull new image
  → stop old container
  → start new container
  → verify health check
  → ✅ deployed
```

### The Full YAML for a Combined Workflow (Simpler for Solo Projects)

For a solo project, you can merge everything into one file with job dependencies:

```yaml
# .github/workflows/pipeline.yml

name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:

  # ── Job 1: Test ─────────────────────────────────────────────────────────────
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Prepare model for tests
        run: |
          mkdir -p app/ml_models logs
          python -c "
          import torch, torch.nn as nn
          from torchvision import models
          m = models.efficientnet_v2_s(weights=None)
          m.classifier[1] = nn.Linear(1280, 4)
          torch.save(m.state_dict(), 'app/ml_models/best.pth')
          "
      
      - name: Run tests
        env:
          API_KEYS: sk-tumor-test-abc123
          MODEL_PATH: app/ml_models/best.pth
        run: pytest tests/ -v --tb=short

  # ── Job 2: Build & Push ──────────────────────────────────────────────────────
  # needs: [test] means this job only starts if 'test' job passes
  # Only runs on pushes to main, not on PRs
  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [test]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/brain-tumor-api:latest
            ghcr.io/${{ github.repository_owner }}/brain-tumor-api:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ── Job 3: Deploy ────────────────────────────────────────────────────────────
  deploy:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build]     # Only after build succeeds
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    # Environments gate deployment behind approvals (optional, great for teams)
    environment: production
    
    steps:
      - name: Deploy via SSH
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SERVER_SSH_KEY }}
          script: |
            docker pull ghcr.io/${{ github.repository_owner }}/brain-tumor-api:latest
            docker stop tumor-api || true
            docker rm tumor-api || true
            docker run -d \
              --name tumor-api \
              --restart unless-stopped \
              -p 8000:8000 \
              --env-file /home/ubuntu/.env \
              -v /home/ubuntu/logs:/app/logs \
              ghcr.io/${{ github.repository_owner }}/brain-tumor-api:latest
            sleep 15
            curl -f http://localhost:8000/api/v1/health/
            docker image prune -f
```

### What You Now Have

| Manual Before | Automated Now |
|--------------|---------------|
| `pytest tests/` on laptop (sometimes) | Runs on every push automatically |
| `docker build` manually | Triggered by merge to main |
| SSH to server and run commands | Fully automated deploy |
| "Did I run tests before pushing?" | Impossible to merge without tests passing |
| Unknown if deploy broke production | Health check verified after every deploy |

---

## Summary — The Mental Map

**Docker** answers: "How do I make my app run identically everywhere?"
- Dockerfile = recipe for building the environment
- Image = the built, immutable environment
- Container = a running instance of that image
- Compose = orchestrate multiple containers together
- Registry = store and share images

**CI/CD** answers: "How do I ensure quality and ship fast without breaking things?"
- CI = every push triggers automated tests — broken code is caught immediately
- CD = every passing build on main is automatically deployed
- GitHub Actions = the engine that runs all of this on GitHub's servers
- Secrets = credentials injected at runtime, never stored in code

**Together:** you write code, push to GitHub, and the system handles testing, building, and deploying. You focus on the model and the API. The pipeline handles everything else.

---

*Docker is how you stop saying "it works on my machine." CI/CD is how you stop saying "I forgot to run tests before deploying." Together they're the foundation of every serious engineering team.*
