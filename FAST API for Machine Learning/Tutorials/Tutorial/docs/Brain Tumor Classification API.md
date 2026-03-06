# 🧠 Brain Tumor Classification API
## End-to-End FastAPI Deployment — Learn by Doing

> **How to use this guide:** Read every section top to bottom. Understand the *why* before you write the *how*. Type every code block yourself — don't copy-paste. Each section builds directly on the previous one. By the end, you'll have a production-ready medical imaging API deployed with Docker.

---

## Table of Contents

1. [The Big Picture — What We're Building](#1-the-big-picture)
2. [What You Already Have — Understanding Your Model](#2-what-you-already-have)
3. [Environment & Project Setup](#3-environment--project-setup)
4. [Config & Settings — The Right Way](#4-config--settings--the-right-way)
5. [The ML Service — Loading & Inference](#5-the-ml-service--loading--inference)
6. [Pydantic Schemas — Validating Everything](#6-pydantic-schemas--validating-everything)
7. [API Endpoints — Health, Predict, Info](#7-api-endpoints)
8. [Authentication — API Key Protection](#8-authentication--api-key-protection)
9. [Middleware — Logging, CORS, Timing](#9-middleware--logging-cors-timing)
10. [Error Handling — Never Crash Silently](#10-error-handling--never-crash-silently)
11. [Main App — Wiring It All Together](#11-main-app--wiring-it-all-together)
12. [Testing — Prove It Works](#12-testing--prove-it-works)
13. [Docker — Containerize Everything](#13-docker--containerize-everything)
14. [Running & Manual Testing](#14-running--manual-testing)
15. [Production Checklist & Next Steps](#15-production-checklist--next-steps)

---

## 1. The Big Picture

### What We're Building

A **production-ready REST API** that accepts brain MRI images and classifies them into 4 categories:
- `glioma` — a type of tumor that occurs in the brain and spinal cord
- `meningioma` — tumor in the membranes surrounding the brain
- `notumor` — healthy brain scan
- `pituitary` — tumor in the pituitary gland

### The Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              FastAPI Application         │
                         │                                         │
Client (Postman/         │  ┌──────────┐    ┌──────────────────┐  │
Frontend/Script)  ──────►│  │Middleware│───►│   API Router     │  │
                  ◄──────│  │ (Logging │    │  /api/v1/...     │  │
                         │  │  CORS    │    └────────┬─────────┘  │
                         │  │  Timing) │             │            │
                         │  └──────────┘    ┌────────▼─────────┐  │
                         │                  │  Pydantic         │  │
                         │                  │  (Validate Input) │  │
                         │                  └────────┬─────────┘  │
                         │                           │            │
                         │                  ┌────────▼─────────┐  │
                         │                  │   ML Service      │  │
                         │                  │ (EfficientNetV2)  │  │
                         │                  │  PyTorch Model    │  │
                         │                  └────────┬─────────┘  │
                         │                           │            │
                         │                  ┌────────▼─────────┐  │
                         │                  │  Pydantic         │  │
                         │                  │  (Format Output)  │  │
                         └──────────────────┴──────────────────┘  │
                                                                    │
                                                                    └
```

### Request Flow (trace this in your head every time you get confused)

```
POST /api/v1/predict/tumor  with image file
    ↓
Middleware: log request, start timer
    ↓
API Key Header → Auth check → reject if invalid
    ↓
Pydantic: validate file is an image, size ok
    ↓
ML Service: preprocess → model.forward() → softmax → class + confidence
    ↓
Pydantic: format response with prediction, confidence, all probabilities
    ↓
Middleware: log response time
    ↓
JSON response back to client
```

### Final Project Structure

```
brain-tumor-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py                      ← FastAPI app, lifespan, exception handlers
│   ├── config.py                    ← Settings via environment variables
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py            ← Combines all v1 endpoints
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── health.py        ← GET /health
│   │           ├── predict.py       ← POST /predict/tumor, POST /predict/batch
│   │           └── model_info.py    ← GET /model/info
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py              ← API key validation logic
│   │   └── middleware.py            ← Request logging, timing
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request_schemas.py       ← Input validation models
│   │   └── response_schemas.py      ← Output formatting models
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── tumor_classifier.py      ← Model loading, preprocessing, inference
│   │
│   └── ml_models/
│       └── best.pth                 ← Your trained EfficientNetV2 weights
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  ← Test fixtures
│   └── test_predict.py              ← Endpoint tests
│
├── sample_images/                   ← Test images for manual testing
│   └── .gitkeep
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 2. What You Already Have — Understanding Your Model

Before writing a single line of API code, you must deeply understand what your model does. This section explains the notebook's model so you can wrap it correctly.

### The Model: EfficientNetV2-S

Your notebook trains `torchvision.models.efficientnet_v2_s` — a state-of-the-art CNN pretrained on ImageNet, fine-tuned for brain tumor classification.

```python
# From your notebook — this is what the model IS
model = models.efficientnet_v2_s(weights='DEFAULT')
model.classifier[1] = torch.nn.Linear(1280, 4)  # 4 classes
```

**Why EfficientNetV2?** It achieves near-SOTA accuracy at a fraction of the computation. The `S` (small) variant is fast enough for real-time inference even on CPU.

**What the model expects:**
- Input tensor shape: `(batch_size, 3, 224, 224)` — batch of RGB images
- Pixel values: normalized with ImageNet stats
- Output: raw logits of shape `(batch_size, 4)` — NOT probabilities yet

**What your notebook's preprocessing does (the exact pipeline you must replicate in the API):**
```python
# From your notebook — test_transforms (NO augmentation in inference)
test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**The 4 classes and their integer IDs:**
```python
label_dict = {
    'glioma': 0,
    'meningioma': 1,
    'notumor': 2,
    'pituitary': 3
}
```

### What the Saved `.pth` File Contains

`best.pth` from your notebook contains **only the model weights** (`state_dict`), not the full model architecture. This means to load it you must:
1. Recreate the exact same model architecture in code
2. Load the weights into that architecture with `model.load_state_dict(...)`

This is why having the exact architecture code in your service is critical.

### Converting Logits to Probabilities

The model outputs raw scores (logits). To get probabilities:
```python
import torch.nn.functional as F

logits = model(image_tensor)           # Raw scores, e.g. [2.3, -0.5, 1.1, -1.2]
probs = F.softmax(logits, dim=1)       # Probabilities that sum to 1.0
confidence, class_idx = probs.max(1)   # Highest probability = prediction
```

---

## 3. Environment & Project Setup

### Step 1 — Create the Project

```bash
mkdir brain-tumor-api
cd brain-tumor-api

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Verify
which python   # Should show path inside venv/
```

### Step 2 — Install Dependencies

```bash
# Core API
pip install fastapi==0.110.0
pip install "uvicorn[standard]==0.27.1"
pip install pydantic==2.6.0
pip install pydantic-settings==2.2.0    # For config management
pip install python-multipart             # Required for file uploads

# ML / Image Processing
pip install torch torchvision            # PyTorch + torchvision
pip install Pillow==10.2.0              # Image opening

# Observability
pip install loguru==0.7.2               # Better logging than stdlib

# Testing
pip install httpx==0.27.0              # Async HTTP client for tests
pip install pytest==8.0.0
pip install pytest-asyncio==0.23.0

# Save
pip freeze > requirements.txt
```

### Step 3 — Create the Directory Structure

```bash
mkdir -p app/api/v1/endpoints
mkdir -p app/core
mkdir -p app/schemas
mkdir -p app/services
mkdir -p app/ml_models
mkdir -p tests
mkdir -p sample_images

# Create all __init__.py files
touch app/__init__.py
touch app/api/__init__.py
touch app/api/v1/__init__.py
touch app/api/v1/endpoints/__init__.py
touch app/core/__init__.py
touch app/schemas/__init__.py
touch app/services/__init__.py
touch tests/__init__.py

# Create placeholder for model weights
touch app/ml_models/.gitkeep
```

**Now copy your trained `best.pth` file into `app/ml_models/`.**

### Step 4 — Create `.env.example`

Create `.env.example` (this documents what environment variables are needed):

```bash
# .env.example
# Copy this to .env and fill in real values

# API Authentication
API_KEYS=sk-tumor-dev-abc123,sk-tumor-prod-xyz789

# App Settings
APP_ENV=development
LOG_LEVEL=INFO

# Model
MODEL_PATH=app/ml_models/best.pth
MODEL_NAME=EfficientNetV2-S-BrainTumor
MODEL_VERSION=1.0.0

# Server
HOST=0.0.0.0
PORT=8000
```

```bash
cp .env.example .env
```

---

## 4. Config & Settings — The Right Way

**Why have a config module?** Hardcoding values (API keys, paths, model names) in your code is a disaster. When you deploy to Docker or a cloud server, those hardcoded values break or become security risks. Instead, **read everything from environment variables** through a single config object.

Create `app/config.py`:

```python
# app/config.py

"""
WHY pydantic-settings?
  Regular os.getenv() gives you raw strings with no type safety.
  pydantic-settings automatically:
    - Reads from .env file
    - Type-casts values (str → list, str → int, etc.)
    - Validates required fields are present
    - Gives you IDE autocomplete

HOW it works:
  1. You define a Settings class with typed fields
  2. Pydantic reads matching env var names (case-insensitive)
  3. You get a validated, typed settings object
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from loguru import logger
import sys


class Settings(BaseSettings):
    # ── Application ───────────────────────────────────────────────
    app_name: str = "Brain Tumor Classification API"
    app_version: str = "1.0.0"
    app_env: str = Field(default="development", env="APP_ENV")
    
    # ── Model ─────────────────────────────────────────────────────
    # This path is relative to where you run uvicorn from (project root)
    model_path: str = Field(default="app/ml_models/best.pth", env="MODEL_PATH")
    model_name: str = Field(default="EfficientNetV2-S-BrainTumor", env="MODEL_NAME")
    model_version: str = Field(default="1.0.0", env="MODEL_VERSION")
    
    # ── Authentication ────────────────────────────────────────────
    # Stored as comma-separated string in env: "key1,key2,key3"
    # pydantic-settings will parse it as a list automatically
    api_keys: str = Field(
        default="sk-tumor-dev-abc123",
        env="API_KEYS",
        description="Comma-separated list of valid API keys"
    )

    # ── Inference ─────────────────────────────────────────────────
    max_image_size_mb: float = Field(default=10.0)   # Reject images larger than this
    max_batch_size: int = Field(default=8)            # Max images in one batch request

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        # Tell pydantic-settings to also look in .env file
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False      # API_KEYS and api_keys are the same

    def get_api_keys_list(self) -> list[str]:
        """Parse the comma-separated API keys string into a list."""
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]


# ── Logging Setup ─────────────────────────────────────────────────────────────
def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure loguru for structured, human-readable logs.
    
    WHY loguru over stdlib logging?
    - Zero config for sensible defaults
    - Automatic exception tracebacks
    - Colored output in terminal
    - Easy to add file sinks later
    """
    logger.remove()   # Remove default handler
    
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    
    # Also log to file (useful in production)
    logger.add(
        "logs/api.log",
        level="INFO",
        rotation="10 MB",      # New file after 10MB
        retention="7 days",    # Keep logs for 7 days
        compression="zip",     # Compress old logs
    )


# ── Singleton Settings Instance ───────────────────────────────────────────────
# We create ONE settings object imported everywhere.
# This avoids re-reading .env file on every import.
settings = Settings()
```

**Test that config works:**
```bash
python -c "from app.config import settings; print(settings.model_path)"
# Should print: app/ml_models/best.pth
```

---

## 5. The ML Service — Loading & Inference

This is the heart of your project. **The ML service has exactly one job:** take a PIL image, run the model, return a structured result. It should know nothing about HTTP, FastAPI, or requests — only about the model.

**Why separate this from the endpoint?** Separation of concerns. If you later want to swap EfficientNetV2 for a different model, you only change this file. The API layer doesn't care.

Create `app/services/tumor_classifier.py`:

```python
# app/services/tumor_classifier.py

"""
THE ML SERVICE

This module owns everything model-related:
  - Loading the .pth weights into the EfficientNetV2 architecture
  - Image preprocessing (MUST match training's test_transforms exactly)
  - Running inference (forward pass)
  - Converting logits → probabilities → structured result dict

IMPORTANT: This service is framework-agnostic.
  - No FastAPI imports here
  - No HTTP concepts (no Request, no Response)
  - Just pure ML logic, testable in isolation
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2
from PIL import Image
from loguru import logger


# ── Class Mapping ─────────────────────────────────────────────────────────────
# Must match label_dict from your notebook exactly.
# Index → class name mapping (the inverse of label_dict)
CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary",
}

# Human-readable descriptions for each class (useful for API consumers)
CLASS_DESCRIPTIONS = {
    "glioma": "Tumor arising from glial cells in the brain or spinal cord",
    "meningioma": "Tumor in the protective membranes covering the brain and spinal cord",
    "notumor": "No tumor detected — healthy brain scan",
    "pituitary": "Tumor in the pituitary gland at the base of the brain",
}


# ── Preprocessing Pipeline ────────────────────────────────────────────────────
# CRITICAL: This must exactly match test_transforms from your notebook.
# If you accidentally use train_transforms (with augmentation), predictions
# will be random and noisy.
INFERENCE_TRANSFORMS = v2.Compose([
    v2.Resize((224, 224)),                               # Fixed size for EfficientNetV2
    v2.PILToTensor(),                                    # PIL Image → torch.Tensor (C, H, W)
    v2.ToDtype(torch.float32),                           # int8 pixel values → float32
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],                      # ImageNet mean (R, G, B)
        std=[0.229, 0.224, 0.225]                        # ImageNet std  (R, G, B)
    ),
])

# WHY ImageNet normalization?
# EfficientNetV2-S was pretrained on ImageNet. Those pretrained weights
# expect inputs normalized with ImageNet's pixel distribution.
# Using different normalization = wrong activations = garbage predictions.


class TumorClassifier:
    """
    Wraps the EfficientNetV2-S model for brain tumor classification.
    
    Usage:
        classifier = TumorClassifier()
        classifier.load("app/ml_models/best.pth")
        
        from PIL import Image
        image = Image.open("scan.jpg")
        result = classifier.predict(image)
        # result = {
        #   "predicted_class": "glioma",
        #   "confidence": 0.94,
        #   "probabilities": {"glioma": 0.94, "meningioma": 0.03, ...},
        #   "inference_time_ms": 45.2,
        #   "model_version": "1.0.0"
        # }
    """

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.is_loaded: bool = False
        self.model_path: Optional[str] = None
        self.model_version: str = "unknown"
        
        logger.info(f"TumorClassifier initialized | device={self.device}")

    def _build_architecture(self) -> nn.Module:
        """
        Recreate the exact same model architecture from the notebook.
        
        WHY do this in a separate method?
        - The .pth file contains weights only, not architecture
        - You must define the same architecture before loading weights
        - If you change the architecture, old weights won't load
        """
        # Load EfficientNetV2-S without pretrained weights
        # (we'll load our fine-tuned weights from .pth)
        model = models.efficientnet_v2_s(weights=None)
        
        # Replace the final classifier — exactly as in your notebook:
        # model.classifier[1] = torch.nn.Linear(1280, class_size)
        model.classifier[1] = nn.Linear(1280, len(CLASS_NAMES))
        
        return model

    def load(self, model_path: str, model_version: str = "1.0.0") -> None:
        """
        Load the trained .pth weights into the model.
        
        WHY call this separately from __init__?
        - In FastAPI, you load the model ONCE at startup (lifespan event)
        - This gives you explicit control over when the expensive load happens
        - Makes testing easier (can create classifier without loading weights)
        
        Args:
            model_path: Path to the .pth file (best.pth from training)
            model_version: Version string to include in predictions
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{model_path}'. "
                f"Did you copy best.pth into app/ml_models/?"
            )
        
        logger.info(f"Loading model from '{model_path}'...")
        start_time = time.time()
        
        # Step 1: Build the architecture
        model = self._build_architecture()
        
        # Step 2: Load the saved weights
        # map_location=self.device handles CPU/GPU mismatch:
        # If you trained on GPU but deploy on CPU, this remaps automatically
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        
        # Step 3: Set to eval mode
        # CRITICAL: Without this, BatchNorm and Dropout behave differently
        # (training mode uses batch statistics; eval mode uses running statistics)
        # Forgetting model.eval() is one of the most common deployment bugs.
        model.eval()
        
        # Step 4: Move to device
        model = model.to(self.device)
        
        self.model = model
        self.model_path = model_path
        self.model_version = model_version
        self.is_loaded = True
        
        load_time = (time.time() - start_time) * 1000
        logger.success(
            f"Model loaded successfully | "
            f"path={model_path} | "
            f"device={self.device} | "
            f"load_time={load_time:.1f}ms"
        )

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Convert a PIL image to a model-ready tensor.
        
        Pipeline:
          PIL Image (any size, any mode)
            → convert to RGB (handles grayscale MRIs, RGBA PNGs)
            → apply INFERENCE_TRANSFORMS
            → shape: (3, 224, 224)
            → unsqueeze(0) → shape: (1, 3, 224, 224)  # Add batch dimension
            → move to device
        
        WHY .convert('RGB')?
          Medical MRI images are sometimes:
          - Grayscale (1 channel) — DICOM exports
          - RGBA (4 channels) — PNG with alpha channel
          The model expects exactly 3 channels (R, G, B).
          convert('RGB') handles all these cases safely.
        """
        image = image.convert("RGB")
        tensor = INFERENCE_TRANSFORMS(image)        # Shape: (3, 224, 224)
        tensor = tensor.unsqueeze(0)                # Shape: (1, 3, 224, 224)
        return tensor.to(self.device)

    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a single PIL Image.
        
        Returns:
            dict with keys:
              - predicted_class (str): "glioma" | "meningioma" | "notumor" | "pituitary"
              - confidence (float): 0.0 to 1.0 — probability of predicted class
              - probabilities (dict): probability for each class
              - inference_time_ms (float): how long inference took
              - model_version (str): version of the loaded model
        
        Raises:
            RuntimeError: If model hasn't been loaded yet
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model not loaded. Call classifier.load(model_path) first."
            )
        
        start_time = time.time()
        
        # Preprocess
        tensor = self._preprocess(image)
        
        # Inference — no_grad() tells PyTorch not to track gradients
        # WHY? Gradient tracking uses memory and computation we don't need
        # during inference. Always use torch.no_grad() in production.
        with torch.no_grad():
            logits = self.model(tensor)              # Shape: (1, 4) — raw scores
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)             # Shape: (1, 4) — sums to 1.0
        probs_np = probs.cpu().numpy()[0]            # Shape: (4,) as numpy array
        
        # Get the winning class
        class_idx = int(probs_np.argmax())
        confidence = float(probs_np[class_idx])
        predicted_class = CLASS_NAMES[class_idx]
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Build per-class probability dict
        probabilities = {
            CLASS_NAMES[i]: round(float(probs_np[i]), 4)
            for i in range(len(CLASS_NAMES))
        }
        
        result = {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "description": CLASS_DESCRIPTIONS[predicted_class],
            "inference_time_ms": round(inference_time_ms, 2),
            "model_version": self.model_version,
        }
        
        logger.debug(
            f"Prediction: class={predicted_class} | "
            f"confidence={confidence:.3f} | "
            f"time={inference_time_ms:.1f}ms"
        )
        
        return result

    def predict_batch(self, images: list[Image.Image]) -> list[dict]:
        """
        Run inference on multiple images at once.
        
        WHY batching?
          - Single GPU processes multiple images in parallel
          - Much faster than calling predict() in a loop
          - Modern GPUs are underutilized by single-image inference
        
        Args:
            images: List of PIL Images (max defined in config)
        
        Returns:
            List of prediction dicts (same format as predict())
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        
        start_time = time.time()
        
        # Preprocess all images and stack into a single batch tensor
        tensors = [self._preprocess(img) for img in images]  # List of (1, 3, 224, 224)
        batch_tensor = torch.cat(tensors, dim=0)              # Shape: (N, 3, 224, 224)
        
        with torch.no_grad():
            logits = self.model(batch_tensor)                 # Shape: (N, 4)
        
        probs = F.softmax(logits, dim=1)                      # Shape: (N, 4)
        probs_np = probs.cpu().numpy()                        # Shape: (N, 4) numpy
        
        total_time_ms = (time.time() - start_time) * 1000
        per_image_time_ms = total_time_ms / len(images)
        
        results = []
        for i in range(len(images)):
            class_idx = int(probs_np[i].argmax())
            confidence = float(probs_np[i][class_idx])
            predicted_class = CLASS_NAMES[class_idx]
            
            results.append({
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4),
                "probabilities": {
                    CLASS_NAMES[j]: round(float(probs_np[i][j]), 4)
                    for j in range(len(CLASS_NAMES))
                },
                "description": CLASS_DESCRIPTIONS[predicted_class],
                "inference_time_ms": round(per_image_time_ms, 2),
                "model_version": self.model_version,
            })
        
        logger.info(
            f"Batch prediction: n={len(images)} | "
            f"total={total_time_ms:.1f}ms | "
            f"per_image={per_image_time_ms:.1f}ms"
        )
        
        return results

    def get_model_info(self) -> dict:
        """Return metadata about the loaded model."""
        param_count = (
            sum(p.numel() for p in self.model.parameters())
            if self.is_loaded
            else 0
        )
        return {
            "model_name": "EfficientNetV2-S",
            "model_version": self.model_version,
            "architecture": "EfficientNetV2-S fine-tuned for brain tumor classification",
            "classes": list(CLASS_NAMES.values()),
            "num_classes": len(CLASS_NAMES),
            "input_size": [224, 224],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "device": str(self.device),
            "total_parameters": param_count,
            "is_loaded": self.is_loaded,
        }


# ── Module-level Singleton ─────────────────────────────────────────────────────
# WHY a singleton?
# Loading EfficientNetV2 takes ~1-2 seconds and uses ~80MB memory.
# You never want to load it fresh on every request.
# Create ONE instance at module level → it's loaded once at startup.
#
# The FastAPI lifespan function will call tumor_classifier.load() at startup.
# Every endpoint function that needs inference imports this instance.
tumor_classifier = TumorClassifier()
```

**Quick smoke test (run this before writing any API code):**

```python
# Run from project root: python scripts/test_model.py
# First create the scripts/ dir: mkdir scripts && touch scripts/test_model.py

from PIL import Image
import numpy as np
from app.services.tumor_classifier import tumor_classifier
from app.config import settings

# Load model
tumor_classifier.load(settings.model_path, settings.model_version)

# Create a dummy random image (224x224, RGB)
dummy_image = Image.fromarray(
    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
)

result = tumor_classifier.predict(dummy_image)
print("Prediction result:")
for k, v in result.items():
    print(f"  {k}: {v}")

print("\nModel info:")
info = tumor_classifier.get_model_info()
for k, v in info.items():
    print(f"  {k}: {v}")
```

```bash
python scripts/test_model.py
```

If this runs without error, your model loads correctly. Now we build the API layer.

---

## 6. Pydantic Schemas — Validating Everything

**Why Pydantic schemas?** They are the contract between your API and its consumers. Pydantic:
- Validates incoming data and rejects bad input with clear error messages
- Documents your API automatically (FastAPI reads these to generate Swagger docs)
- Formats your output consistently — consumers always know what shape to expect

### Response Schemas

Create `app/schemas/response_schemas.py`:

```python
# app/schemas/response_schemas.py

"""
RESPONSE SCHEMAS — What we send back to clients.

Design principle: always be consistent.
Every response should have the same top-level shape so clients
can write simple error handling:
  if response.success:
      use response.data
  else:
      show response.error
"""

from pydantic import BaseModel, Field
from typing import Any, Optional


# ── Base Response ──────────────────────────────────────────────────────────────
class BaseResponse(BaseModel):
    """
    Every API response inherits from this.
    Consistent envelope means clients don't have to guess the shape.
    """
    success: bool
    message: str


# ── Prediction Results ─────────────────────────────────────────────────────────
class TumorProbabilities(BaseModel):
    """Per-class probability breakdown."""
    glioma: float = Field(..., ge=0.0, le=1.0, description="Probability of glioma")
    meningioma: float = Field(..., ge=0.0, le=1.0, description="Probability of meningioma")
    notumor: float = Field(..., ge=0.0, le=1.0, description="Probability of no tumor")
    pituitary: float = Field(..., ge=0.0, le=1.0, description="Probability of pituitary tumor")


class PredictionResult(BaseModel):
    """A single image's classification result."""
    predicted_class: str = Field(
        ...,
        description="The predicted tumor class",
        examples=["glioma"]
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence of the predicted class (0.0 - 1.0)"
    )
    probabilities: TumorProbabilities = Field(
        ...,
        description="Probability distribution across all classes"
    )
    description: str = Field(
        ...,
        description="Human-readable description of the predicted condition"
    )
    inference_time_ms: float = Field(
        ...,
        description="Time taken to run inference in milliseconds"
    )
    model_version: str = Field(
        ...,
        description="Version of the model that made this prediction"
    )

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
    """Response envelope for single-image prediction."""
    data: PredictionResult


# ── Batch Prediction ───────────────────────────────────────────────────────────
class BatchPredictionItem(BaseModel):
    """Result for one image in a batch."""
    filename: str
    result: Optional[PredictionResult] = None
    error: Optional[str] = None   # If one image failed, don't fail the whole batch


class BatchPredictionResponse(BaseResponse):
    """Response envelope for batch prediction."""
    data: dict


# ── Model Info ─────────────────────────────────────────────────────────────────
class ModelInfoResponse(BaseResponse):
    """Response envelope for model metadata."""
    data: dict


# ── Health ─────────────────────────────────────────────────────────────────────
class HealthData(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    device: str
    uptime_seconds: float


class HealthResponse(BaseResponse):
    data: HealthData


# ── Error ──────────────────────────────────────────────────────────────────────
class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str


class ErrorResponse(BaseModel):
    """Used for error responses — no 'success' field since it's always False."""
    success: bool = False
    error: str
    details: Optional[list[ErrorDetail]] = None
```

---

## 7. API Endpoints

### 7.1 Health Endpoint

Create `app/api/v1/endpoints/health.py`:

```python
# app/api/v1/endpoints/health.py

"""
HEALTH ENDPOINT

WHY do we need a health check?
  - Docker uses it to know if the container is ready for traffic
  - Load balancers route traffic only to healthy instances
  - Monitoring systems alert if it starts returning errors
  - A health endpoint that just returns 200 is NOT enough —
    it should verify the model is actually loaded

GET /api/v1/health/
  - No auth required (monitoring systems call this constantly)
  - Returns: is model loaded, device, uptime
"""

import time
from fastapi import APIRouter
from app.schemas.response_schemas import HealthResponse, HealthData
from app.services.tumor_classifier import tumor_classifier

router = APIRouter(prefix="/health", tags=["Health"])

# Track when the app started
_start_time = time.time()


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API and model are ready to serve predictions."
)
def health_check() -> HealthResponse:
    """
    Returns API and model health status.
    
    - status: 'healthy' or 'degraded'
    - model_loaded: False means predictions will fail
    - device: 'cpu' or 'cuda' — tells you if GPU is being used
    - uptime_seconds: how long the server has been running
    """
    model_info = tumor_classifier.get_model_info()
    
    return HealthResponse(
        success=True,
        message="API is running",
        data=HealthData(
            status="healthy" if tumor_classifier.is_loaded else "degraded",
            model_loaded=tumor_classifier.is_loaded,
            model_version=model_info["model_version"],
            device=model_info["device"],
            uptime_seconds=round(time.time() - _start_time, 2),
        )
    )
```

### 7.2 Model Info Endpoint

Create `app/api/v1/endpoints/model_info.py`:

```python
# app/api/v1/endpoints/model_info.py

"""
MODEL INFO ENDPOINT

WHY expose model metadata?
  - API consumers need to know what classes are available
  - Input requirements (image size, format) should be documented
  - Version info helps track which model is deployed
"""

from fastapi import APIRouter, Depends
from app.schemas.response_schemas import ModelInfoResponse
from app.services.tumor_classifier import tumor_classifier
from app.core.security import require_api_key

router = APIRouter(prefix="/model", tags=["Model"])


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Get metadata about the deployed classification model."
)
def get_model_info(
    api_key: dict = Depends(require_api_key)   # Protected endpoint
) -> ModelInfoResponse:
    """
    Returns metadata about the loaded model including:
    - Architecture name and version
    - Classes it can predict
    - Required input dimensions
    - Normalization parameters
    """
    info = tumor_classifier.get_model_info()
    return ModelInfoResponse(
        success=True,
        message="Model information retrieved",
        data=info,
    )
```

### 7.3 Prediction Endpoint

This is the core of the entire project. Read every comment.

Create `app/api/v1/endpoints/predict.py`:

```python
# app/api/v1/endpoints/predict.py

"""
PREDICTION ENDPOINTS

Two endpoints:
  1. POST /predict/tumor — Single image classification
  2. POST /predict/batch  — Multiple images in one request

FILE UPLOADS IN FASTAPI:
  - Use UploadFile (FastAPI type) to receive image files
  - File content is in memory (small files) or spooled to disk (large files)
  - You must await file.read() to get the raw bytes
  - Convert bytes → PIL Image for the ML service

VALIDATION STRATEGY:
  We validate at multiple levels:
  1. File type: reject non-images by checking content type header
  2. File size: reject huge files that could OOM the server
  3. PIL open: catch corrupted/invalid image files
  4. Channel validation: ensure image has 3 channels after conversion
"""

import io
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image
from loguru import logger

from app.schemas.response_schemas import (
    PredictionResponse,
    PredictionResult,
    TumorProbabilities,
    BatchPredictionResponse,
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
    """
    Validate an uploaded file and return a PIL Image.
    
    Raises HTTPException with a clear message if validation fails.
    This function is reused by both single and batch prediction.
    
    Validation steps:
      1. Check content type header (quick check, but can be spoofed)
      2. Check file size
      3. Actually open with PIL (definitive check — PIL will reject invalid images)
    """
    # Check 1: Content type
    # content_type might be None if client doesn't set it
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{file.content_type}'. "
                f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}"
            )
        )
    
    # Check 2: File size
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > settings.max_image_size_mb:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {file_size_mb:.1f}MB. "
                f"Maximum allowed: {settings.max_image_size_mb}MB"
            )
        )
    
    # Check 3: Try to open with PIL
    # This is the real validation — it reads the actual image data
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()   # Verify it's not corrupted
        # Re-open after verify() (verify() consumes the file object)
        image = Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not open image file '{file.filename}': {str(e)}"
        )
    
    return image


def dict_to_prediction_result(result_dict: dict) -> PredictionResult:
    """
    Convert the raw dict from tumor_classifier.predict() to a Pydantic model.
    
    WHY do this separately?
    - tumor_classifier returns plain dicts (no FastAPI dependency)
    - We convert to Pydantic here (at the API layer)
    - Clean separation of concerns
    """
    probs = result_dict["probabilities"]
    return PredictionResult(
        predicted_class=result_dict["predicted_class"],
        confidence=result_dict["confidence"],
        probabilities=TumorProbabilities(
            glioma=probs["glioma"],
            meningioma=probs["meningioma"],
            notumor=probs["notumor"],
            pituitary=probs["pituitary"],
        ),
        description=result_dict["description"],
        inference_time_ms=result_dict["inference_time_ms"],
        model_version=result_dict["model_version"],
    )


# ── Single Image Prediction ────────────────────────────────────────────────────
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
        description="Brain MRI image file (JPEG or PNG, max 10MB)"
    ),
    api_key: dict = Depends(require_api_key),   # Inject auth dependency
) -> PredictionResponse:
    """
    Classify a brain MRI scan into one of four categories:
    - **glioma**: tumor from glial cells
    - **meningioma**: tumor in brain membranes
    - **notumor**: no tumor detected
    - **pituitary**: tumor in the pituitary gland
    
    Returns confidence score and probability distribution across all classes.
    """
    # Guard: model must be loaded
    if not tumor_classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Service is starting up, try again shortly."
        )
    
    logger.info(f"Single prediction request | file={file.filename} | type={file.content_type}")
    
    # Read file bytes
    # WHY await? UploadFile.read() is async — it reads from async buffer
    file_bytes = await file.read()
    
    # Validate and open image
    image = validate_and_open_image(file, file_bytes)
    
    # Run inference
    try:
        result_dict = tumor_classifier.predict(image)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )
    
    result = dict_to_prediction_result(result_dict)
    
    logger.info(
        f"Prediction complete | "
        f"file={file.filename} | "
        f"class={result.predicted_class} | "
        f"confidence={result.confidence:.3f}"
    )
    
    return PredictionResponse(
        success=True,
        message=f"Successfully classified as '{result.predicted_class}'",
        data=result,
    )


# ── Batch Prediction ────────────────────────────────────────────────────────────
@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Classify Multiple MRI Images",
    description=(
        f"Upload multiple brain MRI images in one request. "
        f"Max {settings.max_batch_size} images per request."
    ),
)
async def predict_batch(
    files: list[UploadFile] = File(
        ...,
        description=f"Multiple brain MRI image files (max {settings.max_batch_size})"
    ),
    api_key: dict = Depends(require_api_key),
) -> BatchPredictionResponse:
    """
    Classify multiple brain MRI scans in a single API call.
    
    WHY use batch endpoint vs calling single endpoint in a loop?
    - Batch processes all images in parallel on GPU (much faster)
    - Fewer HTTP round-trips
    - More efficient memory usage
    
    Results are returned in the same order as the uploaded files.
    If one image fails validation, its result will have an 'error' field
    instead of a 'result' field — other images still get classified.
    """
    if not tumor_classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Max {settings.max_batch_size} per batch request."
        )
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided.")
    
    logger.info(f"Batch prediction request | n_files={len(files)}")
    
    # Step 1: Read all files and validate each one
    # Invalid images get an error entry; valid ones are collected for batch inference
    valid_images: list[Image.Image] = []
    valid_indices: list[int] = []
    
    # Pre-allocate results list with error placeholders
    results: list[BatchPredictionItem] = []
    
    for i, file in enumerate(files):
        try:
            file_bytes = await file.read()
            image = validate_and_open_image(file, file_bytes)
            valid_images.append(image)
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
    
    # Step 2: Run batch inference on all valid images
    if valid_images:
        try:
            batch_results = tumor_classifier.predict_batch(valid_images)
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
        data={
            "total": len(files),
            "successful": successful,
            "failed": failed,
            "results": [r.model_dump() for r in results],
        }
    )
```

### 7.4 V1 Router — Combining All Endpoints

Create `app/api/v1/router.py`:

```python
# app/api/v1/router.py

"""
The router combines all endpoint modules under the /api/v1 prefix.

WHY versioned prefix (/api/v1)?
  When you need to change the API in a breaking way (different response
  format, removed fields), you create /api/v2 instead of breaking clients
  that depend on /api/v1.
  
  This lets you run v1 and v2 side-by-side during a migration period.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import health, predict, model_info

v1_router = APIRouter(prefix="/api/v1")

# Each sub-router has its own prefix defined in its module:
#   health  → /api/v1/health/
#   predict → /api/v1/predict/tumor, /api/v1/predict/batch
#   model   → /api/v1/model/info
v1_router.include_router(health.router)
v1_router.include_router(predict.router)
v1_router.include_router(model_info.router)
```

---

## 8. Authentication — API Key Protection

**Why protect the prediction endpoint?** Running model inference is computationally expensive. Without auth, anyone who discovers your API URL can exhaust your server's resources. API keys let you control who has access.

Create `app/core/security.py`:

```python
# app/core/security.py

"""
API KEY AUTHENTICATION

Pattern:
  1. Client sends "X-API-Key: sk-tumor-dev-abc123" in request header
  2. FastAPI's Depends() calls require_api_key() before the endpoint function
  3. require_api_key() checks the key against VALID_KEYS
  4. If invalid → HTTPException(401) → endpoint never runs
  5. If valid → return key info dict → endpoint receives it as parameter

WHY use Depends() instead of checking inside each endpoint?
  - DRY: auth logic in one place, not copy-pasted into every endpoint
  - FastAPI documents the security requirement automatically in Swagger
  - Endpoint functions are clean — they just receive the validated key info
  - Easy to swap to JWT later by changing only this file
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from loguru import logger
from app.config import settings

# This tells FastAPI to look for the header "X-API-Key" in requests
# auto_error=False: we handle the error ourselves with a custom message
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(API_KEY_HEADER)) -> dict:
    """
    FastAPI dependency for API key authentication.
    
    HOW to use in an endpoint:
        @router.post("/predict/tumor")
        async def predict(
            file: UploadFile,
            api_key: dict = Depends(require_api_key)
        ):
            ...
    
    Returns a dict with key metadata (extensible — you can add user_id, tier, etc.)
    Raises HTTPException(401) if the key is missing or invalid.
    """
    valid_keys = settings.get_api_keys_list()
    
    if not api_key:
        logger.warning("Request rejected: no API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Add header: X-API-Key: your-key-here",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key not in valid_keys:
        # Log the prefix only (never log full API keys — they're credentials)
        key_prefix = api_key[:12] + "..." if len(api_key) > 12 else api_key
        logger.warning(f"Request rejected: invalid API key starting with '{key_prefix}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Return key metadata — extensible for future use
    return {
        "api_key": api_key,
        "key_prefix": api_key[:8] + "...",
    }
```

---

## 9. Middleware — Logging, CORS, Timing

**Why middleware?** Middleware wraps every request and response without touching your endpoint code. Perfect for cross-cutting concerns: logging every request, adding response headers, measuring latency.

Create `app/core/middleware.py`:

```python
# app/core/middleware.py

"""
MIDDLEWARE

Middleware intercepts every request BEFORE it reaches your endpoint
and every response AFTER your endpoint returns.

Timeline:
  Request → Middleware (enter) → Endpoint → Middleware (exit) → Response

WHY log request/response here instead of in each endpoint?
  - You'd have to add logging code to every single endpoint function
  - Middleware guarantees every request is logged, even unexpected paths
  - Timing the full request (including Pydantic validation) gives true latency
"""

import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every incoming request and outgoing response with:
      - A unique request ID (for correlating logs when debugging)
      - Method and path
      - Response status code
      - Total processing time
    
    Log format:
      → REQUEST  | id=abc123 | method=POST | path=/api/v1/predict/tumor
      ← RESPONSE | id=abc123 | status=200  | time=47.3ms
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate unique ID for this request
        # WHY? When you have hundreds of concurrent requests in logs,
        # you need a way to match the request log to its response log.
        request_id = str(uuid.uuid4())[:8]
        
        # Attach to request state so endpoints can access it if needed
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Skip detailed logging for health checks (they happen every ~10s)
        is_health = "/health" in request.url.path
        
        if not is_health:
            logger.info(
                f"→ REQUEST  | id={request_id} | "
                f"method={request.method} | "
                f"path={request.url.path} | "
                f"client={request.client.host if request.client else 'unknown'}"
            )
        
        # Process the request (call the actual endpoint)
        response = await call_next(request)
        
        process_time_ms = (time.time() - start_time) * 1000
        
        # Add request ID to response headers
        # Clients can use this to report issues: "I got an error, my request ID was abc123"
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-ms"] = f"{process_time_ms:.1f}"
        
        if not is_health:
            log_fn = logger.info if response.status_code < 400 else logger.warning
            log_fn(
                f"← RESPONSE | id={request_id} | "
                f"status={response.status_code} | "
                f"time={process_time_ms:.1f}ms"
            )
        
        return response
```

---

## 10. Error Handling — Never Crash Silently

**Why custom error handlers?** FastAPI's default errors return technical messages that expose internals. You want clean, consistent error messages that help API consumers without revealing implementation details.

These handlers go in `main.py`, but understand them now.

**The error types you'll encounter:**

| Error | When it happens | Status Code |
|-------|-----------------|-------------|
| `RequestValidationError` | Pydantic rejects the input | 422 |
| `HTTPException` | You raise it explicitly | Varies |
| `Exception` | Unexpected crash | 500 |

---

## 11. Main App — Wiring It All Together

This is the final assembly. Every concept from previous sections plugs in here.

Create `app/main.py`:

```python
# app/main.py

"""
THE MAIN APPLICATION FILE

Responsibilities:
  1. Create the FastAPI app instance with metadata
  2. Register middleware (order matters — added last, runs first)
  3. Register exception handlers
  4. Register routers
  5. Manage startup/shutdown lifecycle (load/unload model)

LIFESPAN vs @app.on_event (deprecated):
  FastAPI now recommends the @asynccontextmanager lifespan pattern.
  Code before `yield` runs at startup.
  Code after `yield` runs at shutdown.
  
  WHY load model at startup?
  - Avoids 1-2 second cold start on the first request
  - If model fails to load, the server fails to start (fail fast)
  - Model is in memory for ALL requests, not reloaded each time
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.v1.router import v1_router
from app.services.tumor_classifier import tumor_classifier
from app.core.middleware import RequestLoggingMiddleware
from app.config import settings, setup_logging


# ── Startup / Shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle.
    
    Startup sequence:
      1. Initialize logging
      2. Load ML model (blocking — takes ~1-2s)
      3. Log ready message
    
    Shutdown sequence:
      4. Log shutdown (model memory released automatically by Python GC)
    """
    # ═══ STARTUP ════════════════════════════════════════════════════
    # Create logs directory if it doesn't exist
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
        # If model file is missing, the API is useless — fail fast
        logger.critical(f"❌ Model load failed: {e}")
        logger.critical("Place 'best.pth' in app/ml_models/ and restart.")
        # Don't raise — let API start but health check will report 'degraded'
    
    yield   # ← API serves requests between yield and shutdown
    
    # ═══ SHUTDOWN ═══════════════════════════════════════════════════
    logger.info("Shutting down gracefully...")


# ── App Instance ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## 🧠 Brain Tumor Classification API

Upload brain MRI images to classify them using a fine-tuned **EfficientNetV2-S** model.

### Classes
| Class | Description |
|-------|-------------|
| `glioma` | Tumor from glial cells |
| `meningioma` | Tumor in brain membranes |
| `notumor` | No tumor detected |
| `pituitary` | Pituitary gland tumor |

### Authentication
All prediction and model endpoints require an **API key** via the `X-API-Key` header.

```
X-API-Key: sk-tumor-dev-abc123
```

### Quick Test
1. Hit **GET /api/v1/health/** — check if model is loaded
2. Add your API key in the Authorize button (🔒) above
3. Hit **POST /api/v1/predict/tumor** — upload a test MRI image
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Middleware ─────────────────────────────────────────────────────────────────
# IMPORTANT: Middleware runs in REVERSE order of how it's added.
# Last added = outermost = runs first on request.

# 1. Request logging (added first = innermost = runs last before endpoint)
app.add_middleware(RequestLoggingMiddleware)

# 2. CORS (Cross-Origin Resource Sharing)
# WHY? If a frontend at localhost:3000 calls your API at localhost:8000,
# the browser blocks it unless the API explicitly allows it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # Local React dev server
        "http://localhost:8080",    # Other local tools
        # "https://yourdomain.com" ← Add your production frontend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],   # Only allow what we use
    allow_headers=["*"],
)


# ── Exception Handlers ─────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles Pydantic validation errors.
    
    Default FastAPI error message is technical and hard to read.
    This transforms it into clean, field-by-field error messages.
    
    Example — instead of:
      {"detail": [{"loc": ["body", "file"], "msg": "field required", ...}]}
    
    We return:
      {"success": false, "error": "Validation failed", "details": [
        {"field": "body → file", "message": "field required"}
      ]}
    """
    errors = [
        {
            "field": " → ".join(str(loc) for loc in err["loc"]),
            "message": err["msg"]
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Request validation failed",
            "details": errors,
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unexpected errors.
    
    WHY? Without this, unhandled exceptions return a raw 500 with
    a Python traceback, which may leak implementation details.
    
    We log the full traceback for our debugging but return a generic
    safe message to the client.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error. Please try again or contact support.",
        }
    )


# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(v1_router)


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    """API root — useful for quick connectivity checks."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health/",
    }
```

---

## 12. Testing — Prove It Works

**Why write tests?** Because "it worked in Postman" is not a deployment strategy. Tests let you refactor with confidence, catch regressions before they reach users, and document how your API is supposed to behave.

### Test Fixtures

Create `tests/conftest.py`:

```python
# tests/conftest.py

"""
TEST FIXTURES

pytest fixtures are reusable setup/teardown functions.
The TestClient simulates HTTP requests without starting a real server.

WHY TestClient instead of running uvicorn?
  - Tests run in-process (no network overhead)
  - No port conflicts
  - Fully deterministic (no async timing issues)
  - Works in CI/CD without port management
"""

import pytest
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.services.tumor_classifier import tumor_classifier
from app.config import settings


@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient for the FastAPI app.
    scope="module" means this runs once per test file,
    not once per test function — model loads only once.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def valid_api_key():
    """Return the first valid API key from settings."""
    return settings.get_api_keys_list()[0]


@pytest.fixture
def valid_image_bytes():
    """
    Create a valid 224x224 RGB JPEG image in memory.
    We use a random array — the model will classify it, we just
    care that the endpoint WORKS, not what it classifies.
    """
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def tiny_image_bytes():
    """A tiny 10x10 image — model should still handle it after resize."""
    img_array = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()
```

### Prediction Tests

Create `tests/test_predict.py`:

```python
# tests/test_predict.py

"""
PREDICTION ENDPOINT TESTS

Test strategy:
  1. Happy path — valid image, valid key → 200 with correct response shape
  2. Auth failures — no key, bad key → 401
  3. File validation — wrong type, too large → 415, 413
  4. Response structure — check all required fields exist and have correct types
  5. Batch endpoint — multiple files, mixed valid/invalid

Good tests verify BEHAVIOR, not implementation.
They test what the API promises to do, not how it does it.
"""

import pytest
from io import BytesIO


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health/")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/api/v1/health/")
        data = response.json()
        assert "success" in data
        assert "data" in data
        assert "model_loaded" in data["data"]
        assert "device" in data["data"]

    def test_health_model_is_loaded(self, client):
        response = client.get("/api/v1/health/")
        data = response.json()
        # If this fails, your model weights aren't at the configured path
        assert data["data"]["model_loaded"] is True


class TestAuthentication:
    def test_predict_without_api_key_returns_401(self, client, valid_image_bytes):
        response = client.post(
            "/api/v1/predict/tumor",
            files={"file": ("scan.jpg", valid_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 401

    def test_predict_with_invalid_api_key_returns_401(self, client, valid_image_bytes):
        response = client.post(
            "/api/v1/predict/tumor",
            files={"file": ("scan.jpg", valid_image_bytes, "image/jpeg")},
            headers={"X-API-Key": "this-is-not-a-valid-key"},
        )
        assert response.status_code == 401

    def test_predict_with_valid_api_key_returns_200(self, client, valid_image_bytes, valid_api_key):
        response = client.post(
            "/api/v1/predict/tumor",
            files={"file": ("scan.jpg", valid_image_bytes, "image/jpeg")},
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 200


class TestSinglePrediction:
    def test_prediction_response_has_correct_structure(
        self, client, valid_image_bytes, valid_api_key
    ):
        response = client.post(
            "/api/v1/predict/tumor",
            files={"file": ("scan.jpg", valid_image_bytes, "image/jpeg")},
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 200
        data = response.json()
        
        # Top-level
        assert data["success"] is True
        assert "data" in data
        
        result = data["data"]
        
        # Required fields
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "description" in result
        assert "inference_time_ms" in result
        assert "model_version" in result

    def test_predicted_class_is_valid(self, client, valid_image_bytes, valid_api_key):
        VALID_CLASSES = {"glioma", "meningioma", "notumor", "pituitary"}
        response = client.post(
            "/api/v1/predict/tumor",
            files={"file": ("scan.jpg", valid_image_bytes, "image/jpeg")},
            headers={"X-API-Key": valid_api_key},
        )
        data = response.json()
        assert data["data"]["predicted_class"] in VALID_CLASSES

    def test_confidence_is_between_0_and_1(self, client, valid_image_bytes, valid_api_key):
        response = client.post(
            "/api/v1/predict/tumor",
            files={"file": ("scan.jpg", valid_image_bytes, "image/jpeg")},
            headers={"X-API-Key": valid_api_key},
        )
        data = response.json()
        confidence = data["data"]["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_probabilities_sum_to_one(self, client, valid_image_bytes, valid_api_key):
        response = client.post(
            "/api/v1/predict/tumor",
            files={"file": ("scan.jpg", valid_image_bytes, "image/jpeg")},
            headers={"X-API-Key": valid_api_key},
        )
        data = response.json()
        probs = data["data"]["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01   # Allow small floating point errors

    def test_non_image_file_returns_415_or_422(self, client, valid_api_key):
        # Send a text file as if it were an image
        response = client.post(
            "/api/v1/predict/tumor",
            files={"file": ("data.txt", b"not an image", "text/plain")},
            headers={"X-API-Key": valid_api_key},
        )
        # Either 415 (wrong content type) or 422 (can't open as image)
        assert response.status_code in (415, 422)

    def test_corrupted_image_returns_422(self, client, valid_api_key):
        response = client.post(
            "/api/v1/predict/tumor",
            files={"file": ("broken.jpg", b"\x00\xFF\xD8corrupted", "image/jpeg")},
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 422

    def test_png_image_works(self, client, valid_api_key):
        """Verify PNG is also accepted, not just JPEG."""
        import numpy as np
        from PIL import Image
        
        img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        response = client.post(
            "/api/v1/predict/tumor",
            files={"file": ("scan.png", buffer.getvalue(), "image/png")},
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 200

    def test_response_includes_request_id_header(self, client, valid_image_bytes, valid_api_key):
        response = client.post(
            "/api/v1/predict/tumor",
            files={"file": ("scan.jpg", valid_image_bytes, "image/jpeg")},
            headers={"X-API-Key": valid_api_key},
        )
        assert "x-request-id" in response.headers
        assert "x-process-time-ms" in response.headers


class TestBatchPrediction:
    def test_batch_with_two_images(self, client, valid_image_bytes, valid_api_key):
        response = client.post(
            "/api/v1/predict/batch",
            files=[
                ("files", ("scan1.jpg", valid_image_bytes, "image/jpeg")),
                ("files", ("scan2.jpg", valid_image_bytes, "image/jpeg")),
            ],
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total"] == 2
        assert data["data"]["successful"] == 2
        assert len(data["data"]["results"]) == 2

    def test_batch_too_many_files_returns_400(self, client, valid_image_bytes, valid_api_key):
        from app.config import settings
        
        # Create max+1 files
        files = [
            ("files", (f"scan{i}.jpg", valid_image_bytes, "image/jpeg"))
            for i in range(settings.max_batch_size + 1)
        ]
        response = client.post(
            "/api/v1/predict/batch",
            files=files,
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 400

    def test_batch_partial_failure(self, client, valid_image_bytes, valid_api_key):
        """One bad image shouldn't fail the whole batch."""
        response = client.post(
            "/api/v1/predict/batch",
            files=[
                ("files", ("good.jpg", valid_image_bytes, "image/jpeg")),
                ("files", ("bad.txt", b"not an image", "text/plain")),
            ],
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["successful"] == 1
        assert data["data"]["failed"] == 1


class TestModelInfo:
    def test_model_info_returns_metadata(self, client, valid_api_key):
        response = client.get(
            "/api/v1/model/info",
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert "classes" in data
        assert len(data["classes"]) == 4
        assert "glioma" in data["classes"]
```

### Run Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pip install pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing

# Run a specific test class
pytest tests/test_predict.py::TestSinglePrediction -v

# Run a specific test
pytest tests/test_predict.py::TestSinglePrediction::test_probabilities_sum_to_one -v
```

---

## 13. Docker — Containerize Everything

**Why Docker?** Your laptop and your server have different Python versions, different system libraries, different OS. Docker packages your entire environment (OS, Python, dependencies, code) into a single reproducible container. "Works on my machine" becomes "works everywhere."

### Dockerfile

Create `Dockerfile`:

```dockerfile
# Dockerfile

# ── Stage 1: Base Image ────────────────────────────────────────────────────────
# python:3.11-slim is ~150MB vs python:3.11 at ~900MB
# Smaller image = faster downloads, smaller attack surface
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# ── Environment Variables ──────────────────────────────────────────────────────
# PYTHONDONTWRITEBYTECODE: Don't write .pyc files (saves disk space)
# PYTHONUNBUFFERED: Print logs immediately (not buffered)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System Dependencies ────────────────────────────────────────────────────────
# These are needed for Pillow to open various image formats
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    
# WHY --no-install-recommends?
# Avoids installing docs, man pages, extra packages we don't need.
# Keeps the image smaller.

# ── Python Dependencies ────────────────────────────────────────────────────────
# Copy requirements first (before code) for Docker layer caching.
# WHY? Docker caches each layer. If requirements.txt doesn't change,
# Docker reuses the cached pip install layer — much faster rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Code ───────────────────────────────────────────────────────────
# Copy code after installing deps (so code changes don't bust the dep cache)
COPY app/ ./app/

# ── Model Weights ──────────────────────────────────────────────────────────────
# Copy the trained model weights
COPY app/ml_models/best.pth ./app/ml_models/best.pth

# ── Non-root User ─────────────────────────────────────────────────────────────
# Running as root inside containers is a security risk.
# Create a non-root user for running the application.
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# ── Port ───────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Startup Command ────────────────────────────────────────────────────────────
# --workers 1: Single worker (one model instance in memory)
# --host 0.0.0.0: Listen on all interfaces (required in Docker)
# --port 8000: Match the EXPOSE port above
# WHY no --reload? Never use --reload in production. It restarts on file
# changes, which is useful for development but wasteful (and risky) in prod.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
# docker-compose.yml

# Docker Compose orchestrates multiple services.
# For this project: just the API (expandable later to add Redis, DB, etc.)

version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    
    # Map host port 8000 → container port 8000
    ports:
      - "8000:8000"
    
    # Load environment variables from .env file
    # These override defaults in config.py
    env_file:
      - .env
    
    # Mount logs directory from host for persistence
    # Container logs → host ./logs/ (survives container restarts)
    volumes:
      - ./logs:/app/logs
    
    # Restart policy: restart if container crashes (not if stopped manually)
    restart: unless-stopped
    
    # Health check: Docker will monitor this endpoint
    # If /api/v1/health/ fails for 3 consecutive checks,
    # Docker marks the container as unhealthy (visible in `docker ps`)
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health/"]
      interval: 30s      # Check every 30 seconds
      timeout: 10s       # Fail if no response in 10s
      retries: 3         # Mark unhealthy after 3 failures
      start_period: 60s  # Give 60s to start up before first check
    
    # Resource limits (important for prod — prevent memory leaks from killing host)
    deploy:
      resources:
        limits:
          memory: 2G     # EfficientNetV2-S + PyTorch needs ~500MB minimum
          cpus: "2.0"
```

### Docker Commands

```bash
# Build the image
docker build -t brain-tumor-api:latest .

# Run with docker-compose (recommended)
docker compose up -d                   # -d = detached (background)

# View logs
docker compose logs -f api            # -f = follow (stream new logs)

# Check status (look for 'healthy' in STATUS column)
docker ps

# Stop
docker compose down

# Rebuild after code changes
docker compose up --build -d

# Open shell inside running container (for debugging)
docker compose exec api /bin/bash

# Run tests inside container
docker compose exec api pytest tests/ -v
```

---

## 14. Running & Manual Testing

### Start the API (Development)

```bash
# Make sure virtual environment is active
source venv/bin/activate

# Start with auto-reload (dev mode)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
<green>2024-01-15 10:00:00</green> | <cyan>INFO</cyan> | Starting Brain Tumor Classification API v1.0.0...
<green>2024-01-15 10:00:00</green> | <green>SUCCESS</green> | ✅ Model loaded. API is ready to serve predictions.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Test with curl

```bash
# 1. Root
curl http://localhost:8000/

# 2. Health check
curl http://localhost:8000/api/v1/health/ | python -m json.tool

# 3. Predict (replace scan.jpg with your actual MRI image)
curl -X POST http://localhost:8000/api/v1/predict/tumor \
  -H "X-API-Key: sk-tumor-dev-abc123" \
  -F "file=@sample_images/scan.jpg"

# 4. Batch predict
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "X-API-Key: sk-tumor-dev-abc123" \
  -F "files=@sample_images/scan1.jpg" \
  -F "files=@sample_images/scan2.jpg"

# 5. Model info
curl http://localhost:8000/api/v1/model/info \
  -H "X-API-Key: sk-tumor-dev-abc123" \
  | python -m json.tool

# 6. Test auth failure
curl -X POST http://localhost:8000/api/v1/predict/tumor \
  -F "file=@sample_images/scan.jpg"
# Should return 401

# 7. View Swagger UI
open http://localhost:8000/docs
```

### Test with Python

```python
# test_api.py — run this from project root to test manually

import requests

BASE_URL = "http://localhost:8000"
API_KEY = "sk-tumor-dev-abc123"
HEADERS = {"X-API-Key": API_KEY}

# 1. Health check
r = requests.get(f"{BASE_URL}/api/v1/health/")
print("Health:", r.json())

# 2. Single prediction
with open("sample_images/scan.jpg", "rb") as f:
    r = requests.post(
        f"{BASE_URL}/api/v1/predict/tumor",
        headers=HEADERS,
        files={"file": ("scan.jpg", f, "image/jpeg")},
    )
    print("\nSingle Prediction:")
    print(f"  Status: {r.status_code}")
    result = r.json()
    print(f"  Class: {result['data']['predicted_class']}")
    print(f"  Confidence: {result['data']['confidence']:.2%}")
    print(f"  Inference time: {result['data']['inference_time_ms']:.1f}ms")
    print(f"  Probabilities: {result['data']['probabilities']}")
```

### Understanding the Swagger UI

Navigate to `http://localhost:8000/docs`. You'll see:

1. **Authorize button (🔒)**: Click it, enter your API key. All subsequent requests will include it.
2. **POST /api/v1/predict/tumor**: Click → "Try it out" → upload a file → "Execute". See the raw request and response.
3. **Schema section at bottom**: Shows the exact Pydantic model shapes — what you can send and what you'll get back.

---

## 15. Production Checklist & Next Steps

### Before You Deploy Anywhere Public

Work through this checklist methodically:

#### Security
- [ ] API keys are NOT hardcoded — they come from environment variables
- [ ] `.env` file is in `.gitignore` (run `echo ".env" >> .gitignore`)
- [ ] API keys are long, random strings (not "password123")
- [ ] Only `GET` and `POST` methods are allowed (set in CORS middleware)
- [ ] File size limit enforced (`max_image_size_mb` in config)
- [ ] Error messages don't expose stack traces or file paths

#### Reliability
- [ ] Health endpoint correctly reports model load status
- [ ] Docker healthcheck configured and tested
- [ ] Model file is included in Docker image (not fetched at runtime)
- [ ] `restart: unless-stopped` in docker-compose
- [ ] Logs written to persistent volume

#### Performance
- [ ] `model.eval()` called after loading (you'll see this in the service)
- [ ] `torch.no_grad()` used during inference
- [ ] Model loaded once at startup, not per-request
- [ ] Batch endpoint available for multi-image use cases

#### Code Quality
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No hardcoded values anywhere (everything in config or env)
- [ ] Error handling for all edge cases (bad file, model not loaded, etc.)

### Common Bugs and How to Fix Them

**"Model not loaded" on first request:**
Check that `best.pth` is in `app/ml_models/`. The path in `.env` is relative to where you run `uvicorn` from (the project root).

**"RuntimeError: Expected 3D tensor":**
The image likely has 1 or 4 channels. The `image.convert("RGB")` in `_preprocess()` fixes this. Make sure you didn't accidentally remove that line.

**Predictions are garbage (all ~25% confidence on every class):**
You're using `model.train()` mode. Make sure `model.eval()` is called after loading. This is in `tumor_classifier.load()` — verify it's there.

**"CORS error" from browser:**
Add your frontend URL to `allow_origins` in `main.py`. The URL must match exactly (include port number).

**Tests fail with "Model not loaded":**
The TestClient triggers the lifespan, which loads the model. If the model path is wrong in `.env`, the test setup will fail. Run `python scripts/test_model.py` first to verify the path.

### What to Build Next

Now that you have a solid foundation, here's the natural progression:

**Level 2 — Data Persistence:**
Add PostgreSQL + SQLAlchemy to store every prediction with timestamp, confidence, and API key used. This lets you monitor accuracy over time, detect model drift, and bill users.

```python
# The pattern (from Part 2 of the guides):
# After tumor_classifier.predict(), save to DB:
prediction_record = Prediction(
    model_name="EfficientNetV2-S",
    input_data={"filename": file.filename},
    predicted_label=result["predicted_class"],
    confidence=result["confidence"],
    inference_time_ms=result["inference_time_ms"],
)
db.add(prediction_record)
```

**Level 3 — Caching:**
Add Redis. If the same image is uploaded twice (same hash), return the cached result without running inference.

```python
import hashlib, redis
r = redis.Redis()
image_hash = hashlib.md5(file_bytes).hexdigest()
cached = r.get(f"prediction:{image_hash}")
if cached:
    return json.loads(cached)   # Instant response
```

**Level 4 — Metrics & Monitoring:**
Add Prometheus + Grafana to track: request rate, average latency, prediction class distribution, error rate. When your model starts predicting "glioma" on everything, Grafana alerts you before users notice.

**Level 5 — Model Versioning:**
Add MLflow to track training experiments. Register your best.pth as a versioned artifact. Deploy v2 alongside v1 with traffic splitting (95% v1, 5% v2). Promote v2 to 100% when it's better.

---

## Complete File Reference

Here's a summary of every file you created and what it does:

| File | Purpose |
|------|---------|
| `app/config.py` | Typed settings from environment variables |
| `app/main.py` | App instance, lifespan, middleware, exception handlers, router |
| `app/api/v1/router.py` | Combines health, predict, model_info routers |
| `app/api/v1/endpoints/health.py` | GET /api/v1/health/ — no auth needed |
| `app/api/v1/endpoints/predict.py` | POST /api/v1/predict/tumor and /batch |
| `app/api/v1/endpoints/model_info.py` | GET /api/v1/model/info |
| `app/core/security.py` | API key validation via FastAPI Depends() |
| `app/core/middleware.py` | Request logging, timing, X-Request-ID header |
| `app/schemas/response_schemas.py` | Pydantic output models |
| `app/services/tumor_classifier.py` | Model loading, preprocessing, inference |
| `tests/conftest.py` | Shared test fixtures (client, api key, image bytes) |
| `tests/test_predict.py` | Full test suite for all endpoints |
| `Dockerfile` | Container image definition |
| `docker-compose.yml` | Service orchestration |
| `.env` | Local environment variables (not committed to git) |

---

*Built specifically around your EfficientNetV2-S brain tumor classifier.  
The difference between a notebook experiment and a production API is exactly this — the wrapper, the validation, the observability, and the discipline.*