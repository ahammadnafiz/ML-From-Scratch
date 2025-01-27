from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from PIL import Image
import torch
from torchvision import models, transforms

# Updated model loading with weights
weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
model.eval()

# Use model's built-in preprocessing
preprocess = weights.transforms()

app = FastAPI()

# Safer CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:80"],  # Add production domains
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

Instrumentator().instrument(app).expose(app)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        return {"class": "dog" if probs[232] > probs[281] else "cat"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))