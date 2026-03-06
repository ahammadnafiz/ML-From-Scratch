#app/services/tumor_classifier.py

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


CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary",
}

CLASS_DESCRIPTIONS = {
    "glioma": "Tumor arising from glial cells in the brain or spinal cord",
    "meningioma": "Tumor in the protective membranes covering the brain and spinal cord",
    "notumor": "No tumor detected — healthy brain scan",
    "pituitary": "Tumor in the pituitary gland at the base of the brain",
}

INFERENCE_TRANSFORMS = v2.Compose([
    v2.Resize((224, 224)),                               # Fixed size for EfficientNetV2
    v2.PILToTensor(),                                    # PIL Image → torch.Tensor (C, H, W)
    v2.ToDtype(torch.float32),                           # int8 pixel values → float32
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],                      # ImageNet mean (R, G, B)
        std=[0.229, 0.224, 0.225]                        # ImageNet std  (R, G, B)
    ),
])

class TumorClassifier:
    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded: bool = False
        self.model_path: Optional[str] = None
        self.model_version: str = "unknown"

        logger.info(f"Initialized TumorClassifier on device: {self.device}")

    def _build_architecture(self) -> nn.Module:
        model = models.efficientnet_v2_s(weights=None)  # Start with uninitialized weights
        model.classifier[1] = nn.Linear(1280, len(CLASS_NAMES))  # Replace final layer for 4 classes
        return model

    def load(self, model_path: str, model_version: str = "1.0.0") -> None:
        path = Path(model_path)
        if not path.exists():
            logger.error(f"Model file not found at path: {model_path}")
            raise FileNotFoundError(f"Model file not found at path: {model_path}")
        logger.info(f"Loading model from path: {model_path}")
        start_time = time.time()

        model = self._build_architecture()
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)

        model.eval()  # Set model to evaluation mode
        model = model.to(self.device)

        self.model = model
        self.model_path = model_path
        self.model_version = model_version
        self.is_loaded = True

        load_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        logger.success(
            f"Model loaded successfully in {load_time:.2f} ms | "
            f"Model path: {self.model_path} | "
            f"device: {self.device} | "
        )

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")  # Ensure image is in RGB format
        tensor = INFERENCE_TRANSFORMS(image)  # Apply transformations
        tensor = tensor.unsqueeze(0)  # Add batch dimension (1, C, H, W)
        return tensor.to(self.device)
        
    def predict(self, image: Image.Image) -> dict:
        if not self.is_loaded or self.model is None:
            logger.error("Model is not loaded. Call load() before predict().")
            raise RuntimeError("Model is not loaded. Call load() before predict().")

        start_time = time.time()
        tensor = self._preprocess_image(image)

        with torch.no_grad():
            logits = self.model(tensor)
        
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # Convert to numpy array
        pred_class_idx = int(probs.argmax())
        confidence = float(probs[pred_class_idx])
        predicted_class = CLASS_NAMES[pred_class_idx]

        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        probabilities = {
            CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(CLASS_NAMES))
        }

        result = {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "description": CLASS_DESCRIPTIONS[predicted_class],
            "inference_time_ms": round(inference_time, 2),
            "model_version": self.model_version,
        }

        logger.debug(
            f"Prediction: class={predicted_class} | "
            f"confidence={confidence:.3f} | "
            f"time={inference_time:.1f}ms"
        )
        
        return result

    def predict_batch(self, images: list[Image.Image]) -> list[dict]:
        if not self.is_loaded or self.model is None:
            logger.error("Model is not loaded. Call load() before predict_batch().")
            raise RuntimeError("Model is not loaded. Call load() before predict_batch().")

        start_time = time.time()
        tensors = [self._preprocess_image(img) for img in images]
        batch_tensor = torch.cat(tensors, dim=0)  # Shape: (batch_size, C, H, W)

        with torch.no_grad():
            logits = self.model(batch_tensor)
        
        probs = F.softmax(logits, dim=1).cpu().numpy()  # Shape: (batch_size, num_classes)

        total_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
        per_image_time_ms = total_time_ms / len(images) if images else 0

        results = []
        for i in range(len(images)):
            pred_class_idx = int(probs[i].argmax())
            confidence = float(probs[i][pred_class_idx])
            predicted_class = CLASS_NAMES[pred_class_idx]

            probabilities = {
                CLASS_NAMES[j]: round(float(probs[i][j]), 4)
                for j in range(len(CLASS_NAMES))
            }

            result = {
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4),
                "probabilities": probabilities,
                "description": CLASS_DESCRIPTIONS[predicted_class],
                "inference_time_ms": round(per_image_time_ms, 2),  # Set the inference time for each image
                "model_version": self.model_version,
            }

    def get_model_info(self) -> dict:
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
    
tumor_classifier = TumorClassifier()