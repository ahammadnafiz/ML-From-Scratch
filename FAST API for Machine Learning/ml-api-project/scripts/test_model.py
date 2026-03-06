#scripts/test_model.py

from PIL import Image
import numpy as np
from app.services.tumor_classifier import tumor_classifier
from app.config import settings

# load model
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