# ONNX single image test for YOLOv12 model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import os
from pathlib import Path

# Update these paths
model_path = "model/best.onnx"  # Your ONNX model path
image_path = "image.png"  # Your test image path

# Your class names
SELECTED_CLASSES = [
    'singara',
    'peyaju',
    'beguni'
]

# YOLO model parameters (adjust based on your model)
INPUT_SIZE = 640  # Standard YOLO input size
CONFIDENCE_THRESHOLD = 0.20
NMS_THRESHOLD = 0.50

def preprocess_image(image, input_size=640):
    """Preprocess image for ONNX model inference"""
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    
    # Resize image while maintaining aspect ratio
    scale = min(input_size / original_width, input_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Create padded image
    padded_image = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    
    # Calculate padding offsets
    pad_x = (input_size - new_width) // 2
    pad_y = (input_size - new_height) // 2
    
    # Place resized image in center
    padded_image[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized_image
    
    # Convert to RGB and normalize
    padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    padded_image = padded_image.astype(np.float32) / 255.0
    
    # Convert to NCHW format (batch, channels, height, width)
    input_tensor = np.transpose(padded_image, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor, scale, pad_x, pad_y

def postprocess_detections(outputs, scale, pad_x, pad_y, conf_threshold=0.25, nms_threshold=0.45):
    """Post-process ONNX model outputs"""
    # Get predictions (assuming standard YOLO output format)
    predictions = outputs[0]  # Shape: [1, num_detections, 5 + num_classes] or [1, 7, 8400]
    
    print(f"Raw predictions shape: {predictions.shape}")
    
    # Handle different output formats
    if len(predictions.shape) == 3 and predictions.shape[1] > predictions.shape[2]:
        # Format: [1, num_detections, features] - transpose to [1, features, num_detections]
        predictions = np.transpose(predictions, (0, 2, 1))
        print(f"Transposed predictions shape: {predictions.shape}")
    
    # Remove batch dimension: [features, num_detections]
    predictions = predictions[0]
    print(f"After removing batch dim: {predictions.shape}")
    
    # For YOLOv11/v12: predictions format is [num_features, num_detections]
    # where num_features = 4 (bbox) + 1 (conf) + num_classes
    # Transpose to get [num_detections, num_features]
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T
        print(f"Final predictions shape: {predictions.shape}")
    
    # Extract boxes and scores
    boxes = predictions[:, :4]  # x_center, y_center, width, height
    
    # For YOLOv11, the format might be: [x, y, w, h, class0_conf, class1_conf, class2_conf]
    # No separate objectness score
    if predictions.shape[1] == 7:  # 4 bbox + 3 classes
        class_scores = predictions[:, 4:]  # All class confidences
        scores = np.max(class_scores, axis=1)  # Use max class confidence as objectness
    else:
        # Traditional format: [x, y, w, h, objectness, class0, class1, ...]
        scores = predictions[:, 4]  # objectness score
        class_scores = predictions[:, 5:]  # class probabilities
    
    # Get class with highest probability for each detection
    class_ids = np.argmax(class_scores, axis=1)
    class_confidences = np.max(class_scores, axis=1)
    
    # Use class confidence as final confidence (common in newer YOLO versions)
    confidences = class_confidences
    
    print(f"Max confidence: {np.max(confidences):.4f}, Min confidence: {np.min(confidences):.4f}")
    print(f"Class IDs range: {np.min(class_ids)} to {np.max(class_ids)}")
    print(f"Unique class IDs: {np.unique(class_ids)}")
    
    # Filter by confidence threshold
    valid_indices = confidences > conf_threshold
    boxes = boxes[valid_indices]
    confidences = confidences[valid_indices]
    class_ids = class_ids[valid_indices]
    
    print(f"After confidence filtering: {len(boxes)} detections")
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert from center format to corner format
    x_centers = boxes[:, 0]
    y_centers = boxes[:, 1]
    widths = boxes[:, 2]
    heights = boxes[:, 3]
    
    x1 = x_centers - widths / 2
    y1 = y_centers - heights / 2
    x2 = x_centers + widths / 2
    y2 = y_centers + heights / 2
    
    # Adjust coordinates back to original image scale
    # The coordinates are in normalized format (0-1) or pixel format (0-640)
    # Let's check and handle both cases
    if np.max(x2) <= 1.0:  # Normalized coordinates
        print("Detected normalized coordinates")
        x1 = x1 * INPUT_SIZE
        y1 = y1 * INPUT_SIZE
        x2 = x2 * INPUT_SIZE
        y2 = y2 * INPUT_SIZE
    else:
        print("Detected pixel coordinates")
    
    # Adjust for padding and scale
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale
    
    # Ensure coordinates are within image bounds
    x1 = np.clip(x1, 0, None)
    y1 = np.clip(y1, 0, None)
    x2 = np.clip(x2, 0, None)
    y2 = np.clip(y2, 0, None)
    
    boxes_xyxy = np.column_stack([x1, y1, x2, y2])
    
    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        confidences.tolist(),
        conf_threshold,
        nms_threshold
    )
    
    if len(indices) > 0:
        indices = indices.flatten()
        return boxes_xyxy[indices], confidences[indices], class_ids[indices]
    else:
        return np.array([]), np.array([]), np.array([])

def draw_detections(image, boxes, confidences, class_ids, class_names):
    """Draw bounding boxes and labels on image"""
    if len(boxes) == 0:
        return image
    
    annotated_image = image.copy()
    
    # Define colors for each class
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0)
    ]
    
    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Get class name and color
        # Ensure class_id is within valid range
        valid_class_id = class_id % len(class_names) if len(class_names) > 0 else 0
        class_name = class_names[valid_class_id] if valid_class_id < len(class_names) else f"Class_{class_id}"
        color = colors[valid_class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name} {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(
            annotated_image,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    return annotated_image

print("🎯 Testing ONNX YOLOv12 on single image...")
print(f"Model: {model_path}")
print(f"Image: {image_path}")

# Check if files exist
if not os.path.exists(model_path):
    print(f"❌ ONNX model file not found: {model_path}")
    print("Make sure you have exported your model to ONNX format")
    exit()
    
if not os.path.exists(image_path):
    print(f"❌ Image file not found: {image_path}")
    print(f"Current directory: {os.getcwd()}")
    print("Available files:", [f for f in os.listdir('.') if f.endswith(('.jpg', '.png', '.jpeg', '.onnx'))])
    exit()

try:
    # Load ONNX model
    print("📦 Loading ONNX model...")
    ort_session = ort.InferenceSession(model_path)
    
    # Get model input/output info
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    output_names = [output.name for output in ort_session.get_outputs()]
    
    print(f"✅ ONNX model loaded successfully")
    print(f"Input shape: {input_shape}")
    print(f"Input name: {input_name}")
    print(f"Output names: {output_names}")
    
except Exception as e:
    print(f"❌ Error loading ONNX model: {e}")
    print("Make sure you have onnxruntime installed: pip install onnxruntime")
    exit()

try:
    # Load and preprocess image
    print("📷 Loading and preprocessing image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        exit()
    
    original_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess image
    input_tensor, scale, pad_x, pad_y = preprocess_image(image, INPUT_SIZE)
    print(f"✅ Image preprocessed: {input_tensor.shape}")
    
except Exception as e:
    print(f"❌ Error preprocessing image: {e}")
    exit()

try:
    # Run inference
    print("🔍 Running ONNX inference...")
    outputs = ort_session.run(output_names, {input_name: input_tensor})
    print(f"✅ Inference completed")
    print(f"Output shape: {[output.shape for output in outputs]}")
    
except Exception as e:
    print(f"❌ Error during inference: {e}")
    exit()

try:
    # Post-process results
    print("📊 Post-processing detections...")
    boxes, confidences, class_ids = postprocess_detections(
        outputs, scale, pad_x, pad_y, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
    )
    
    print(f"✅ Found {len(boxes)} detections after NMS")
    
except Exception as e:
    print(f"❌ Error post-processing: {e}")
    exit()

try:
    # Draw detections
    print("🎨 Creating visualization...")
    annotated_image = draw_detections(image_rgb, boxes, confidences, class_ids, SELECTED_CLASSES)
    
    # Display results
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(annotated_image)
    plt.title(f"ONNX Detection Results ({len(boxes)} objects)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print(f"\n📊 Detection Summary:")
    if len(boxes) > 0:
        print(f"✅ Found {len(boxes)} objects:")
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            # Map class_id to valid range
            valid_class_id = class_id % len(SELECTED_CLASSES) if len(SELECTED_CLASSES) > 0 else 0
            class_name = SELECTED_CLASSES[valid_class_id] if valid_class_id < len(SELECTED_CLASSES) else f"Unknown_Class_{class_id}"
            x1, y1, x2, y2 = box.astype(int)
            print(f"  {i+1}. {class_name}: {conf:.3f} - Box: ({x1}, {y1}, {x2}, {y2})")
    else:
        print("❌ No objects detected")
        print("Try lowering confidence threshold or check if image contains target objects")

except Exception as e:
    print(f"❌ Error during visualization: {e}")
    # Still print detection results if available
    if 'boxes' in locals() and len(boxes) > 0:
        print(f"\n📊 Raw Detection Results:")
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            valid_class_id = class_id % len(SELECTED_CLASSES) if len(SELECTED_CLASSES) > 0 else 0
            class_name = SELECTED_CLASSES[valid_class_id] if valid_class_id < len(SELECTED_CLASSES) else f"Unknown_Class_{class_id}"
            x1, y1, x2, y2 = box.astype(int)
            print(f"  {i+1}. {class_name}: {conf:.3f} - Box: ({x1}, {y1}, {x2}, {y2})")

print("\n✅ ONNX inference completed!")