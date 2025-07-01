# ONNX single image test for YOLOv12 model - Enhanced Robust Version
import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import os
from pathlib import Path
import json
import time
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'model_path': "model/best.onnx",
    'image_path': "image.jpg",
    'input_size': 640,
    'confidence_threshold': 0.45,
    'nms_threshold': 0.55,
    'selected_classes': ['singara', 'peyaju', 'beguni'],
    'max_detections': 1000,
    'agnostic_nms': True,  # Set True for class-agnostic NMS
    'multi_label': True,   # Set True if model supports multi-label detection
    'letterbox_color': (114, 114, 114),  # Padding color
    'save_results': True,
    'output_dir': 'results'
}

class YOLOv12Detector:
    def __init__(self, config: dict):
        self.config = config
        self.model_path = config['model_path']
        self.input_size = config['input_size']
        self.confidence_threshold = config['confidence_threshold']
        self.nms_threshold = config['nms_threshold']
        self.class_names = config['selected_classes']
        self.max_detections = config['max_detections']
        self.agnostic_nms = config['agnostic_nms']
        self.multi_label = config['multi_label']
        self.letterbox_color = config['letterbox_color']
        
        # Initialize ONNX session
        self.session = self._load_model()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Model metadata
        self._analyze_model_properties()
        
    def _load_model(self) -> ort.InferenceSession:
        """Load ONNX model with optimizations"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Set up session options for better performance
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Enable all available providers (CPU, CUDA if available)
        providers = ['CPUExecutionProvider']
        if ort.get_available_providers():
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers.insert(0, 'CUDAExecutionProvider')
            if 'TensorrtExecutionProvider' in available:
                providers.insert(0, 'TensorrtExecutionProvider')
        
        try:
            session = ort.InferenceSession(
                self.model_path, 
                sess_options=session_options,
                providers=providers
            )
            print(f"✅ Model loaded with providers: {session.get_providers()}")
            return session
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _analyze_model_properties(self):
        """Analyze model input/output properties"""
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()
        
        print(f"📊 Model Analysis:")
        print(f"   Input: {input_info.name} - Shape: {input_info.shape} - Type: {input_info.type}")
        for i, output in enumerate(output_info):
            print(f"   Output {i}: {output.name} - Shape: {output.shape} - Type: {output.type}")
        
        # Determine expected input size from model
        if input_info.shape and len(input_info.shape) >= 3:
            model_input_size = input_info.shape[-1] if input_info.shape[-1] > 0 else self.input_size
            if model_input_size != self.input_size:
                print(f"⚠️  Adjusting input size from {self.input_size} to {model_input_size}")
                self.input_size = model_input_size
    
    def letterbox_resize(self, image: np.ndarray, new_shape: int = 640, 
                        color: Tuple[int, int, int] = (114, 114, 114), 
                        auto: bool = False, scaleup: bool = True, 
                        stride: int = 32) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Enhanced letterbox resize with better aspect ratio handling
        """
        shape = image.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                  cv2.BORDER_CONSTANT, value=color)  # add border
        
        return image, ratio[0], (left, top)
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Enhanced preprocessing with better normalization"""
        # Store original dimensions
        original_shape = image.shape[:2]
        
        # Apply letterbox resize
        processed_image, ratio, pad = self.letterbox_resize(
            image, new_shape=self.input_size, color=self.letterbox_color
        )
        
        # Convert BGR to RGB
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] with proper dtype
        processed_image = processed_image.astype(np.float32) / 255.0
        
        # Convert to NCHW format and add batch dimension
        input_tensor = np.transpose(processed_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Ensure contiguous array for better performance
        input_tensor = np.ascontiguousarray(input_tensor)
        
        return input_tensor, ratio, pad
    
    def run_inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Run inference with error handling and timing"""
        try:
            start_time = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            inference_time = time.time() - start_time
            print(f"⚡ Inference time: {inference_time:.4f}s")
            return outputs
        except Exception as e:
            print(f"❌ Inference error: {e}")
            raise
    
    def postprocess_detections(self, outputs: List[np.ndarray], ratio: float, 
                             pad: Tuple[int, int], original_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enhanced post-processing with multiple output format support"""
        predictions = outputs[0]
        
        print(f"🔍 Raw predictions shape: {predictions.shape}")
        
        # Handle different YOLO output formats
        predictions = self._normalize_predictions(predictions)
        
        if len(predictions) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Extract components
        boxes, scores, class_ids = self._extract_detections(predictions)
        
        # Filter by confidence
        valid_mask = scores >= self.confidence_threshold
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]
        
        print(f"📊 After confidence filtering ({self.confidence_threshold}): {len(boxes)} detections")
        
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert boxes to original image coordinates
        boxes = self._rescale_boxes(boxes, ratio, pad, original_shape)
        
        # Apply Non-Maximum Suppression
        keep_indices = self._apply_nms(boxes, scores, class_ids)
        
        return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]
    
    def _normalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Normalize predictions to standard format [N, features]"""
        # Remove batch dimension if present
        if predictions.ndim == 3:
            predictions = predictions[0]
        
        # Handle different orientations
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        
        print(f"📐 Normalized predictions shape: {predictions.shape}")
        return predictions
    
    def _extract_detections(self, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract boxes, scores, and class IDs from predictions"""
        num_classes = len(self.class_names)
        
        # Extract bounding boxes (first 4 columns)
        boxes = predictions[:, :4]  # [x_center, y_center, width, height]
        
        # Handle different prediction formats
        if predictions.shape[1] == 4 + num_classes:
            # Format: [x, y, w, h, class0_conf, class1_conf, ...]
            class_scores = predictions[:, 4:]
            scores = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)
        elif predictions.shape[1] == 5 + num_classes:
            # Format: [x, y, w, h, objectness, class0_prob, class1_prob, ...]
            objectness = predictions[:, 4]
            class_probs = predictions[:, 5:]
            class_scores = objectness[:, np.newaxis] * class_probs
            scores = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)
        else:
            # Fallback: assume last columns are class scores
            class_scores = predictions[:, 4:]
            scores = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)
        
        # Handle multi-label detection if enabled
        if self.multi_label:
            # For multi-label, we might want to keep multiple high-confidence classes per box
            # This is a simplified implementation
            pass
        
        print(f"📈 Score statistics: min={np.min(scores):.4f}, max={np.max(scores):.4f}, mean={np.mean(scores):.4f}")
        print(f"🏷️  Class distribution: {dict(zip(*np.unique(class_ids, return_counts=True)))}")
        
        return boxes, scores, class_ids
    
    def _rescale_boxes(self, boxes: np.ndarray, ratio: float, pad: Tuple[int, int], 
                      original_shape: Tuple[int, int]) -> np.ndarray:
        """Convert boxes back to original image coordinates"""
        if len(boxes) == 0:
            return boxes
        
        pad_x, pad_y = pad
        
        # Convert from center format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Handle both normalized and pixel coordinates
        if np.max(boxes) <= 1.0:
            # Normalized coordinates
            x_center *= self.input_size
            y_center *= self.input_size
            width *= self.input_size
            height *= self.input_size
        
        # Convert to corner coordinates
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Remove padding
        x1 -= pad_x
        y1 -= pad_y
        x2 -= pad_x
        y2 -= pad_y
        
        # Scale back to original image size
        x1 /= ratio
        y1 /= ratio
        x2 /= ratio
        y2 /= ratio
        
        # Clip to image boundaries
        x1 = np.clip(x1, 0, original_shape[1])
        y1 = np.clip(y1, 0, original_shape[0])
        x2 = np.clip(x2, 0, original_shape[1])
        y2 = np.clip(y2, 0, original_shape[0])
        
        return np.column_stack([x1, y1, x2, y2])
    
    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray, 
                   class_ids: np.ndarray) -> np.ndarray:
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return np.array([], dtype=int)
        
        if self.agnostic_nms:
            # Class-agnostic NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(),
                self.confidence_threshold, self.nms_threshold
            )
        else:
            # Class-aware NMS
            keep_indices = []
            for class_id in np.unique(class_ids):
                class_mask = class_ids == class_id
                class_boxes = boxes[class_mask]
                class_scores = scores[class_mask]
                
                if len(class_boxes) > 0:
                    indices = cv2.dnn.NMSBoxes(
                        class_boxes.tolist(), class_scores.tolist(),
                        self.confidence_threshold, self.nms_threshold
                    )
                    
                    if len(indices) > 0:
                        indices = indices.flatten()
                        original_indices = np.where(class_mask)[0][indices]
                        keep_indices.extend(original_indices)
            
            indices = np.array(keep_indices) if keep_indices else np.array([])
        
        if isinstance(indices, tuple) and len(indices) > 0:
            indices = indices[0] if len(indices[0]) > 0 else np.array([])
        elif len(indices) > 0:
            indices = indices.flatten()
        else:
            indices = np.array([])
        
        # Limit number of detections
        if len(indices) > self.max_detections:
            # Sort by confidence and keep top detections
            sorted_indices = indices[np.argsort(scores[indices])[::-1][:self.max_detections]]
            indices = sorted_indices
        
        print(f"🎯 After NMS: {len(indices)} final detections")
        return indices
    
    def draw_detections(self, image: np.ndarray, boxes: np.ndarray, 
                       scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        """Enhanced visualization with better styling"""
        if len(boxes) == 0:
            return image
        
        annotated_image = image.copy()
        
        # Enhanced color palette
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0),
            (255, 192, 203), (165, 42, 42), (255, 69, 0), (34, 139, 34), (70, 130, 180)
        ]
        
        # Calculate font scale based on image size
        font_scale = max(0.3, min(image.shape[0], image.shape[1]) / 1000)
        thickness = max(1, int(font_scale * 3))
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            
            # Ensure valid class ID
            valid_class_id = max(0, min(class_id, len(self.class_names) - 1))
            class_name = self.class_names[valid_class_id]
            color = colors[valid_class_id % len(colors)]
            
            # Draw bounding box with variable thickness
            box_thickness = max(1, thickness)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, box_thickness)
            
            # Create label with confidence
            label = f"{class_name} {score:.3f}"
            
            # Calculate label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Draw label background
            label_y = max(y1, label_height + 10)
            cv2.rectangle(
                annotated_image,
                (x1, label_y - label_height - baseline - 5),
                (x1 + label_width + 5, label_y + baseline),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image, label,
                (x1 + 2, label_y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness
            )
        
        return annotated_image
    
    def detect(self, image_path: str) -> dict:
        """Main detection pipeline"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_shape = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor, ratio, pad = self.preprocess_image(image)
        
        # Run inference
        outputs = self.run_inference(input_tensor)
        
        # Post-process
        boxes, scores, class_ids = self.postprocess_detections(outputs, ratio, pad, original_shape)
        
        # Draw results
        annotated_image = self.draw_detections(image_rgb, boxes, scores, class_ids)
        
        return {
            'original_image': image_rgb,
            'annotated_image': annotated_image,
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'detections': len(boxes)
        }

def main():
    """Main execution function"""
    print("🎯 Enhanced ONNX YOLOv12 Detection Pipeline")
    print("=" * 50)
    
    # Create output directory
    if CONFIG['save_results']:
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Validate inputs
    if not os.path.exists(CONFIG['model_path']):
        print(f"❌ Model file not found: {CONFIG['model_path']}")
        return
    
    if not os.path.exists(CONFIG['image_path']):
        print(f"❌ Image file not found: {CONFIG['image_path']}")
        print(f"📁 Current directory: {os.getcwd()}")
        available_images = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))]
        if available_images:
            print(f"📷 Available images: {available_images}")
        return
    
    try:
        # Initialize detector
        detector = YOLOv12Detector(CONFIG)
        
        # Run detection
        print("\n🔍 Running detection...")
        results = detector.detect(CONFIG['image_path'])
        
        # Display results
        print(f"\n📊 Detection Results:")
        print(f"✅ Found {results['detections']} objects")
        
        if results['detections'] > 0:
            print("\n📋 Detailed detections:")
            for i, (box, score, class_id) in enumerate(zip(results['boxes'], results['scores'], results['class_ids'])):
                class_name = CONFIG['selected_classes'][min(class_id, len(CONFIG['selected_classes']) - 1)]
                x1, y1, x2, y2 = box.astype(int)
                print(f"   {i+1:2d}. {class_name:12s} | Conf: {score:.4f} | Box: ({x1:4d},{y1:4d},{x2:4d},{y2:4d}) | Area: {(x2-x1)*(y2-y1):6d}")
        
        # Visualization
        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(results['original_image'])
        plt.title("Original Image", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(results['annotated_image'])
        plt.title(f"Detection Results ({results['detections']} objects)", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save results if enabled
        if CONFIG['save_results']:
            timestamp = int(time.time())
            result_path = os.path.join(CONFIG['output_dir'], f'detection_result_{timestamp}.png')
            plt.savefig(result_path, dpi=300, bbox_inches='tight')
            print(f"💾 Results saved to: {result_path}")
            
            # Save detection data
            json_path = os.path.join(CONFIG['output_dir'], f'detection_data_{timestamp}.json')
            detection_data = {
                'image_path': CONFIG['image_path'],
                'model_path': CONFIG['model_path'],
                'detections': [
                    {
                        'class_name': CONFIG['selected_classes'][min(int(class_id), len(CONFIG['selected_classes']) - 1)],
                        'class_id': int(class_id),
                        'confidence': float(score),
                        'bbox': [float(x) for x in box]
                    }
                    for box, score, class_id in zip(results['boxes'], results['scores'], results['class_ids'])
                ],
                'config': CONFIG
            }
            
            with open(json_path, 'w') as f:
                json.dump(detection_data, f, indent=2)
            print(f"📄 Detection data saved to: {json_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()