import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path
import json
import time
from typing import Tuple, List, Optional
import warnings
from PIL import Image
import io
import base64

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🥘 Bangladesh Street Food Detector",
    page_icon="🥘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
/* Dark theme styles */
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

.main-header {
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
}

.detection-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 15px;
    margin: 5px 0;
    border-left: 4px solid #4ECDC4;
}

.stButton > button {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.5rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.sidebar .sidebar-content {
    background-color: #262730;
}

.stSelectbox > div > div {
    background-color: #262730;
    color: #FAFAFA;
}

.stSlider > div > div > div {
    background-color: #4ECDC4;
}

/* Hide Streamlit menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Loading animation */
.loading {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100px;
}

.spinner {
    border: 4px solid rgba(255, 255, 255, 0.1);
    border-left: 4px solid #4ECDC4;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

class YOLOv12Detector:
    def __init__(self, model_path: str, class_names: List[str], 
                 input_size: int = 640, confidence_threshold: float = 0.45, 
                 nms_threshold: float = 0.55):
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.class_names = class_names
        self.letterbox_color = (114, 114, 114)
        
        # Initialize ONNX session
        self.session = self._load_model()
        if self.session:
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
        else:
            self.session = None
    
    def _load_model(self) -> Optional[ort.InferenceSession]:
        """Load ONNX model with optimizations"""
        if not os.path.exists(self.model_path):
            st.error(f"Model file not found: {self.model_path}")
            return None
        
        try:
            # Set up session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Enable available providers
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
            
            session = ort.InferenceSession(
                self.model_path, 
                sess_options=session_options,
                providers=providers
            )
            
            return session
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    def letterbox_resize(self, image: np.ndarray, new_shape: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize with aspect ratio preservation"""
        shape = image.shape[:2]  # height, width
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                 cv2.BORDER_CONSTANT, value=self.letterbox_color)
        
        return image, r, (left, top)
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for inference"""
        # Apply letterbox resize
        processed_image, ratio, pad = self.letterbox_resize(image, self.input_size)
        
        # Convert BGR to RGB
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # Normalize and convert to tensor format
        processed_image = processed_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(processed_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor)
        
        return input_tensor, ratio, pad
    
    def postprocess_detections(self, outputs: List[np.ndarray], ratio: float, 
                             pad: Tuple[int, int], original_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Post-process detection outputs"""
        predictions = outputs[0]
        
        # Handle different output formats
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        
        if len(predictions) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Extract boxes and scores
        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Filter by confidence
        valid_mask = scores >= self.confidence_threshold
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]
        
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert to original coordinates
        boxes = self._rescale_boxes(boxes, ratio, pad, original_shape)
        
        # Apply NMS
        keep_indices = self._apply_nms(boxes, scores)
        
        return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]
    
    def _rescale_boxes(self, boxes: np.ndarray, ratio: float, pad: Tuple[int, int], 
                      original_shape: Tuple[int, int]) -> np.ndarray:
        """Rescale boxes to original image coordinates"""
        if len(boxes) == 0:
            return boxes
        
        pad_x, pad_y = pad
        
        # Convert center format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        if np.max(boxes) <= 1.0:
            x_center *= self.input_size
            y_center *= self.input_size
            width *= self.input_size
            height *= self.input_size
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Remove padding and scale
        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio
        
        # Clip to image boundaries
        x1 = np.clip(x1, 0, original_shape[1])
        y1 = np.clip(y1, 0, original_shape[0])
        x2 = np.clip(x2, 0, original_shape[1])
        y2 = np.clip(y2, 0, original_shape[0])
        
        return np.column_stack([x1, y1, x2, y2])
    
    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return np.array([], dtype=int)
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self.confidence_threshold, self.nms_threshold
        )
        
        if isinstance(indices, tuple) and len(indices) > 0:
            indices = indices[0] if len(indices[0]) > 0 else np.array([])
        elif len(indices) > 0:
            indices = indices.flatten()
        else:
            indices = np.array([])
        
        return indices
    
    def draw_detections(self, image: np.ndarray, boxes: np.ndarray, 
                       scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        """Draw detection results on image"""
        if len(boxes) == 0:
            return image
        
        annotated_image = image.copy()
        
        # Color palette
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        font_scale = max(0.5, min(image.shape[0], image.shape[1]) / 1000)
        thickness = max(2, int(font_scale * 3))
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            
            valid_class_id = max(0, min(class_id, len(self.class_names) - 1))
            class_name = self.class_names[valid_class_id]
            color = colors[valid_class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name} {score:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            label_y = max(y1, label_height + 10)
            cv2.rectangle(
                annotated_image,
                (x1, label_y - label_height - baseline - 5),
                (x1 + label_width + 5, label_y + baseline),
                color, -1
            )
            
            cv2.putText(
                annotated_image, label,
                (x1 + 2, label_y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness
            )
        
        return annotated_image
    
    def detect(self, image: np.ndarray) -> dict:
        """Run detection on image"""
        if self.session is None:
            return {'error': 'Model not loaded'}
        
        original_shape = image.shape[:2]
        
        # Preprocess
        input_tensor, ratio, pad = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        
        # Post-process
        boxes, scores, class_ids = self.postprocess_detections(outputs, ratio, pad, original_shape)
        
        # Convert image for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_image = self.draw_detections(image_rgb, boxes, scores, class_ids)
        
        return {
            'original_image': image_rgb,
            'annotated_image': annotated_image,
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'detections': len(boxes),
            'inference_time': inference_time
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">🥘 Bangladesh Street Food Detector</h1>', unsafe_allow_html=True)
    
    # Configuration
    class_names = ['singara', 'peyaju', 'beguni']
    model_path = "model/best.onnx"
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, max_value=1.0, value=0.45, step=0.05
        )
        
        nms_threshold = st.slider(
            "NMS Threshold", 
            min_value=0.1, max_value=1.0, value=0.55, step=0.05
        )
        
        st.markdown("## 📊 Model Info")
        if os.path.exists(model_path):
            st.success("✅ Model loaded")
            st.info(f"🏷️ Classes: {', '.join(class_names)}")
        else:
            st.error("❌ Model not found")
            st.info("Please ensure 'model/best.onnx' exists")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to detect Bangladesh street food items"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Convert PIL to OpenCV format
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_cv = image_array
    
    with col2:
        st.markdown("### 🔍 Detection Results")
        
        if uploaded_file is not None and os.path.exists(model_path):
            if st.button("🚀 Run Detection", key="detect_btn"):
                with st.spinner("Processing..."):
                    # Initialize detector
                    detector = YOLOv12Detector(
                        model_path=model_path,
                        class_names=class_names,
                        confidence_threshold=confidence_threshold,
                        nms_threshold=nms_threshold
                    )
                    
                    # Run detection
                    results = detector.detect(image_cv)
                    
                    if 'error' not in results:
                        # Display annotated image
                        st.image(
                            results['annotated_image'], 
                            caption=f"Detected {results['detections']} objects", 
                            use_container_width=True
                        )
                        
                        # Display metrics
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        
                        with col_metric1:
                            st.metric("Detections", results['detections'])
                        
                        with col_metric2:
                            st.metric("Inference Time", f"{results['inference_time']:.3f}s")
                        
                        with col_metric3:
                            if results['detections'] > 0:
                                avg_confidence = np.mean(results['scores'])
                                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                            else:
                                st.metric("Avg Confidence", "N/A")
                        
                        # Detailed results
                        if results['detections'] > 0:
                            st.markdown("### 📋 Detection Details")
                            
                            for i, (box, score, class_id) in enumerate(zip(
                                results['boxes'], results['scores'], results['class_ids']
                            )):
                                class_name = class_names[min(class_id, len(class_names) - 1)]
                                x1, y1, x2, y2 = box.astype(int)
                                area = (x2 - x1) * (y2 - y1)
                                
                                st.markdown(f"""
                                <div class="detection-card">
                                    <strong>🏷️ {class_name.title()}</strong><br>
                                    📊 Confidence: {score:.3f}<br>
                                    📐 Box: ({x1}, {y1}) → ({x2}, {y2})<br>
                                    📏 Area: {area:,} pixels
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No objects detected. Try adjusting the confidence threshold.")
                    else:
                        st.error(f"Detection failed: {results['error']}")
        
        elif not os.path.exists(model_path):
            st.warning("Please ensure the model file exists at 'model/best.onnx'")
        else:
            st.info("Upload an image to start detection")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            🥘 Bangladesh Street Food Detector | Powered by YOLOv12 & Streamlit
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
