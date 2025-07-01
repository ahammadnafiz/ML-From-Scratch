import json
import os
import platform
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import onnxruntime as ort

# Configure CustomTkinter appearance
ctk.set_appearance_mode("dark")  # "dark" or "light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# Fix scaling issues on Linux/Ubuntu
if platform.system() == "Linux":
    os.environ["GDK_SCALE"] = "1"
    os.environ["GDK_DPI_SCALE"] = "1"
    # Disable hardware acceleration for better rendering
    os.environ["WEBKIT_DISABLE_COMPOSITING_MODE"] = "1"


class BangladeshFoodDetectorONNX:
    def __init__(self):
        # Initialize main window
        self.root = ctk.CTk()
        self.root.title("Bangladesh Street Food Detector - ONNX Runtime")
        self.root.geometry("1400x900")  # Increased size for better visibility

        # Configure scaling for better appearance on Linux
        if platform.system() == "Linux":
            self.root.tk.call("tk", "scaling", 1.2)  # Increase UI scaling

        # Set minimum window size
        self.root.minsize(1200, 800)

        # Detection parameters optimized for ONNX
        self.CONFIDENCE_THRESHOLD = 0.25  # Lower for better detection
        self.IOU_THRESHOLD = 0.4  # Slightly lower for better NMS
        self.MAX_DETECTIONS = 50  # Reduced for better performance

        # Bangladesh Street Food Classes
        self.CLASSES = ["Singara", "Peyaju", "Beguni"]
        self.CLASS_COLORS = {
            0: (0, 255, 0),      # Singara - Green
            1: (255, 165, 0),    # Peyaju - Orange  
            2: (128, 0, 128),    # Beguni - Purple
        }

        # Application state
        self.ort_session = None
        self.model_loaded = False
        self.camera_active = False
        self.cap = None
        self.current_frame = None
        self.detection_thread = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.last_annotated_frame = None  # Store last frame for saving
        
        # ONNX model parameters optimized for performance
        self.INPUT_SIZE = 640
        self.input_name = None
        self.output_names = None
        # Performance optimization flags
        self.skip_frames = 0  # Skip frames counter for performance
        self.process_every_n_frames = 2  # Process every N frames

        # Statistics
        self.detection_stats = {cls: 0 for cls in self.CLASSES}
        self.total_detections = 0
        self.session_start_time = time.time()  # Track session duration

        # Initialize GUI
        self.setup_gui()

        # Auto-load model if available
        self.auto_load_model()

    def setup_gui(self):
        """Setup the GUI layout and components"""
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Left sidebar for controls
        self.setup_sidebar()

        # Main frame for video display
        self.setup_main_frame()

        # Status bar
        self.setup_status_bar()

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Bind keyboard shortcuts for better UX
        self.root.bind("<Control-q>", lambda e: self.on_closing())
        self.root.bind("<Escape>", lambda e: self.on_closing())

        # Make window focusable
        self.root.focus_set()

    def setup_sidebar(self):
        """Setup the left sidebar with controls"""
        self.sidebar_frame = ctk.CTkFrame(
            self.root, width=350, corner_radius=10
        )  # Increased width
        self.sidebar_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.sidebar_frame.grid_propagate(False)

        # Create scrollable frame for sidebar content
        self.sidebar_scroll = ctk.CTkScrollableFrame(
            self.sidebar_frame, width=330, height=750
        )
        self.sidebar_scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text="🍛 Bangladesh Street Food\nDetector",
            font=ctk.CTkFont(size=20, weight="bold"),  # Increased font size
            justify="center",
        )
        title_label.pack(pady=20)

        # Model section
        model_frame = ctk.CTkFrame(self.sidebar_scroll)
        model_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            model_frame,
            text="Model Management",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10)

        self.load_model_btn = ctk.CTkButton(
            model_frame,
            text="Load ONNX Model",
            command=self.load_model,
            height=40,  # Increased height
            font=ctk.CTkFont(size=14),
        )
        self.load_model_btn.pack(pady=5, padx=15, fill="x")

        self.model_status_label = ctk.CTkLabel(
            model_frame,
            text="No model loaded",
            text_color="red",
            font=ctk.CTkFont(size=13),  # Increased font size
        )
        self.model_status_label.pack(pady=8)

        # Camera controls
        camera_frame = ctk.CTkFrame(self.sidebar_scroll)
        camera_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            camera_frame,
            text="Camera Controls",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10)

        self.start_camera_btn = ctk.CTkButton(
            camera_frame,
            text="Start Camera",
            command=self.start_camera,
            state="disabled",
            height=40,  # Increased height
            font=ctk.CTkFont(size=14),
        )
        self.start_camera_btn.pack(pady=5, padx=15, fill="x")

        self.stop_camera_btn = ctk.CTkButton(
            camera_frame,
            text="Stop Camera",
            command=self.stop_camera,
            state="disabled",
            height=40,  # Increased height
            font=ctk.CTkFont(size=14),
        )
        self.stop_camera_btn.pack(pady=5, padx=15, fill="x")

        # Save screenshot button
        save_frame_btn = ctk.CTkButton(
            camera_frame,
            text="📸 Save Screenshot",
            command=self.save_screenshot,
            state="disabled",
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color=("blue", "darkblue"),
            hover_color=("darkblue", "blue"),
        )
        save_frame_btn.pack(pady=5, padx=15, fill="x")
        self.save_frame_btn = save_frame_btn

        # Detection parameters
        params_frame = ctk.CTkFrame(self.sidebar_scroll)
        params_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            params_frame,
            text="Detection Parameters",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10)

        # Confidence threshold
        ctk.CTkLabel(
            params_frame, text="Confidence Threshold", font=ctk.CTkFont(size=13)
        ).pack(pady=(10, 5))
        self.conf_slider = ctk.CTkSlider(
            params_frame,
            from_=0.05,
            to=0.8,
            number_of_steps=75,
            command=self.update_confidence,
            height=20,  # Increased height
        )
        self.conf_slider.set(self.CONFIDENCE_THRESHOLD)
        self.conf_slider.pack(pady=5, padx=15, fill="x")

        self.conf_label = ctk.CTkLabel(
            params_frame,
            text=f"Confidence: {self.CONFIDENCE_THRESHOLD:.2f}",
            font=ctk.CTkFont(size=13),
        )
        self.conf_label.pack(pady=5)

        # IoU threshold
        ctk.CTkLabel(
            params_frame, text="IoU Threshold", font=ctk.CTkFont(size=13)
        ).pack(pady=(10, 5))
        self.iou_slider = ctk.CTkSlider(
            params_frame,
            from_=0.2,
            to=0.7,
            number_of_steps=50,
            command=self.update_iou,
            height=20,  # Increased height
        )
        self.iou_slider.set(self.IOU_THRESHOLD)
        self.iou_slider.pack(pady=5, padx=15, fill="x")

        self.iou_label = ctk.CTkLabel(
            params_frame,
            text=f"IoU: {self.IOU_THRESHOLD:.2f}",
            font=ctk.CTkFont(size=13),
        )
        self.iou_label.pack(pady=5)

        # Processing frequency control for performance
        ctk.CTkLabel(
            params_frame, text="Processing Frequency", font=ctk.CTkFont(size=13)
        ).pack(pady=(10, 5))
        self.freq_slider = ctk.CTkSlider(
            params_frame,
            from_=1,
            to=5,
            number_of_steps=4,
            command=self.update_frequency,
            height=20,
        )
        self.freq_slider.set(self.process_every_n_frames)
        self.freq_slider.pack(pady=5, padx=15, fill="x")

        self.freq_label = ctk.CTkLabel(
            params_frame,
            text=f"Process every {self.process_every_n_frames} frames",
            font=ctk.CTkFont(size=13),
        )
        self.freq_label.pack(pady=5)

        # Statistics - Enhanced visibility
        stats_frame = ctk.CTkFrame(
            self.sidebar_scroll, border_width=2, border_color="gray"
        )
        stats_frame.pack(fill="x", padx=10, pady=15)

        # Statistics header with background color
        stats_header = ctk.CTkFrame(stats_frame, fg_color=("gray70", "gray30"))
        stats_header.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            stats_header,
            text="📊 Detection Statistics",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=("black", "white"),
        ).pack(pady=12)

        # Statistics content frame
        stats_content = ctk.CTkFrame(stats_frame, fg_color="transparent")
        stats_content.pack(fill="x", padx=10, pady=10)

        # FPS and Total detections with better spacing
        fps_frame = ctk.CTkFrame(stats_content, fg_color=("gray85", "gray25"))
        fps_frame.pack(fill="x", pady=5)

        self.fps_label = ctk.CTkLabel(
            fps_frame,
            text="FPS: 0",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=("blue", "lightblue"),
        )
        self.fps_label.pack(pady=8)

        total_frame = ctk.CTkFrame(stats_content, fg_color=("gray85", "gray25"))
        total_frame.pack(fill="x", pady=5)

        self.total_det_label = ctk.CTkLabel(
            total_frame,
            text="Total Detections: 0",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=("green", "lightgreen"),
        )
        self.total_det_label.pack(pady=8)

        # Class-specific counters with enhanced visibility
        class_header = ctk.CTkLabel(
            stats_content,
            text="Detection Counts by Class:",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        class_header.pack(pady=(15, 10))

        self.class_labels = {}
        class_colors = [
            ("green", "lightgreen"),     # Singara
            ("orange", "lightsalmon"),   # Peyaju  
            ("purple", "plum"),          # Beguni
        ]

        for i, cls_name in enumerate(self.CLASSES):
            class_frame = ctk.CTkFrame(stats_content, fg_color=("gray85", "gray25"))
            class_frame.pack(fill="x", pady=3)

            label = ctk.CTkLabel(
                class_frame,
                text=f"{cls_name}: 0",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=class_colors[i % len(class_colors)],
            )
            label.pack(pady=6)
            self.class_labels[i] = label

        # Reset button with enhanced styling
        reset_btn = ctk.CTkButton(
            stats_content,
            text="🔄 Reset Statistics",
            command=self.reset_statistics,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("red", "darkred"),
            hover_color=("darkred", "red"),
        )
        reset_btn.pack(pady=15, padx=10, fill="x")

        # Exit button
        exit_btn = ctk.CTkButton(
            stats_content,
            text="🚪 Exit Application",
            command=self.on_closing,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("gray", "gray20"),
            hover_color=("darkgray", "gray10"),
        )
        exit_btn.pack(pady=10, padx=10, fill="x")

    def setup_main_frame(self):
        """Setup the main video display frame"""
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Video display
        self.video_label = ctk.CTkLabel(
            self.main_frame,
            text="Load an ONNX model and start camera to begin detection",
            font=ctk.CTkFont(size=16),
            width=800,
            height=600,
        )
        self.video_label.pack(expand=True, fill="both", padx=20, pady=20)

    def setup_status_bar(self):
        """Setup the bottom status bar"""
        self.status_frame = ctk.CTkFrame(self.root, height=40, corner_radius=5)
        self.status_frame.grid(
            row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew"
        )

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready - Load an ONNX model to start detection",
            font=ctk.CTkFont(size=12),
        )
        self.status_label.pack(side="left", padx=20, pady=10)

        # Legend
        legend_frame = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        legend_frame.pack(side="right", padx=20, pady=5)

        legend_colors = [
            ("#00FF00", "Singara"),      # Green
            ("#FFA500", "Peyaju"),       # Orange
            ("#800080", "Beguni"),       # Purple
        ]

        for color_hex, cls_name in legend_colors:
            legend_label = ctk.CTkLabel(
                legend_frame,
                text=f"● {cls_name}",
                text_color=color_hex,
                font=ctk.CTkFont(size=11, weight="bold"),
            )
            legend_label.pack(side="left", padx=10)

    def auto_load_model(self):
        """Automatically load model if found in the project directory"""
        # Common model paths to check for ONNX models
        model_paths = [
            "yolov12_bangladesh_street_food/yolov12l_3classes_Singara_Peyaju_Beguni/weights/best.onnx",
            "runs/detect/train/weights/best.onnx",
            "runs/detect/train2/weights/best.onnx", 
            "runs/detect/train3/weights/best.onnx",
            "best.onnx",
            "yolov12l_bangladesh_food.onnx",
            "yolov12l.onnx",
            "model/best.onnx",
        ]

        for model_path in model_paths:
            if os.path.exists(model_path):
                self.load_model_from_path(model_path)
                break

    def load_model(self):
        """Load ONNX model from file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select ONNX Model",
            filetypes=[
                ("ONNX Model", "*.onnx"),
                ("All Files", "*.*"),
            ],
        )

        if file_path:
            self.load_model_from_path(file_path)

    def load_model_from_path(self, model_path):
        """Load ONNX model from specified path"""
        try:
            self.update_status(f"Loading ONNX model: {os.path.basename(model_path)}")
            
            # Load ONNX model with performance optimizations
            providers = ['CPUExecutionProvider']
            # Try to use GPU if available
            if ort.get_available_providers():
                available = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                elif 'TensorrtExecutionProvider' in available:
                    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 4  # Optimize for multi-threading
            sess_options.intra_op_num_threads = 4
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.ort_session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
            
            # Get model input/output info
            self.input_name = self.ort_session.get_inputs()[0].name
            input_shape = self.ort_session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.ort_session.get_outputs()]
            
            # Update input size if available from model
            if len(input_shape) >= 3:
                self.INPUT_SIZE = input_shape[2]  # Usually [batch, channels, height, width]
            
            self.model_loaded = True

            self.model_status_label.configure(
                text=f"✅ {os.path.basename(model_path)}", text_color="green"
            )
            self.start_camera_btn.configure(state="normal")
            self.update_status("ONNX model loaded successfully - Ready for detection")

            print(f"✅ ONNX model loaded successfully")
            print(f"Input shape: {input_shape}")
            print(f"Input name: {self.input_name}")
            print(f"Output names: {self.output_names}")

        except Exception as e:
            error_msg = str(e)
            print(f"ONNX model loading error: {error_msg}")
            messagebox.showerror(
                "Model Loading Error", 
                f"Failed to load ONNX model:\n{error_msg}\n\nMake sure you have an ONNX model file."
            )
            self.update_status("ONNX model loading failed")
            self.model_loaded = False

    def start_camera(self):
        """Start camera for real-time detection"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera")
                return

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency

            self.camera_active = True
            self.start_camera_btn.configure(state="disabled")
            self.stop_camera_btn.configure(state="normal")
            self.save_frame_btn.configure(state="normal")  # Enable screenshot button
            self.save_frame_btn.configure(state="normal")

            # Start detection thread
            self.detection_thread = threading.Thread(
                target=self.detection_loop, daemon=True
            )
            self.detection_thread.start()

            self.update_status("Camera started - Real-time detection active")

        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera:\n{str(e)}")

    def stop_camera(self):
        """Stop camera and detection"""
        try:
            self.camera_active = False

            # Wait a moment for the detection thread to finish
            time.sleep(0.1)

            if self.cap:
                self.cap.release()
                self.cap = None

            self.start_camera_btn.configure(state="normal")
            self.stop_camera_btn.configure(state="disabled")
            self.save_frame_btn.configure(state="disabled")

            # Clear video display
            self.video_label.configure(image=None, text="Camera stopped")
            self.update_status("Camera stopped")

        except Exception as e:
            print(f"Error stopping camera: {e}")
            # Force reset the state even if there's an error
            self.camera_active = False
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            self.start_camera_btn.configure(state="normal")
            self.stop_camera_btn.configure(state="disabled")
            self.save_frame_btn.configure(state="disabled")

    def draw_info_overlay(self, frame):
        """Draw enhanced FPS and detection info overlay on the frame"""
        try:
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Create overlay background
            overlay_height = 100
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (0, 0), (width, overlay_height), (255, 255, 255), 2)
            
            # FPS info
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detection count
            detection_text = f"Total Detections: {self.total_detections}"
            cv2.putText(frame, detection_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Model status
            model_text = "ONNX Model: Loaded" if self.model_loaded else "ONNX Model: Not Loaded"
            cv2.putText(frame, model_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Class counts (right side)
            x_offset = width - 300
            y_start = 25
            for i, cls_name in enumerate(self.CLASSES):
                count = self.detection_stats.get(cls_name, 0)
                color = self.CLASS_COLORS.get(i, (255, 255, 255))
                class_text = f"{cls_name}: {count}"
                cv2.putText(frame, class_text, (x_offset, y_start + i * 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        except Exception as e:
            # If overlay fails, just continue without it
            pass

    def detection_loop(self):
        """Optimized detection loop running in separate thread"""
        print("Starting optimized detection loop...")
        frame_count = 0
        last_detection_result = None
        last_detection_time = time.time()
        
        while self.camera_active and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                self.current_frame = frame.copy()
                frame_count += 1
                current_time = time.time()

                # Performance optimization: process every N frames for detection
                should_process = (frame_count % self.process_every_n_frames == 0)
                
                # Also add time-based processing to ensure we don't process too frequently
                time_since_last_detection = current_time - last_detection_time
                if should_process and time_since_last_detection > 0.05:  # Minimum 50ms between detections
                    # Run ONNX detection with error handling
                    if self.model_loaded and self.ort_session:
                        try:
                            # Preprocess image for ONNX
                            input_tensor, scale, pad_x, pad_y = self.preprocess_image_onnx(frame, self.INPUT_SIZE)
                            
                            # Run ONNX inference
                            outputs = self.ort_session.run(self.output_names, {self.input_name: input_tensor})
                            
                            # Post-process detections
                            boxes, confidences, class_ids = self.postprocess_detections_onnx(
                                outputs, scale, pad_x, pad_y, self.CONFIDENCE_THRESHOLD, self.IOU_THRESHOLD
                            )

                            # Store detection result for reuse
                            last_detection_result = (boxes, confidences, class_ids)
                            last_detection_time = current_time
                            
                            # Update statistics only for new detections
                            if len(boxes) > 0:
                                for cls_id in class_ids:
                                    cls_id = int(cls_id) % len(self.CLASSES) if len(self.CLASSES) > 0 else 0
                                    if cls_id < len(self.CLASSES):
                                        self.detection_stats[self.CLASSES[cls_id]] += 1
                                        self.total_detections += 1
                            
                        except Exception as detection_error:
                            print(f"ONNX detection error: {detection_error}")
                            last_detection_result = (np.array([]), np.array([]), np.array([]))

                # Use last detection result or empty if no detection yet
                if last_detection_result is not None:
                    boxes, confidences, class_ids = last_detection_result
                    # Draw annotations (fast version)
                    annotated_frame = self.draw_detections_fast(frame, boxes, confidences, class_ids)
                elif self.model_loaded and self.ort_session:
                    # Show frame with processing indicator
                    annotated_frame = frame.copy()
                else:
                    # No model loaded
                    annotated_frame = frame.copy()
                    
                    # Add "no model" overlay
                    cv2.putText(annotated_frame, "No ONNX Model Loaded - Raw Camera Feed", 
                              (10, annotated_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Always draw info overlay
                self.draw_info_overlay(annotated_frame)
                self.last_annotated_frame = annotated_frame.copy()

                # Update GUI - less frequently for better performance
                if frame_count % 1 == 0:  # Update display every frame for smooth experience
                    self.update_video_display(annotated_frame)
                
                # Update FPS counter
                self.update_fps()

                # Controlled delay for stable FPS - reduced for better performance
                time.sleep(0.01)  # ~60 FPS max

            except Exception as e:
                print(f"Detection loop error: {e}")
                # Don't break immediately, try to continue
                time.sleep(0.1)
                continue
        
        print("Detection loop ended")

    def process_detections(self, frame, results):
        """Process YOLO detections and draw enhanced annotations"""
        annotated_frame = frame.copy()

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if cls_id < len(self.CLASSES):
                    # Update statistics
                    self.detection_stats[self.CLASSES[cls_id]] += 1
                    self.total_detections += 1

                    # Get coordinates and color
                    x1, y1, x2, y2 = map(int, box)
                    color = self.CLASS_COLORS.get(cls_id, (255, 255, 255))
                    
                    # Calculate box dimensions for adaptive text sizing
                    box_width = x2 - x1
                    box_height = y2 - y1
                    
                    # Draw enhanced bounding box with thickness based on confidence
                    thickness = max(2, int(conf * 4))  # Thicker box for higher confidence
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add corner markers for better visibility
                    corner_length = min(20, box_width // 8, box_height // 8)
                    # Top-left corner
                    cv2.line(annotated_frame, (x1, y1), (x1 + corner_length, y1), color, thickness + 1)
                    cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_length), color, thickness + 1)
                    # Top-right corner  
                    cv2.line(annotated_frame, (x2, y1), (x2 - corner_length, y1), color, thickness + 1)
                    cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_length), color, thickness + 1)
                    # Bottom-left corner
                    cv2.line(annotated_frame, (x1, y2), (x1 + corner_length, y2), color, thickness + 1)
                    cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_length), color, thickness + 1)
                    # Bottom-right corner
                    cv2.line(annotated_frame, (x2, y2), (x2 - corner_length, y2), color, thickness + 1)
                    cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_length), color, thickness + 1)

                    # Create enhanced label with class name and confidence percentage
                    class_name = self.CLASSES[cls_id]
                    confidence_pct = conf * 100
                    label = f"{class_name}: {confidence_pct:.1f}%"
                    
                    # Adaptive font scale based on box size
                    font_scale = min(0.8, max(0.4, (box_width + box_height) / 800))
                    font_thickness = max(1, int(font_scale * 2))
                    
                    # Get label dimensions
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    label_width, label_height = label_size
                    
                    # Position label (above box if there's space, otherwise inside)
                    label_y = y1 - 10 if y1 - label_height - 10 > 0 else y1 + label_height + 10
                    label_x = x1
                    
                    # Ensure label doesn't go outside frame boundaries
                    if label_x + label_width > annotated_frame.shape[1]:
                        label_x = annotated_frame.shape[1] - label_width - 5
                    if label_x < 0:
                        label_x = 5
                        
                    # Draw label background with rounded corners effect
                    bg_padding = 8
                    bg_x1 = max(0, label_x - bg_padding)
                    bg_y1 = max(0, label_y - label_height - bg_padding)
                    bg_x2 = min(annotated_frame.shape[1], label_x + label_width + bg_padding)
                    bg_y2 = min(annotated_frame.shape[0], label_y + bg_padding)
                    
                    # Create semi-transparent background
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                    cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0, annotated_frame)
                    
                    # Draw border around label background
                    cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
                    
                    # Draw label text with shadow effect for better readability
                    # Shadow
                    cv2.putText(
                        annotated_frame,
                        label,
                        (label_x + 1, label_y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),  # Black shadow
                        font_thickness + 1,
                    )
                    # Main text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),  # White text
                        font_thickness,
                    )
                    
                    # Add confidence indicator bar
                    if conf > 0.5:  # Only show for reasonably confident detections
                        bar_width = int((box_width * 0.8) * conf)
                        bar_height = 4
                        bar_x = x1 + (box_width - int(box_width * 0.8)) // 2
                        bar_y = y2 - 15
                        
                        # Background bar
                        cv2.rectangle(annotated_frame, 
                                    (bar_x, bar_y), 
                                    (bar_x + int(box_width * 0.8), bar_y + bar_height), 
                                    (64, 64, 64), -1)
                        # Confidence bar
                        cv2.rectangle(annotated_frame, 
                                    (bar_x, bar_y), 
                                    (bar_x + bar_width, bar_y + bar_height), 
                                    color, -1)

        # Draw enhanced FPS and detection info overlay
        self.draw_info_overlay(annotated_frame)
        
        return annotated_frame
        
    def draw_detections_fast(self, frame, boxes, confidences, class_ids):
        """Fast detection drawing for better performance"""
        annotated_frame = frame.copy()

        if len(boxes) > 0:
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                # Ensure class_id is within valid range
                cls_id = int(cls_id) % len(self.CLASSES) if len(self.CLASSES) > 0 else 0
                
                if cls_id < len(self.CLASSES):
                    # Get coordinates and color
                    x1, y1, x2, y2 = map(int, box)
                    color = self.CLASS_COLORS.get(cls_id, (255, 255, 255))
                    
                    # Draw simple bounding box
                    thickness = 2
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Simple label
                    class_name = self.CLASSES[cls_id]
                    confidence_pct = conf * 100
                    label = f"{class_name}: {confidence_pct:.0f}%"
                    
                    # Simple text rendering
                    font_scale = 0.6
                    font_thickness = 2
                    
                    # Get label dimensions
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    
                    # Position label
                    label_y = y1 - 10 if y1 - label_height - 10 > 0 else y1 + label_height + 10
                    label_x = x1
                    
                    # Draw simple label background
                    cv2.rectangle(annotated_frame, 
                                (label_x - 2, label_y - label_height - 2), 
                                (label_x + label_width + 2, label_y + 2), 
                                color, -1)
                    
                    # Draw label text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),  # White text
                        font_thickness,
                    )

        return annotated_frame

    def update_video_display(self, frame):
        """Update the video display in GUI with CustomTkinter compatible image"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame to fit display
            height, width = frame_rgb.shape[:2]
            max_width, max_height = 800, 600

            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))

            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            
            # Create CTkImage instead of PhotoImage for better compatibility
            ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)

            # Update label in main thread
            self.root.after(0, lambda: self.video_label.configure(image=ctk_image, text=""))
            self.root.after(0, lambda: setattr(self.video_label, "image", ctk_image))  # Keep reference

        except Exception as e:
            print(f"Display update error: {e}")
            # Fallback to text display if image update fails
            self.root.after(0, lambda: self.video_label.configure(image=None, text="Video display error"))

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time

            # Update GUI labels with enhanced formatting
            self.root.after(
                0, lambda: self.fps_label.configure(text=f"FPS: {self.current_fps:.1f}")
            )
            self.root.after(
                0,
                lambda: self.total_det_label.configure(
                    text=f"Total Detections: {self.total_detections}"
                ),
            )

            # Update class-specific counters with better formatting
            for i, cls_name in enumerate(self.CLASSES):
                count = self.detection_stats[cls_name]
                self.root.after(
                    0,
                    lambda i=i, cls=cls_name, count=count: self.class_labels[
                        i
                    ].configure(text=f"{cls}: {count}"),
                )

    def update_confidence(self, value):
        """Update confidence threshold"""
        self.CONFIDENCE_THRESHOLD = float(value)
        self.conf_label.configure(text=f"Confidence: {self.CONFIDENCE_THRESHOLD:.2f}")

    def update_iou(self, value):
        """Update IoU threshold"""
        self.IOU_THRESHOLD = float(value)
        self.iou_label.configure(text=f"IoU: {self.IOU_THRESHOLD:.2f}")

    def update_frequency(self, value):
        """Update processing frequency"""
        self.process_every_n_frames = int(float(value))
        self.freq_label.configure(text=f"Process every {self.process_every_n_frames} frames")

    def save_screenshot(self):
        """Save current detection frame as screenshot"""
        if self.last_annotated_frame is not None:
            try:
                # Create screenshots directory if it doesn't exist
                screenshots_dir = "screenshots"
                os.makedirs(screenshots_dir, exist_ok=True)
                
                # Generate filename with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"bangladesh_food_detection_{timestamp}.jpg"
                filepath = os.path.join(screenshots_dir, filename)
                
                # Save the frame
                cv2.imwrite(filepath, self.last_annotated_frame)
                
                # Update status
                self.update_status(f"Screenshot saved: {filename}")
                
                # Show success message
                messagebox.showinfo("Screenshot Saved", 
                                  f"Detection screenshot saved as:\n{filepath}")
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save screenshot:\n{str(e)}")
                self.update_status("Screenshot save failed")
        else:
            messagebox.showwarning("No Frame", "No detection frame available to save")

    def reset_statistics(self):
        """Reset detection statistics"""
        self.detection_stats = {cls: 0 for cls in self.CLASSES}
        self.total_detections = 0
        self.session_start_time = time.time()  # Reset session timer

        # Update display with enhanced formatting
        self.total_det_label.configure(text="Total Detections: 0")
        for i, cls_name in enumerate(self.CLASSES):
            self.class_labels[i].configure(text=f"{cls_name}: 0")
        
        self.update_status("Statistics reset - New detection session started")

    def update_status(self, message):
        """Update status bar message"""
        self.status_label.configure(text=message)

    def on_closing(self):
        """Handle application closing"""
        try:
            # Stop camera if active
            if self.camera_active:
                self.stop_camera()

            # Release camera resources
            if self.cap:
                self.cap.release()

            # Try to destroy OpenCV windows (may not work on headless systems)
            try:
                cv2.destroyAllWindows()
            except (cv2.error, AttributeError):
                # Ignore OpenCV window destruction errors on headless systems
                pass

            # Destroy the main window
            self.root.quit()  # Stop the mainloop
            self.root.destroy()  # Destroy the window

        except Exception as e:
            print(f"Error during application closing: {e}")
            # Force exit if there's any issue
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass

    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
            self.on_closing()
        except Exception as e:
            print(f"Application error: {e}")
            self.on_closing()

    def preprocess_image_onnx(self, image, input_size=640):
        """Preprocess image for ONNX model inference - optimized for performance"""
        # Get original dimensions
        original_height, original_width = image.shape[:2]
        
        # Resize image while maintaining aspect ratio
        scale = min(input_size / original_width, input_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Use faster interpolation for better performance
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded_image = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (input_size - new_width) // 2
        pad_y = (input_size - new_height) // 2
        
        # Place resized image in center
        padded_image[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized_image
        
        # Convert to RGB and normalize (optimized)
        padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        padded_image = padded_image.astype(np.float32) / 255.0
        
        # Convert to NCHW format (batch, channels, height, width)
        input_tensor = np.transpose(padded_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Ensure contiguous array for better performance
        input_tensor = np.ascontiguousarray(input_tensor)
        
        return input_tensor, scale, pad_x, pad_y

    def postprocess_detections_onnx(self, outputs, scale, pad_x, pad_y, conf_threshold=0.25, nms_threshold=0.4):
        """Post-process ONNX model outputs - optimized for performance"""
        # Get predictions
        predictions = outputs[0]  # Shape: [1, num_detections, 5 + num_classes] or [1, 7, 8400]
        
        # Handle different output formats
        if len(predictions.shape) == 3 and predictions.shape[1] > predictions.shape[2]:
            # Format: [1, num_detections, features] - transpose to [1, features, num_detections]
            predictions = np.transpose(predictions, (0, 2, 1))
        
        # Remove batch dimension: [features, num_detections]
        predictions = predictions[0]
        
        # For YOLOv11/v12: predictions format is [num_features, num_detections]
        # Transpose to get [num_detections, num_features]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        
        # Extract boxes and scores
        boxes = predictions[:, :4]  # x_center, y_center, width, height
        
        # For YOLOv11, the format might be: [x, y, w, h, class0_conf, class1_conf, class2_conf]
        if predictions.shape[1] == 7:  # 4 bbox + 3 classes
            class_scores = predictions[:, 4:]  # All class confidences
        else:
            # Traditional format: [x, y, w, h, objectness, class0, class1, ...]
            class_scores = predictions[:, 5:]  # class probabilities
        
        # Get class with highest probability for each detection
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)
        
        # Early filtering by confidence threshold for performance
        valid_mask = confidences > conf_threshold
        if not np.any(valid_mask):
            return np.array([]), np.array([]), np.array([])
            
        boxes = boxes[valid_mask]
        confidences = confidences[valid_mask]
        class_ids = class_ids[valid_mask]
        
        # Convert from center format to corner format (vectorized)
        x_centers = boxes[:, 0]
        y_centers = boxes[:, 1]
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        
        half_widths = widths / 2
        half_heights = heights / 2
        
        x1 = x_centers - half_widths
        y1 = y_centers - half_heights
        x2 = x_centers + half_widths
        y2 = y_centers + half_heights
        
        # Check if coordinates are normalized (vectorized)
        if np.max(x2) <= 1.0:  # Normalized coordinates
            x1 *= self.INPUT_SIZE
            y1 *= self.INPUT_SIZE
            x2 *= self.INPUT_SIZE
            y2 *= self.INPUT_SIZE
        
        # Adjust for padding and scale (vectorized)
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        
        # Ensure coordinates are within image bounds (vectorized)
        x1 = np.maximum(x1, 0)
        y1 = np.maximum(y1, 0)
        x2 = np.maximum(x2, 0)
        y2 = np.maximum(y2, 0)
        
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
            # Limit number of detections for performance
            if len(indices) > self.MAX_DETECTIONS:
                # Sort by confidence and keep top detections
                sorted_indices = indices[np.argsort(confidences[indices])[::-1]]
                indices = sorted_indices[:self.MAX_DETECTIONS]
            
            return boxes_xyxy[indices], confidences[indices], class_ids[indices]
        else:
            return np.array([]), np.array([]), np.array([])

    def process_detections_onnx(self, frame, boxes, confidences, class_ids):
        """Process ONNX detections and draw enhanced annotations"""
        annotated_frame = frame.copy()

        if len(boxes) > 0:
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                # Ensure class_id is within valid range
                cls_id = int(cls_id) % len(self.CLASSES) if len(self.CLASSES) > 0 else 0
                
                if cls_id < len(self.CLASSES):
                    # Update statistics
                    self.detection_stats[self.CLASSES[cls_id]] += 1
                    self.total_detections += 1

                    # Get coordinates and color
                    x1, y1, x2, y2 = map(int, box)
                    color = self.CLASS_COLORS.get(cls_id, (255, 255, 255))
                    
                    # Calculate box dimensions for adaptive text sizing
                    box_width = x2 - x1
                    box_height = y2 - y1
                    
                    # Draw enhanced bounding box with thickness based on confidence
                    thickness = max(2, int(conf * 4))  # Thicker box for higher confidence
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add corner markers for better visibility
                    corner_length = min(20, box_width // 8, box_height // 8)
                    # Top-left corner
                    cv2.line(annotated_frame, (x1, y1), (x1 + corner_length, y1), color, thickness + 1)
                    cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_length), color, thickness + 1)
                    # Top-right corner  
                    cv2.line(annotated_frame, (x2, y1), (x2 - corner_length, y1), color, thickness + 1)
                    cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_length), color, thickness + 1)
                    # Bottom-left corner
                    cv2.line(annotated_frame, (x1, y2), (x1 + corner_length, y2), color, thickness + 1)
                    cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_length), color, thickness + 1)
                    # Bottom-right corner
                    cv2.line(annotated_frame, (x2, y2), (x2 - corner_length, y2), color, thickness + 1)
                    cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_length), color, thickness + 1)

                    # Create enhanced label with class name and confidence percentage
                    class_name = self.CLASSES[cls_id]
                    confidence_pct = conf * 100
                    label = f"{class_name}: {confidence_pct:.1f}%"
                    
                    # Adaptive font scale based on box size
                    font_scale = min(0.8, max(0.4, (box_width + box_height) / 800))
                    font_thickness = max(1, int(font_scale * 2))
                    
                    # Get label dimensions
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    label_width, label_height = label_size
                    
                    # Position label (above box if there's space, otherwise inside)
                    label_y = y1 - 10 if y1 - label_height - 10 > 0 else y1 + label_height + 10
                    label_x = x1
                    
                    # Ensure label doesn't go outside frame boundaries
                    if label_x + label_width > annotated_frame.shape[1]:
                        label_x = annotated_frame.shape[1] - label_width - 5
                    if label_x < 0:
                        label_x = 5
                        
                    # Draw label background with rounded corners effect
                    bg_padding = 8
                    bg_x1 = max(0, label_x - bg_padding)
                    bg_y1 = max(0, label_y - label_height - bg_padding)
                    bg_x2 = min(annotated_frame.shape[1], label_x + label_width + bg_padding)
                    bg_y2 = min(annotated_frame.shape[0], label_y + bg_padding)
                    
                    # Create semi-transparent background
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                    cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0, annotated_frame)
                    
                    # Draw border around label background
                    cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
                    
                    # Draw label text with shadow effect for better readability
                    # Shadow
                    cv2.putText(
                        annotated_frame,
                        label,
                        (label_x + 1, label_y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),  # Black shadow
                        font_thickness + 1,
                    )
                    # Main text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),  # White text
                        font_thickness,
                    )
                    
                    # Add confidence indicator bar
                    if conf > 0.5:  # Only show for reasonably confident detections
                        bar_width = int((box_width * 0.8) * conf)
                        bar_height = 4
                        bar_x = x1 + (box_width - int(box_width * 0.8)) // 2
                        bar_y = y2 - 15
                        
                        # Background bar
                        cv2.rectangle(annotated_frame, 
                                    (bar_x, bar_y), 
                                    (bar_x + int(box_width * 0.8), bar_y + bar_height), 
                                    (64, 64, 64), -1)
                        # Confidence bar
                        cv2.rectangle(annotated_frame, 
                                    (bar_x, bar_y), 
                                    (bar_x + bar_width, bar_y + bar_height), 
                                    color, -1)

        # Draw enhanced FPS and detection info overlay
        self.draw_info_overlay(annotated_frame)
        
        return annotated_frame

def main():
    """Main function to run the application"""
    try:
        # Check if required packages are installed
        import customtkinter
        import cv2
        import onnxruntime
        from PIL import Image, ImageTk

        print("🍛 Starting ONNX Bangladesh Street Food Detector...")
        print("Classes: Singara, Peyaju, Beguni")
        print("📋 Target Food Items:")
        print("   🔺 Singara  - Traditional triangular fried pastry")
        print("   🧅 Peyaju   - Onion fritters with spices") 
        print("   🍆 Beguni   - Eggplant fritters in gram flour batter")

        # Check OpenCV build info for Ubuntu compatibility
        if platform.system() == "Linux":
            build_info = cv2.getBuildInformation()
            if "GTK" not in build_info and "Qt" not in build_info:
                print("⚠️  Warning: OpenCV may have limited GUI support on this system")
                print(
                    "   If you encounter issues, consider installing: libgtk2.0-dev pkg-config"
                )

        print("=" * 50)

        # Create and run application
        app = BangladeshFoodDetectorONNX()
        app.run()

    except ImportError as e:
        print("❌ Missing required packages!")
        print("Please install the following packages:")
        print("pip install customtkinter opencv-python onnxruntime Pillow numpy")
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✋ Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()