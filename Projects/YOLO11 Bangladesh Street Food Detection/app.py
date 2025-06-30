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
from ultralytics import YOLO

# Configure CustomTkinter appearance
ctk.set_appearance_mode("dark")  # "dark" or "light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# Fix scaling issues on Linux/Ubuntu
if platform.system() == "Linux":
    os.environ["GDK_SCALE"] = "1"
    os.environ["GDK_DPI_SCALE"] = "1"
    # Disable hardware acceleration for better rendering
    os.environ["WEBKIT_DISABLE_COMPOSITING_MODE"] = "1"


class YOLOv12BangladeshFoodDetector:
    def __init__(self):
        # Initialize main window
        self.root = ctk.CTk()
        self.root.title("YOLOv12 Bangladesh Street Food Detector")
        self.root.geometry("1400x900")  # Increased size for better visibility

        # Configure scaling for better appearance on Linux
        if platform.system() == "Linux":
            self.root.tk.call("tk", "scaling", 1.2)  # Increase UI scaling

        # Set minimum window size
        self.root.minsize(1200, 800)

        # Detection parameters
        self.CONFIDENCE_THRESHOLD = 0.25
        self.IOU_THRESHOLD = 0.45
        self.MAX_DETECTIONS = 100

        # Bangladesh Street Food Classes
        self.CLASSES = ["Singara", "Peyaju", "Puri"]
        self.CLASS_COLORS = {
            0: (0, 255, 0),  # Singara - Green
            1: (255, 165, 0),  # Peyaju - Orange
            2: (255, 0, 255),  # Puri - Magenta
        }

        # Application state
        self.model = None
        self.model_loaded = False
        self.camera_active = False
        self.cap = None
        self.current_frame = None
        self.detection_thread = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Statistics
        self.detection_stats = {cls: 0 for cls in self.CLASSES}
        self.total_detections = 0

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
            text="Load Model",
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
            from_=0.1,
            to=0.9,
            number_of_steps=80,
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
            from_=0.3,
            to=0.8,
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
            ("red", "lightcoral"),
            ("orange", "lightsalmon"),
            ("purple", "plum"),
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
            text="Load a model and start camera to begin detection",
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
            text="Ready - Load a model to start detection",
            font=ctk.CTkFont(size=12),
        )
        self.status_label.pack(side="left", padx=20, pady=10)

        # Legend
        legend_frame = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        legend_frame.pack(side="right", padx=20, pady=5)

        for i, (cls_name, color) in enumerate(
            zip(self.CLASSES, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        ):
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            legend_label = ctk.CTkLabel(
                legend_frame,
                text=f"● {cls_name}",
                text_color=color_hex,
                font=ctk.CTkFont(size=11),
            )
            legend_label.pack(side="left", padx=10)

    def auto_load_model(self):
        """Automatically load model if found in the project directory"""
        # Common model paths to check
        model_paths = [
            "yolov12_bangladesh_street_food/yolov12l_3classes_Singara_Peyaju_Puri/weights/best.pt",
            "runs/detect/train/weights/best.pt",
            "best.pt",
            "yolov12l.pt",
        ]

        for model_path in model_paths:
            if os.path.exists(model_path):
                self.load_model_from_path(model_path)
                break

    def load_model(self):
        """Load YOLOv12 model from file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select YOLOv12 Model",
            filetypes=[
                ("PyTorch Model", "*.pt"),
                ("ONNX Model", "*.onnx"),
                ("All Files", "*.*"),
            ],
        )

        if file_path:
            self.load_model_from_path(file_path)

    def load_model_from_path(self, model_path):
        """Load model from specified path"""
        try:
            self.update_status(f"Loading model: {os.path.basename(model_path)}")
            self.model = YOLO(model_path)
            self.model_loaded = True

            self.model_status_label.configure(
                text=f"✅ {os.path.basename(model_path)}", text_color="green"
            )
            self.start_camera_btn.configure(state="normal")
            self.update_status("Model loaded successfully - Ready for detection")

            # Warm up the model
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_img, verbose=False)

        except Exception as e:
            messagebox.showerror(
                "Model Loading Error", f"Failed to load model:\n{str(e)}"
            )
            self.update_status("Model loading failed")

    def start_camera(self):
        """Start camera for real-time detection"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera")
                return

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.camera_active = True
            self.start_camera_btn.configure(state="disabled")
            self.stop_camera_btn.configure(state="normal")

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

    def detection_loop(self):
        """Main detection loop running in separate thread"""
        while self.camera_active and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.current_frame = frame.copy()

                # Run YOLO detection
                if self.model_loaded and self.model:
                    results = self.model(
                        frame,
                        conf=self.CONFIDENCE_THRESHOLD,
                        iou=self.IOU_THRESHOLD,
                        max_det=self.MAX_DETECTIONS,
                        verbose=False,
                    )[0]

                    # Process detections
                    annotated_frame = self.process_detections(frame, results)
                else:
                    annotated_frame = frame

                # Update GUI
                self.update_video_display(annotated_frame)
                self.update_fps()

                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)

            except Exception as e:
                print(f"Detection error: {e}")
                break

    def process_detections(self, frame, results):
        """Process YOLO detections and draw annotations"""
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

                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box)
                    color = self.CLASS_COLORS.get(cls_id, (255, 255, 255))

                    # Draw rectangle
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label background
                    label = f"{self.CLASSES[cls_id]}: {conf:.2f}"
                    label_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )[0]
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        color,
                        -1,
                    )

                    # Draw label text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

        # Draw FPS
        cv2.putText(
            annotated_frame,
            f"FPS: {self.current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return annotated_frame

    def update_video_display(self, frame):
        """Update the video display in GUI"""
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

            # Convert to PIL Image and then to PhotoImage
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)

            # Update label in main thread
            self.root.after(0, lambda: self.video_label.configure(image=photo, text=""))
            self.root.after(
                0, lambda: setattr(self.video_label, "image", photo)
            )  # Keep reference

        except Exception as e:
            print(f"Display update error: {e}")

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

    def reset_statistics(self):
        """Reset detection statistics"""
        self.detection_stats = {cls: 0 for cls in self.CLASSES}
        self.total_detections = 0

        # Update display with enhanced formatting
        self.total_det_label.configure(text="Total Detections: 0")
        for i, cls_name in enumerate(self.CLASSES):
            self.class_labels[i].configure(text=f"{cls_name}: 0")

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


def main():
    """Main function to run the application"""
    try:
        # Check if required packages are installed
        import customtkinter
        import cv2
        import ultralytics
        from PIL import Image, ImageTk

        print("🍛 Starting YOLOv12 Bangladesh Street Food Detector...")
        print("Classes: Singara, Peyaju, Puri")

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
        app = YOLOv12BangladeshFoodDetector()
        app.run()

    except ImportError as e:
        print("❌ Missing required packages!")
        print("Please install the following packages:")
        print("pip install customtkinter opencv-python ultralytics Pillow numpy")
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
