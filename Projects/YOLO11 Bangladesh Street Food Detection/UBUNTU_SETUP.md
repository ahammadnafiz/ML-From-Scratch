# Ubuntu Setup Guide for YOLO Bangladesh Street Food Detector

## System Requirements
- Ubuntu 18.04+ or similar Linux distribution
- Python 3.8+
- Webcam/Camera device

## Installation Steps

### 1. Install System Dependencies (Required for GUI support)
```bash
# Update package list
sudo apt update

# Install GTK development libraries (fixes OpenCV GUI issues)
sudo apt install -y libgtk2.0-dev pkg-config

# Install additional GUI libraries for better CustomTkinter rendering
sudo apt install -y python3-tk python3-pil python3-pil.imagetk

# Install camera and video libraries
sudo apt install -y libv4l-dev v4l-utils

# Install font libraries for better text rendering
sudo apt install -y fontconfig fonts-dejavu-core

# Optional: Install additional fonts for better appearance
sudo apt install -y fonts-liberation fonts-roboto
```

### 2. Python Environment Setup
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 3. Fix OpenCV Installation (if needed)
If you still encounter OpenCV GUI errors, reinstall OpenCV with GUI support:
```bash
# Uninstall current OpenCV
pip uninstall opencv-python opencv-contrib-python

# Install OpenCV with GUI support
pip install opencv-contrib-python-headless==4.8.1.78
# OR for full GUI support (larger download):
pip install opencv-python==4.8.1.78
```

### 4. Font and DPI Configuration
Add these lines to your `~/.bashrc` or `~/.profile`:
```bash
# Improve GUI scaling and font rendering
export GDK_SCALE=1
export GDK_DPI_SCALE=1
export QT_AUTO_SCREEN_SCALE_FACTOR=1
```

Then reload:
```bash
source ~/.bashrc
```

### 5. Test Camera Access
```bash
# Test camera access
v4l2-ctl --list-devices

# Test camera functionality
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Pixelated/Blurry GUI
- Install additional fonts: `sudo apt install fonts-liberation`
- Adjust system scaling in Ubuntu Settings > Displays
- Use the scaling settings in the app initialization

#### 2. Camera Access Issues
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again, or use:
newgrp video
```

#### 3. OpenCV destroyAllWindows Error
This is already handled in the app code, but if you see warnings:
```bash
# Install full GUI support
sudo apt install libgtk-3-dev
```

#### 4. CustomTkinter Appearance Issues
- Try switching between "dark" and "light" themes
- Adjust the scaling factor in the app settings
- Install additional theme packages:
```bash
sudo apt install gtk2-engines-murrine gtk2-engines-pixbuf
```

## Performance Optimization

### For better performance on Ubuntu:
1. Close unnecessary applications
2. Ensure good lighting for camera
3. Adjust detection thresholds (confidence/IoU) for your needs
4. Use a lower resolution camera setting if needed

## Running the Application
```bash
# Navigate to project directory
cd /path/to/your/project

# Activate virtual environment
source venv/bin/activate

# Run the application
python3 app.py
```

## Model Files
Place your trained YOLO model (.pt file) in one of these locations:
- `yolov12_bangladesh_street_food/yolov12l_3classes_Singara_Peyaju_Puri/weights/best.pt`
- `runs/detect/train/weights/best.pt`
- `best.pt` (in project root)
- Or use the "Load Model" button in the GUI

The app will automatically detect and load the model on startup.
