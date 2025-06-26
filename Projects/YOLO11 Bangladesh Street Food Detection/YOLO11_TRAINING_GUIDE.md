# YOLO11 Bangladesh Street Food Detection - Step-by-Step Training Guide

## 🎯 Overview
This guide walks you through training a YOLO11 model for detecting 3 classes of Bangladesh street food using Google Colab with T4 GPU optimization.

---

## 📋 Prerequisites

### Required Accounts & Setup
1. **Google Account** - For Google Colab access
2. **Roboflow Account** - For dataset management
3. **Weights & Biases (W&B) Account** (Optional) - For experiment tracking

### Hardware Requirements
- **Recommended**: Google Colab Pro with T4 GPU
- **Minimum**: Google Colab with GPU runtime
- **Local**: NVIDIA GPU with 8GB+ VRAM

---

## 🚀 Step-by-Step Implementation Guide

### **STEP 1: Environment Setup**

#### 1.1 Open Google Colab
- Go to [Google Colab](https://colab.research.google.com/)
- Create a new notebook or upload the `test.py` file

#### 1.2 Enable GPU Runtime
```python
# In Colab: Runtime → Change runtime type → Hardware accelerator: GPU (T4)
```

#### 1.3 Install Required Packages
```python
# Uncomment and run these lines in Colab:
!pip install ultralytics>=8.3.0 roboflow supervision wandb -q
!pip install torch torchvision torchaudio --index-url https://pytorch.org/whl/cu118 -q
```

#### 1.4 Verify Installation
```python
# The script will automatically check:
# - CUDA availability
# - GPU type
# - YOLO11 model availability
```

**Expected Output:**
```
CUDA available: True
GPU: Tesla T4
CUDA version: 11.8
✅ YOLO11 models available
```

---

### **STEP 2: Dataset Configuration**

#### 2.1 Create Roboflow Account
1. Go to [Roboflow](https://roboflow.com/)
2. Sign up for a free account
3. Create a new project for Bangladesh Street Food

#### 2.2 Get API Credentials
1. Go to Roboflow Settings → API
2. Copy your API key
3. Note your workspace name and project name

#### 2.3 Update Configuration
```python
# Replace these values in the script:
RF_API_KEY = "your_actual_api_key_here"
RF_WORKSPACE = "your_workspace_name"
RF_PROJECT = "your_project_name"
RF_VERSION = 1  # Your dataset version number
```

#### 2.4 Select Your 3 Classes
```python
# Choose 3 classes from the available 15:
SELECTED_CLASSES = [
    'Fuska',        # Popular street food
    'Singara',      # Triangular shape
    'Jhalmuri'      # Different texture
]
```

**Available Classes:**
- Tiler Khaja, Kotkoti, Jhalmuri, Peyaju, Beguni
- Singara, Papor Vaja, Vel Puri, Chopti, Fuska
- Vorta, Murobba, Dim Cake, Halim, Muglai Parata

---

### **STEP 3: Dataset Download & Processing**

#### 3.1 Download Dataset
```python
# The script automatically downloads from Roboflow
# Expected output: "Dataset downloaded to: /path/to/dataset"
```

#### 3.2 Filter to 3 Classes
```python
# Automatic filtering process:
# - Creates filtered_dataset_3_classes directory
# - Remaps class indices (0, 1, 2)
# - Shows statistics for each split
```

**Expected Output:**
```
Original classes: ['Tiler Khaja', 'Kotkoti', ...]
Selected class indices: [9, 5, 2]
Processing train split...
  Kept 150/200 images from train split
```

---

### **STEP 4: Model Configuration**

#### 4.1 Training Configuration
The script uses optimized settings for T4 GPU:
```python
TRAINING_CONFIG = {
    'model_size': 'yolo11n',      # Nano model for efficiency
    'epochs': 200,                # Training epochs
    'batch_size': 32,             # Batch size for T4
    'imgsz': 640,                # Image size
    # ... other parameters
}
```

#### 4.2 Augmentation Settings
```python
AUGMENTATION_CONFIG = {
    'hsv_h': 0.015,              # Color augmentation
    'hsv_s': 0.7,
    'mosaic': 1.0,               # Mosaic augmentation
    'mixup': 0.15,               # MixUp augmentation
    # ... other parameters
}
```

---

### **STEP 5: Initialize Experiment Tracking (Optional)**

#### 5.1 Setup Weights & Biases
1. Create account at [wandb.ai](https://wandb.ai)
2. Get API key from settings
3. Login in Colab:
```python
import wandb
wandb.login()  # Enter your API key when prompted
```

#### 5.2 W&B Integration
```python
# The script automatically initializes W&B with:
# - Project name: bangladesh-street-food-yolo11-3class
# - Experiment tracking
# - Metric logging
```

---

### **STEP 6: Start Training**

#### 6.1 Memory Optimization
```python
# Automatic GPU memory optimization for T4:
# - Sets memory fraction to 90%
# - Clears cache
# - Shows memory usage
```

#### 6.2 Begin Training Process
```python
# Training starts automatically with:
print("STARTING YOLO11 TRAINING")
model, results = train_yolo11_model(dataset_path, TRAINING_CONFIG)
```

**Expected Training Output:**
```
Starting YOLO11 training...
Model: yolo11n
Classes: ['Fuska', 'Singara', 'Jhalmuri']
Epochs: 200
Batch size: 32

Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/200      2.1G     1.234     0.567     1.123        123        640
  2/200      2.1G     1.180     0.534     1.089        119        640
...
```

#### 6.3 Monitor Training Progress
- **Loss curves**: box_loss, cls_loss, dfl_loss decreasing
- **Metrics**: mAP50, mAP50-95 increasing
- **Memory usage**: Should stay within T4 limits
- **ETA**: Training time estimation

---

### **STEP 7: Model Validation**

#### 7.1 Automatic Validation
```python
# After training completes:
validation_results = validate_yolo11_model(best_model_path, dataset_path)
```

**Expected Output:**
```
YOLO11 Validation Results:
mAP50: 0.8234
mAP50-95: 0.6789
Fuska mAP50-95: 0.7123
Singara mAP50-95: 0.6834
Jhalmuri mAP50-95: 0.6410
```

#### 7.2 Performance Metrics
- **mAP50**: Mean Average Precision at IoU 0.5
- **mAP50-95**: Mean Average Precision across IoU 0.5-0.95
- **Per-class metrics**: Individual class performance

---

### **STEP 8: Test Inference**

#### 8.1 Sample Image Testing
```python
# Automatic testing on sample images:
test_yolo11_inference(best_model_path, test_image)
```

**Expected Output:**
```
Testing YOLO11 inference on: /path/to/test_image.jpg
Detected 3 objects:
Class: Fuska, Confidence: 0.89
Class: Singara, Confidence: 0.76
Class: Jhalmuri, Confidence: 0.82
```

---

### **STEP 9: Model Export & Optimization**

#### 9.1 Export to Multiple Formats
```python
# Automatic export to:
# - ONNX (cross-platform inference)
# - TensorRT (NVIDIA GPU optimization)
# - CoreML (iOS deployment)
```

**Expected Output:**
```
YOLO11 model exported to ONNX format: /path/to/best.onnx
YOLO11 model exported to TensorRT format: /path/to/best.engine
YOLO11 model exported to CoreML format: /path/to/best.mlmodel
```

---

### **STEP 10: Performance Benchmarking**

#### 10.1 Speed Benchmarking
```python
# Automatic benchmarking on test images:
benchmark_results = benchmark_yolo11_model(best_model_path, test_images_path)
```

**Expected Output:**
```
YOLO11 BENCHMARK RESULTS
Total images processed: 50
Average inference time: 12.34 ms
Average FPS: 81.02
Total detections: 127
Average detections per image: 2.54
```

#### 10.2 Model Comparison
```python
# Compare YOLO11 vs YOLOv8:
model_comparison = compare_yolo_models(dataset_path)
```

**Expected Output:**
```
MODEL COMPARISON SUMMARY
mAP50 Improvement: +5.67%
mAP50-95 Improvement: +3.42%
Precision Improvement: +2.18%
Recall Improvement: +4.23%
```

---

### **STEP 11: Deployment Package Creation**

#### 11.1 Automatic Package Generation
```python
# Creates complete deployment package:
deployment_package_path = prepare_deployment_package(best_model_path)
```

**Package Contents:**
- `best_yolo11_model.pt` - Trained model
- `inference.py` - Production inference script
- `requirements.txt` - Dependencies
- `README.md` - Usage instructions

#### 11.2 Production Inference Script
```python
# Ready-to-use inference with:
# - Image detection
# - Video processing
# - Batch processing
# - Command-line interface
```

---

### **STEP 12: Final Results & Cleanup**

#### 12.1 Training Summary
```python
# Comprehensive final report showing:
generate_final_summary()
```

**Expected Summary:**
```
YOLO11 BANGLADESH STREET FOOD DETECTION - FINAL SUMMARY
📊 EXPERIMENT CONFIGURATION:
   Model Architecture: yolo11n
   Selected Classes: ['Fuska', 'Singara', 'Jhalmuri']
   Total Classes: 3
   Training Epochs: 200
   Batch Size: 32
   Image Size: 640
   GPU: T4

📁 OUTPUT FILES:
   Training Results: street_food_yolo11_3class/yolo11n_Fuska_Singara_Jhalmuri
   Best Model: .../weights/best.pt
   Last Model: .../weights/last.pt

🎯 PERFORMANCE SUMMARY:
   mAP50: 0.8234
   mAP50-95: 0.6789
   Average Inference Time: 12.3ms
   Average FPS: 81.0
   Avg Detections/Image: 2.5

🚀 DEPLOYMENT:
   Deployment Package: deployment_package
   Ready for production deployment!

✅ TRAINING COMPLETED SUCCESSFULLY!
```

---

## 🔧 Troubleshooting Guide

### Common Issues & Solutions

#### **Issue 1: CUDA Out of Memory**
```python
# Solutions:
# - Reduce batch_size from 32 to 16 or 8
# - Reduce imgsz from 640 to 416
# - Use mixed precision (already enabled)
```

#### **Issue 2: Roboflow Download Fails**
```python
# Solutions:
# 1. Check API key and credentials
# 2. Verify internet connection
# 3. Use manual dataset upload:
dataset_path = "/content/your-dataset-folder"
```

#### **Issue 3: Low mAP Scores**
```python
# Solutions:
# 1. Increase training epochs
# 2. Adjust learning rate
# 3. Check dataset quality
# 4. Increase data augmentation
```

#### **Issue 4: W&B Connection Issues**
```python
# Solutions:
# 1. Skip W&B by setting wandb_enabled = False
# 2. Check internet connection
# 3. Login again: wandb.login()
```

---

## 📊 Expected Results Timeline

### **Training Phase (2-4 hours on T4 GPU)**
- **0-30min**: Environment setup & dataset download
- **30min-3h**: Model training (200 epochs)
- **3-3.5h**: Validation & testing
- **3.5-4h**: Export & benchmarking

### **Performance Expectations**
- **mAP50**: 75-85% (good performance)
- **mAP50-95**: 60-75% (excellent for 3-class)
- **Inference Speed**: 50-100 FPS on T4 GPU
- **Model Size**: ~6MB (YOLO11n)

---

## 🎯 Success Criteria

### **Training Success Indicators:**
✅ Training completes without errors  
✅ mAP50 > 0.70 (70%)  
✅ mAP50-95 > 0.50 (50%)  
✅ All 3 classes detected in test images  
✅ Inference speed > 30 FPS  
✅ Deployment package created successfully  

### **Next Steps After Training:**
1. **Test with new images** - Upload your own street food images
2. **Fine-tune parameters** - Adjust thresholds for your use case
3. **Deploy to production** - Use the deployment package
4. **Scale up** - Train on more classes or larger dataset

---

## 💡 Tips for Best Results

### **Data Quality:**
- Ensure good lighting in training images
- Include diverse angles and backgrounds
- Balance the number of objects per class
- Remove poor quality or mislabeled images

### **Training Optimization:**
- Monitor training curves for overfitting
- Use early stopping if validation mAP plateaus
- Experiment with different augmentation settings
- Try different model sizes (yolo11s, yolo11m) if you have more compute

### **Production Deployment:**
- Test thoroughly with real-world images
- Set appropriate confidence thresholds
- Implement proper error handling
- Monitor performance in production

---

## 📚 Additional Resources

- **YOLO11 Documentation**: [Ultralytics Docs](https://docs.ultralytics.com/)
- **Roboflow Tutorials**: [Roboflow Learn](https://blog.roboflow.com/)
- **W&B Guides**: [Weights & Biases Docs](https://docs.wandb.ai/)
- **Google Colab Tips**: [Colab FAQ](https://research.google.com/colaboratory/faq.html)

---

**🎉 Congratulations! You now have a complete guide to train your YOLO11 Bangladesh Street Food Detection model!**
