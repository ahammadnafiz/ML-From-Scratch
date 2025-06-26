# YOLO11 Bangladesh Street Food Detection

A complete YOLO11-based object detection system for identifying Bangladesh street food items, optimized for Google Colab T4 GPU training.

## 🎯 Project Overview

This project uses YOLO11 (latest Ultralytics architecture) to detect 3 classes of Bangladesh street food:
- **Fuska** - Popular street snack
- **Singara** - Triangular fried pastry  
- **Jhalmuri** - Spicy puffed rice mixture

## 📋 Prerequisites

### Accounts Required
- **Roboflow Account** - For dataset management
- **Google Colab Pro** (recommended) - For T4 GPU access
- **Weights & Biases Account** (optional) - For experiment tracking

### Your Dataset Information
- **Project**: Bangladesh Street Food Detection
- **Workspace**: Bengali Street Food Dataset
- **API Keys**: Already configured in `.env` file

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install required packages
pip install -r requirements.txt

# For Google Colab (uncomment in script):
# !pip install ultralytics>=8.3.0 roboflow supervision wandb python-dotenv -q
```

### 2. Verify Configuration

```bash
# Run setup verification
python verify_setup.py
```

Expected output:
```
✅ REQUIRED CONFIGURATION:
   ✅ ROBOFLOW_API_KEY: hJyCBPX2... (Private API key for dataset access)
   ✅ ROBOFLOW_WORKSPACE: Bengali Street Food Dataset (Roboflow workspace name)
   ✅ ROBOFLOW_PROJECT: Bangladesh Street Food Detection (Roboflow project name)
   ✅ ROBOFLOW_VERSION: 1 (Dataset version number)

🎉 SETUP VERIFICATION SUCCESSFUL!
```

### 3. Start Training

```bash
# Run the complete training pipeline
python Bangladesh_Street_Food_Detection.py
```

Or in Google Colab:
1. Upload the script to Colab
2. Upload the `.env` file to Colab
3. Enable GPU runtime (T4 recommended)
4. Run all cells

## 📁 Project Structure

```
YOLO11 Bangladesh Street Food Detection/
├── .env                              # Environment variables (API keys, config)
├── Bangladesh_Street_Food_Detection.py  # Main training script
├── verify_setup.py                   # Setup verification script
├── requirements.txt                  # Python dependencies
├── README.md                        # This file
└── deployment_package/              # Created after training
    ├── best_yolo11_model.pt         # Trained model
    ├── inference.py                 # Production inference script
    ├── requirements.txt             # Deployment dependencies
    └── README.md                    # Usage instructions
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Required - Roboflow Configuration
ROBOFLOW_API_KEY=hJyCBPX27mCArI95l2dv
ROBOFLOW_WORKSPACE=Bengali Street Food Dataset
ROBOFLOW_PROJECT=Bangladesh Street Food Detection
ROBOFLOW_VERSION=1

# Optional - Training Configuration
TRAINING_EPOCHS=200
BATCH_SIZE=32
IMAGE_SIZE=640

# Optional - Class Selection
CLASS_1=Fuska
CLASS_2=Singara
CLASS_3=Jhalmuri

# Optional - Experiment Tracking
WANDB_API_KEY=your_wandb_key_here
```

### Customization Options

1. **Change Classes**: Modify `CLASS_1`, `CLASS_2`, `CLASS_3` in `.env`
2. **Adjust Training**: Modify `TRAINING_EPOCHS`, `BATCH_SIZE`, `IMAGE_SIZE`
3. **Model Size**: Change `model_size` in script (yolo11n, yolo11s, yolo11m)

## 🏋️ Training Process

### Expected Timeline (T4 GPU)
- **Setup & Download**: 10-15 minutes
- **Training**: 2-3 hours (200 epochs)
- **Validation**: 15-30 minutes
- **Export**: 10-15 minutes

### Training Stages
1. **Environment Setup** - Package installation, GPU verification
2. **Dataset Download** - Roboflow integration, 3-class filtering  
3. **Model Training** - YOLO11n training with T4 optimization
4. **Validation** - Performance evaluation on test set
5. **Export** - ONNX, TensorRT, CoreML format exports
6. **Deployment Package** - Production-ready inference scripts

### Expected Performance
- **mAP50**: 75-85% (target: >70%)
- **mAP50-95**: 60-75% (target: >50%)
- **Inference Speed**: 50-100 FPS on T4 GPU
- **Model Size**: ~6MB (YOLO11n)

## 📊 Monitoring Training

### Real-time Monitoring
- **Loss curves**: box_loss, cls_loss, dfl_loss should decrease
- **mAP metrics**: mAP50, mAP50-95 should increase
- **GPU memory**: Should stay under 15GB for T4

### Key Milestones
- **Epoch 50**: Losses stabilizing
- **Epoch 100**: mAP50 > 0.50
- **Epoch 150**: mAP50 > 0.70
- **Epoch 200**: Final mAP50 > 0.75

## 🚀 Deployment

After training, use the generated deployment package:

```bash
cd deployment_package

# Image detection
python inference.py --input image.jpg --output result.jpg

# Video processing
python inference.py --input video.mp4 --output result.mp4

# Batch processing
python inference.py --input folder/ --output results/
```

## 🔧 Troubleshooting

### Common Issues

#### Memory Errors
```bash
# Reduce batch size in .env
BATCH_SIZE=16  # or 8
```

#### Dataset Download Fails
```bash
# Check API credentials in .env
# Verify internet connection
# Check Roboflow project permissions
```

#### Low Performance
```bash
# Increase epochs
TRAINING_EPOCHS=300

# Try larger model
# Change model_size to 'yolo11s' in script
```

#### Colab Disconnection
```bash
# Save progress frequently
# Use Colab Pro for longer runtimes
# Enable background execution
```

## 📈 Performance Optimization

### For Better Accuracy
- Increase training epochs (300-500)
- Use larger model (yolo11s, yolo11m)
- Add more data augmentation
- Fine-tune hyperparameters

### For Faster Training
- Reduce image size (416, 512)
- Decrease batch size for memory
- Use mixed precision (already enabled)
- Cache dataset in RAM (already enabled)

### For Production Deployment
- Export to ONNX/TensorRT
- Optimize confidence thresholds
- Implement batch inference
- Use appropriate hardware

## 🆘 Support

### If You Get Stuck
1. **Run verification**: `python verify_setup.py`
2. **Check logs**: Look for error messages in output
3. **Restart runtime**: Clear cache and restart
4. **Check requirements**: Ensure all packages installed

### Expected Results
```
Final mAP50: 0.8234 (82.34%)
Final mAP50-95: 0.6789 (67.89%)
Average Inference: 12.3ms
Average FPS: 81.0
Model Size: 6.2MB
```

## 📚 Additional Resources

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [Weights & Biases Guides](https://docs.wandb.ai/)
- [Google Colab Tips](https://research.google.com/colaboratory/faq.html)

---

**🎉 Ready to train your Bangladesh Street Food Detection model!**

Your configuration is complete and ready for training. Simply run the main script and monitor the progress.
