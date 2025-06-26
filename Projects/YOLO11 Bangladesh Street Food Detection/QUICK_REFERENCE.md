# YOLO11 Training - Quick Reference Checklist

## 🚀 Pre-Training Checklist

### **Environment Setup**
- [ ] Open Google Colab with GPU runtime (T4 recommended)
- [ ] Install packages: `ultralytics`, `roboflow`, `wandb`
- [ ] Verify CUDA and GPU availability
- [ ] Check YOLO11 model download

### **Dataset Preparation**
- [ ] Create Roboflow account and project
- [ ] Get API credentials (API key, workspace, project name)
- [ ] Update script with your Roboflow credentials
- [ ] Select your 3 target classes from available 15
- [ ] Verify dataset download and filtering

### **Configuration Review**
- [ ] Confirm training config (epochs: 200, batch: 32, imgsz: 640)
- [ ] Review augmentation settings for your data
- [ ] Set up W&B tracking (optional but recommended)

---

## 🏃‍♂️ During Training Checklist

### **Monitor Training Progress**
- [ ] Watch loss curves (box_loss, cls_loss, dfl_loss should decrease)
- [ ] Check mAP metrics (should increase over time)
- [ ] Monitor GPU memory usage (should stay < 15GB for T4)
- [ ] Verify all 3 classes are being detected

### **Expected Milestones**
- [ ] **Epoch 50**: Losses start stabilizing
- [ ] **Epoch 100**: mAP50 > 0.50
- [ ] **Epoch 150**: mAP50 > 0.70
- [ ] **Epoch 200**: Final mAP50 > 0.75

---

## ✅ Post-Training Validation

### **Model Performance**
- [ ] mAP50 ≥ 0.70 (Good: 0.80+, Excellent: 0.85+)
- [ ] mAP50-95 ≥ 0.50 (Good: 0.60+, Excellent: 0.70+)
- [ ] Per-class mAP reasonable for all 3 classes
- [ ] Inference speed > 30 FPS (Target: 50-100 FPS)

### **Quality Checks**
- [ ] Test inference on sample images
- [ ] Verify detections look correct
- [ ] Check confidence scores are reasonable (> 0.25)
- [ ] No major class confusion in results

### **Export & Deployment**
- [ ] Model exported to ONNX successfully
- [ ] TensorRT export completed (optional)
- [ ] Deployment package created with all files
- [ ] README and inference script generated

---

## 🎯 Success Metrics

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| mAP50 | < 0.60 | 0.60-0.70 | 0.70-0.80 | > 0.80 |
| mAP50-95 | < 0.40 | 0.40-0.50 | 0.50-0.65 | > 0.65 |
| Inference Speed | < 20 FPS | 20-40 FPS | 40-80 FPS | > 80 FPS |
| Model Size | > 50MB | 20-50MB | 10-20MB | < 10MB |

---

## 🔧 Quick Fixes

### **If Training Fails:**
- Reduce batch_size: 32 → 16 → 8
- Reduce image size: 640 → 512 → 416
- Check dataset paths and credentials
- Restart runtime and clear cache

### **If Performance is Poor:**
- Increase epochs: 200 → 300
- Adjust learning rate: 0.01 → 0.005
- Check data quality and labeling
- Try different model size: yolo11n → yolo11s

### **If Memory Issues:**
- Set batch_size = 8
- Enable half precision (already enabled)
- Clear cache: `torch.cuda.empty_cache()`
- Restart Colab runtime

---

## 📋 File Outputs to Expect

```
street_food_yolo11_3class/
├── yolo11n_Fuska_Singara_Jhalmuri/
│   ├── weights/
│   │   ├── best.pt          # Best model weights
│   │   ├── last.pt          # Last epoch weights
│   │   ├── best.onnx        # ONNX export
│   │   └── best.engine      # TensorRT export
│   ├── results.png          # Training curves
│   ├── confusion_matrix.png # Confusion matrix
│   └── val_batch0_pred.jpg  # Validation predictions

deployment_package/
├── best_yolo11_model.pt     # Production model
├── inference.py             # Inference script
├── requirements.txt         # Dependencies
└── README.md               # Usage guide
```

---

## ⏰ Time Estimates

- **Setup**: 10-15 minutes
- **Dataset Download**: 5-10 minutes
- **Training**: 2-3 hours (200 epochs on T4)
- **Validation & Export**: 15-30 minutes
- **Total**: 3-4 hours

---

## 🆘 Emergency Contacts

### **If Stuck:**
1. Check the full training guide: `YOLO11_TRAINING_GUIDE.md`
2. Review error messages carefully
3. Check Ultralytics documentation
4. Restart Colab runtime as last resort

### **Common Error Solutions:**
- **"RuntimeError: CUDA out of memory"** → Reduce batch_size
- **"FileNotFoundError"** → Check dataset paths
- **"ModuleNotFoundError"** → Reinstall packages
- **"Roboflow API Error"** → Check credentials

---

**✨ Remember: The complete training process is automated - just update your credentials and run the script!**
