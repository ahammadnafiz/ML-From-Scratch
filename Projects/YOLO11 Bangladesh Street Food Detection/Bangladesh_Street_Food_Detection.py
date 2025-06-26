# YOLO11 3-Class Bangladesh Street Food Object Detection Training
# Optimized for Google Colab T4 GPU with latest YOLO11 architecture

# ============================================================================
# SECTION 1: ENVIRONMENT SETUP
# ============================================================================

# Install required packages with YOLO11 support
# !pip install ultralytics>=8.3.0 roboflow supervision wandb python-dotenv -q
# !pip install torch torchvision torchaudio --index-url https://pytorch.org/whl/cu118 -q

import os
import torch
import yaml
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO
from roboflow import Roboflow
import wandb
from IPython.display import Image, display
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
print("✅ Environment variables loaded from .env file")

# Check GPU availability and YOLO11 compatibility
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"CUDA version: {torch.version.cuda}")

# Verify YOLO11 availability
try:
    model_test = YOLO('yolo11n.pt')
    print("✅ YOLO11 models available")
    del model_test
except Exception as e:
    print(f"⚠️ YOLO11 not available, falling back to YOLOv8: {e}")

# ============================================================================
# SECTION 2: DATASET CONFIGURATION AND DOWNLOAD
# ============================================================================

# Load Roboflow credentials from environment variables
RF_API_KEY = os.getenv('ROBOFLOW_API_KEY')
RF_WORKSPACE = os.getenv('ROBOFLOW_WORKSPACE')
RF_PROJECT = os.getenv('ROBOFLOW_PROJECT')
RF_VERSION = int(os.getenv('ROBOFLOW_VERSION', 1))

# Verify credentials are loaded
if not RF_API_KEY:
    print("❌ ROBOFLOW_API_KEY not found in .env file!")
    print("Please check your .env file contains: ROBOFLOW_API_KEY=your_api_key")
else:
    print(f"✅ Roboflow API Key loaded: {RF_API_KEY[:8]}...")

if not RF_WORKSPACE:
    print("❌ ROBOFLOW_WORKSPACE not found in .env file!")
else:
    print(f"✅ Workspace: {RF_WORKSPACE}")

if not RF_PROJECT:
    print("❌ ROBOFLOW_PROJECT not found in .env file!")
else:
    print(f"✅ Project: {RF_PROJECT}")

print(f"✅ Dataset Version: {RF_VERSION}")

# ============================================================================
# SELECT YOUR 3 CLASSES FROM 15 BANGLADESH STREET FOODS
# ============================================================================
# Available classes from your dataset:
# ['Tiler Khaja', 'Kotkoti', 'Jhalmuri', 'Peyaju', 'Beguni', 'Singara', 
#  'Papor Vaja', 'Vel Puri', 'Chopti', 'Fuska', 'Vorta', 'Murobba', 
#  'Dim Cake', 'Halim', 'Muglai Parata']

# Load selected classes from environment variables (with fallback defaults)
SELECTED_CLASSES = [
    os.getenv('CLASS_1', 'Fuska'),        # Popular street food - good for detection
    os.getenv('CLASS_2', 'Singara'),      # Distinct triangular shape
    os.getenv('CLASS_3', 'Jhalmuri')      # Different texture and appearance
]

print(f"Selected classes for training: {SELECTED_CLASSES}")
print(f"Total classes: {len(SELECTED_CLASSES)}")

# Initialize Roboflow and download dataset
def download_roboflow_dataset():
    """Download dataset from Roboflow"""
    try:
        rf = Roboflow(api_key=RF_API_KEY)
        project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
        dataset = project.version(RF_VERSION).download("yolov8")  # Compatible with YOLO11
        return dataset.location
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

# Download dataset
dataset_path = download_roboflow_dataset()
if dataset_path:
    print(f"Dataset downloaded to: {dataset_path}")
else:
    print("Please manually download your dataset or check Roboflow credentials")
    dataset_path = "/content/your-dataset-folder"  # Update this path

# ============================================================================
# SECTION 3: DATASET FILTERING AND OPTIMIZATION FOR 3 CLASSES
# ============================================================================

def filter_dataset_for_3_classes(dataset_path, selected_classes):
    """
    Filter the 15-class dataset to keep only 3 selected classes
    This creates a new dataset structure with only the selected classes
    """
    import shutil
    from collections import defaultdict
    
    # Create filtered dataset directory
    filtered_dataset_path = "/content/filtered_dataset_3_classes"
    if os.path.exists(filtered_dataset_path):
        shutil.rmtree(filtered_dataset_path)
    os.makedirs(filtered_dataset_path)
    
    # Read original data.yaml
    original_yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(original_yaml_path, 'r') as file:
        original_data = yaml.safe_load(file)
    
    original_classes = original_data['names']
    print(f"Original classes: {original_classes}")
    print(f"Total original classes: {len(original_classes)}")
    
    # Create mapping from original class names to indices
    original_class_to_idx = {name: idx for idx, name in enumerate(original_classes)}
    selected_class_indices = [original_class_to_idx[cls] for cls in selected_classes if cls in original_class_to_idx]
    
    print(f"Selected class indices: {selected_class_indices}")
    
    # Create new class mapping (0, 1, 2 for the 3 selected classes)
    new_class_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_class_indices)}
    
    # Statistics
    stats = defaultdict(lambda: defaultdict(int))
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Create directories
        images_src = os.path.join(dataset_path, split, 'images')
        labels_src = os.path.join(dataset_path, split, 'labels')
        
        images_dst = os.path.join(filtered_dataset_path, split, 'images')
        labels_dst = os.path.join(filtered_dataset_path, split, 'labels')
        
        os.makedirs(images_dst, exist_ok=True)
        os.makedirs(labels_dst, exist_ok=True)
        
        if not os.path.exists(images_src) or not os.path.exists(labels_src):
            print(f"  {split} split not found, skipping...")
            continue
        
        # Get all label files
        label_files = [f for f in os.listdir(labels_src) if f.endswith('.txt')]
        
        kept_images = 0
        total_images = len(label_files)
        
        for label_file in label_files:
            label_path = os.path.join(labels_src, label_file)
            image_file = label_file.replace('.txt', '.jpg')  # Assuming jpg format
            if not os.path.exists(os.path.join(images_src, image_file)):
                image_file = label_file.replace('.txt', '.jpeg')
            if not os.path.exists(os.path.join(images_src, image_file)):
                image_file = label_file.replace('.txt', '.png')
            
            image_path = os.path.join(images_src, image_file)
            
            if not os.path.exists(image_path):
                continue
            
            # Read and filter labels
            new_labels = []
            has_selected_class = False
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id in selected_class_indices:
                        # Update class ID to new mapping (0, 1, 2)
                        new_class_id = new_class_mapping[class_id]
                        new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                        new_labels.append(new_line)
                        has_selected_class = True
                        stats[split][selected_classes[new_class_id]] += 1
            
            # Only keep images that have at least one selected class
            if has_selected_class:
                # Copy image
                shutil.copy2(image_path, os.path.join(images_dst, image_file))
                
                # Write filtered labels
                with open(os.path.join(labels_dst, label_file), 'w') as f:
                    f.writelines(new_labels)
                
                kept_images += 1
        
        print(f"  Kept {kept_images}/{total_images} images from {split} split")
    
    # Create new data.yaml for filtered dataset
    new_data_yaml = {
        'path': filtered_dataset_path,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(selected_classes),
        'names': selected_classes
    }
    
    with open(os.path.join(filtered_dataset_path, 'data.yaml'), 'w') as f:
        yaml.dump(new_data_yaml, f, default_flow_style=False)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("FILTERED DATASET STATISTICS")
    print(f"{'='*60}")
    for split in stats:
        print(f"\n{split.upper()} split:")
        total_objects = sum(stats[split].values())
        for class_name, count in stats[split].items():
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"  {class_name}: {count} objects ({percentage:.1f}%)")
        print(f"  Total objects: {total_objects}")
    
    return filtered_dataset_path, stats

def analyze_dataset(dataset_path):
    """Analyze dataset for optimization"""
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        print("Dataset Analysis:")
        print(f"Classes: {data.get('names', [])}")
        print(f"Number of classes: {data.get('nc', 'Unknown')}")
        
        # Count images in each split
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(dataset_path, split, 'images')
            if os.path.exists(split_path):
                img_count = len([f for f in os.listdir(split_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"{split.capitalize()} images: {img_count}")
        
        return data
    else:
        print("data.yaml not found. Please check dataset path.")
        return None

# Filter dataset to 3 classes and analyze
if dataset_path and os.path.exists(dataset_path):
    print("Filtering dataset to 3 selected classes...")
    filtered_dataset_path, dataset_stats = filter_dataset_for_3_classes(dataset_path, SELECTED_CLASSES)
    
    # Update dataset path to use filtered dataset
    dataset_path = filtered_dataset_path
    
    # Analyze the filtered dataset
    dataset_info = analyze_dataset(dataset_path)
else:
    print("Original dataset not found. Please check your dataset setup.")

# ============================================================================
# SECTION 4: YOLO11 MODEL CONFIGURATION FOR T4 GPU OPTIMIZATION
# ============================================================================

# YOLO11 optimized training configuration for T4 GPU with 3 classes
TRAINING_CONFIG = {
    'model_size': 'yolo11n',      # YOLO11 nano model for T4 GPU efficiency
    'epochs': int(os.getenv('TRAINING_EPOCHS', 200)),                # Load from .env or default to 200
    'batch_size': int(os.getenv('BATCH_SIZE', 32)),             # Load from .env or default to 32
    'imgsz': int(os.getenv('IMAGE_SIZE', 640)),                # Load from .env or default to 640
    'patience': 25,               # Increased patience for YOLO11 training
    'save_period': 20,            # Save checkpoint every 20 epochs
    'workers': 4,                 # Increased workers for YOLO11
    'device': 0,                  # GPU device
    'project': 'street_food_yolo11_3class',
    'name': f'yolo11n_{SELECTED_CLASSES[0]}_{SELECTED_CLASSES[1]}_{SELECTED_CLASSES[2]}',
    'exist_ok': True,
    'pretrained': True,
    'optimizer': 'auto',          # YOLO11's automatic optimizer selection
    'lr0': 0.01,                 # Initial learning rate
    'lrf': 0.001,                # Lower final learning rate for fine-tuning
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,          # YOLO11 needs less warmup
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,                  # Box loss gain
    'cls': 1.5,                  # Increased class loss gain for 3 classes
    'dfl': 1.5,                  # DFL loss gain
    'label_smoothing': 0.1,      # Label smoothing for better generalization
    'nbs': 64,                   # Nominal batch size
    'overlap_mask': True,
    'mask_ratio': 4,
    'dropout': 0.0,
    'val': True,
    'plots': True,
    'save_json': False,          # Disable to save space
    'save_hybrid': False,        # Disable to save space
    'conf': 0.001,               # Lower confidence for validation
    'iou': 0.7,                  # Higher IoU threshold for YOLO11
    'max_det': 300,              # Maximum detections per image
    'half': True,                # Use mixed precision training
    'cache': 'ram',              # Cache images in RAM for faster training
    'rect': False,               # Disable rectangular training for stability
    'cos_lr': True,              # Use cosine learning rate scheduler
    'close_mosaic': 10,          # Close mosaic augmentation earlier in YOLO11
    'resume': False,
    'amp': True,                 # Automatic Mixed Precision
    'fraction': 1.0,             # Use full dataset
    'profile': False,
    'freeze': None,              # Freeze layers (None = don't freeze)
    'multi_scale': True,         # Enable multi-scale training
    'copy_paste': 0.3,           # Increased copy-paste for YOLO11
    'auto_augment': 'randaugment', # YOLO11's enhanced augmentation
    'erasing': 0.4,              # Random erasing probability
    'crop_fraction': 1.0,        # Crop fraction for training
}

# YOLO11 enhanced data augmentation settings
AUGMENTATION_CONFIG = {
    'hsv_h': 0.015,              # Hue augmentation (±1.5%)
    'hsv_s': 0.7,                # Saturation augmentation (±70%)
    'hsv_v': 0.4,                # Value augmentation (±40%)
    'degrees': 0.0,              # Rotation (disabled for food items)
    'translate': 0.1,            # Translation (±10%)
    'scale': 0.5,                # Scale (±50%)
    'shear': 0.0,                # Shear (disabled for stability)
    'perspective': 0.0,          # Perspective transformation
    'flipud': 0.0,               # Vertical flip probability
    'fliplr': 0.5,               # Horizontal flip probability
    'mosaic': 1.0,               # Mosaic augmentation probability
    'mixup': 0.15,               # Increased MixUp for YOLO11
    'copy_paste': 0.3,           # Copy-paste augmentation
}

# Display configuration summary
print(f"\n{'='*60}")
print("🔧 CONFIGURATION SUMMARY")
print(f"{'='*60}")
print(f"📊 Dataset Configuration:")
print(f"   Workspace: {RF_WORKSPACE}")
print(f"   Project: {RF_PROJECT}")
print(f"   Version: {RF_VERSION}")
print(f"   Selected Classes: {SELECTED_CLASSES}")
print(f"\n🏋️ Training Configuration:")
print(f"   Epochs: {TRAINING_CONFIG['epochs']}")
print(f"   Batch Size: {TRAINING_CONFIG['batch_size']}")
print(f"   Image Size: {TRAINING_CONFIG['imgsz']}")
print(f"   Model: {TRAINING_CONFIG['model_size']}")
print(f"{'='*60}")

# ============================================================================
# SECTION 5: INITIALIZE WANDB FOR EXPERIMENT TRACKING
# ============================================================================

def setup_wandb():
    """Initialize Weights & Biases for experiment tracking"""
    try:
        # Check if W&B API key is provided in environment
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if wandb_api_key:
            print(f"✅ W&B API Key found: {wandb_api_key[:8]}...")
        
        wandb.init(
            project="bangladesh-street-food-yolo11-3class",
            name=f"yolo11n_{SELECTED_CLASSES[0]}_{SELECTED_CLASSES[1]}_{SELECTED_CLASSES[2]}_experiment",
            config={
                **TRAINING_CONFIG,
                **AUGMENTATION_CONFIG,
                'gpu': 'T4',
                'framework': 'YOLO11',
                'dataset': 'Bangladesh Street Food (3 classes)',
                'classes': SELECTED_CLASSES,
                'total_classes': len(SELECTED_CLASSES),
                'experiment_type': '3_class_subset_yolo11',
                'model_architecture': 'YOLO11n',
                'roboflow_project': RF_PROJECT,
                'roboflow_workspace': RF_WORKSPACE
            }
        )
        print("✅ W&B initialized successfully for YOLO11 experiment")
        return True
    except Exception as e:
        print(f"⚠️ W&B initialization failed: {e}")
        print("💡 Tip: Add WANDB_API_KEY to your .env file for experiment tracking")
        return False

# Initialize W&B (optional)
wandb_enabled = setup_wandb()

# ============================================================================
# SECTION 6: YOLO11 MODEL INITIALIZATION AND TRAINING
# ============================================================================

def train_yolo11_model(dataset_path, config):
    """Train YOLO11 model with optimized settings"""
    
    # Initialize YOLO11 model
    try:
        model = YOLO(f"{config['model_size']}.pt")
        print(f"✅ Successfully loaded {config['model_size']} model")
    except Exception as e:
        print(f"⚠️ YOLO11 not available, falling back to YOLOv8n: {e}")
        model = YOLO("yolov8n.pt")
        config['model_size'] = 'yolov8n'
    
    # Update config with augmentation settings
    full_config = {**config, **AUGMENTATION_CONFIG}
    
    # Start training
    print("Starting YOLO11 training...")
    print(f"Model: {config['model_size']}")
    print(f"Classes: {SELECTED_CLASSES}")
    print(f"Epochs: {full_config['epochs']}")
    print(f"Batch size: {full_config['batch_size']}")
    
    results = model.train(
        data=os.path.join(dataset_path, "data.yaml"),
        epochs=full_config['epochs'],
        batch=full_config['batch_size'],
        imgsz=full_config['imgsz'],
        patience=full_config['patience'],
        save_period=full_config['save_period'],
        workers=full_config['workers'],
        device=full_config['device'],
        project=full_config['project'],
        name=full_config['name'],
        exist_ok=full_config['exist_ok'],
        pretrained=full_config['pretrained'],
        optimizer=full_config['optimizer'],
        lr0=full_config['lr0'],
        lrf=full_config['lrf'],
        momentum=full_config['momentum'],
        weight_decay=full_config['weight_decay'],
        warmup_epochs=full_config['warmup_epochs'],
        warmup_momentum=full_config['warmup_momentum'],
        warmup_bias_lr=full_config['warmup_bias_lr'],
        box=full_config['box'],
        cls=full_config['cls'],
        dfl=full_config['dfl'],
        nbs=full_config['nbs'],
        overlap_mask=full_config['overlap_mask'],
        mask_ratio=full_config['mask_ratio'],
        dropout=full_config['dropout'],
        val=full_config['val'],
        plots=full_config['plots'],
        save_json=full_config['save_json'],
        save_hybrid=full_config['save_hybrid'],
        conf=full_config['conf'],
        iou=full_config['iou'],
        max_det=full_config['max_det'],
        half=full_config['half'],
        cache=full_config['cache'],
        rect=full_config['rect'],
        cos_lr=full_config['cos_lr'],
        close_mosaic=full_config['close_mosaic'],
        resume=full_config['resume'],
        amp=full_config['amp'],
        fraction=full_config['fraction'],
        profile=full_config['profile'],
        freeze=full_config['freeze'],
        # Augmentation parameters
        hsv_h=full_config['hsv_h'],
        hsv_s=full_config['hsv_s'],
        hsv_v=full_config['hsv_v'],
        degrees=full_config['degrees'],
        translate=full_config['translate'],
        scale=full_config['scale'],
        shear=full_config['shear'],
        perspective=full_config['perspective'],
        flipud=full_config['flipud'],
        fliplr=full_config['fliplr'],
        mosaic=full_config['mosaic'],
        mixup=full_config['mixup'],
        copy_paste=full_config['copy_paste'],
        auto_augment=full_config.get('auto_augment'),
        erasing=full_config['erasing'],
    )
    
    return model, results

# ============================================================================
# SECTION 7: MEMORY OPTIMIZATION FOR T4 GPU WITH YOLO11
# ============================================================================

def optimize_memory():
    """Optimize memory usage for T4 GPU with YOLO11"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before training: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        # Set memory fraction for YOLO11
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory

# Optimize memory before training
optimize_memory()

# ============================================================================
# SECTION 8: START YOLO11 TRAINING
# ============================================================================

if dataset_path and os.path.exists(dataset_path):
    print("=" * 60)
    print("STARTING YOLO11 TRAINING")
    print("=" * 60)
    
    # Train the model
    model, results = train_yolo11_model(dataset_path, TRAINING_CONFIG)
    
    print("=" * 60)
    print("YOLO11 TRAINING COMPLETED")
    print("=" * 60)
    
    # Print training results
    print(f"Best weights saved at: {results}")
    
else:
    print("Dataset path not found. Please check your dataset setup.")

# ============================================================================
# SECTION 9: YOLO11 MODEL VALIDATION AND TESTING
# ============================================================================

def validate_yolo11_model(model_path, dataset_path):
    """Validate the trained YOLO11 model"""
    model = YOLO(model_path)
    
    # Validate on test set with YOLO11 optimized settings
    results = model.val(
        data=os.path.join(dataset_path, "data.yaml"),
        imgsz=640,
        batch=16,           # Higher batch size for YOLO11 validation
        conf=0.25,
        iou=0.7,            # Higher IoU for YOLO11
        device=0,
        half=True,
        plots=True,
        save_json=True,     # Save detailed results
        save_hybrid=False,
        split='test'
    )
    
    print("YOLO11 Validation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    # Print per-class metrics
    if hasattr(results.box, 'maps'):
        for i, class_name in enumerate(SELECTED_CLASSES):
            if i < len(results.box.maps):
                print(f"{class_name} mAP50-95: {results.box.maps[i]:.4f}")
    
    return results

# Validate the trained YOLO11 model
best_model_path = f"{TRAINING_CONFIG['project']}/{TRAINING_CONFIG['name']}/weights/best.pt"
if os.path.exists(best_model_path):
    validation_results = validate_yolo11_model(best_model_path, dataset_path)

# ============================================================================
# SECTION 10: YOLO11 INFERENCE AND VISUALIZATION
# ============================================================================

def test_yolo11_inference(model_path, test_image_path):
    """Test YOLO11 inference on a sample image"""
    model = YOLO(model_path)
    
    # Run inference with YOLO11 optimized settings
    results = model(
        test_image_path, 
        conf=0.25, 
        iou=0.7,
        agnostic_nms=False,
        max_det=300,
        classes=None,
        half=True,
        device=0
    )
    
    # Display results
    for r in results:
        # Plot results
        im_array = r.plot(
            conf=True,
            labels=True,
            boxes=True,
            line_width=2
        )
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()
        
        # Print detection details
        if len(r.boxes) > 0:
            print(f"Detected {len(r.boxes)} objects:")
            for box in r.boxes:
                class_name = r.names[int(box.cls)]
                confidence = box.conf.item()
                print(f"Class: {class_name}, Confidence: {confidence:.3f}")
        else:
            print("No objects detected")

# Test inference on a sample image
sample_image_path = f"{dataset_path}/test/images"
if os.path.exists(sample_image_path):
    test_images = [f for f in os.listdir(sample_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if test_images:
        test_image = os.path.join(sample_image_path, test_images[0])
        print(f"Testing YOLO11 inference on: {test_image}")
        test_yolo11_inference(best_model_path, test_image)

# ============================================================================
# SECTION 11: YOLO11 MODEL EXPORT AND OPTIMIZATION
# ============================================================================

def export_yolo11_model(model_path):
    """Export YOLO11 model in different formats for deployment"""
    model = YOLO(model_path)
    
    # Export to ONNX for better inference speed
    onnx_path = model.export(
        format='onnx', 
        imgsz=640, 
        half=True, 
        int8=False, 
        dynamic=True,
        simplify=True,
        opset=17
    )
    print(f"YOLO11 model exported to ONNX format: {onnx_path}")
    
    # Export to TensorRT for NVIDIA GPU deployment (optional)
    try:
        trt_path = model.export(
            format='engine', 
            imgsz=640, 
            half=True, 
            workspace=4,
            int8=False,
            data=os.path.join(dataset_path, "data.yaml")
        )
        print(f"YOLO11 model exported to TensorRT format: {trt_path}")
    except Exception as e:
        print(f"TensorRT export failed: {e}")
    
    # Export to CoreML for iOS deployment (optional)
    try:
        coreml_path = model.export(
            format='coreml',
            imgsz=640,
            half=True,
            int8=False
        )
        print(f"YOLO11 model exported to CoreML format: {coreml_path}")
    except Exception as e:
        print(f"CoreML export failed: {e}")
    
    return True

# Export the trained YOLO11 model
if os.path.exists(best_model_path):
    export_yolo11_model(best_model_path)

# ============================================================================
# SECTION 12: YOLO11 PERFORMANCE METRICS AND REPORTING
# ============================================================================

def generate_yolo11_training_report(project_path):
    """Generate comprehensive YOLO11 training report"""
    results_path = os.path.join(project_path, "results.png")
    confusion_matrix_path = os.path.join(project_path, "confusion_matrix.png")
    val_batch_path = os.path.join(project_path, "val_batch0_pred.jpg")
    
    print("YOLO11 Training Report:")
    print("=" * 50)
    
    # Display training curves
    if os.path.exists(results_path):
        print("Training curves saved at:", results_path)
        display(Image(results_path))
    
    # Display confusion matrix
    if os.path.exists(confusion_matrix_path):
        print("Confusion matrix saved at:", confusion_matrix_path)
        display(Image(confusion_matrix_path))
    
    # Display validation predictions
    if os.path.exists(val_batch_path):
        print("Validation predictions saved at:", val_batch_path)
        display(Image(val_batch_path))
    
    # Memory usage report
    if torch.cuda.is_available():
        print(f"\nFinal GPU memory usage:")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

# Generate final report
project_path = f"{TRAINING_CONFIG['project']}/{TRAINING_CONFIG['name']}"
if os.path.exists(project_path):
    generate_yolo11_training_report(project_path)

# ============================================================================
# SECTION 13: YOLO11 BENCHMARKING AND COMPARISON
# ============================================================================

def benchmark_yolo11_model(model_path, test_images_path, num_images=50):
    """Benchmark YOLO11 model performance"""
    import time
    
    model = YOLO(model_path)
    
    if not os.path.exists(test_images_path):
        print("Test images path not found for benchmarking")
        return
    
    test_images = [f for f in os.listdir(test_images_path) if f.endswith(('.jpg', '.jpeg', '.png'))][:num_images]
    
    if not test_images:
        print("No test images found for benchmarking")
        return
    
    print(f"Benchmarking YOLO11 model on {len(test_images)} images...")
    
    # Benchmark metrics
    total_inference_time = 0
    total_detections = 0
    fps_measurements = []
    
    # Warm up the model
    warmup_image = os.path.join(test_images_path, test_images[0])
    for _ in range(5):
        _ = model(warmup_image)
    
    print("Starting benchmark...")
    
    for i, image_name in enumerate(test_images):
        image_path = os.path.join(test_images_path, image_name)
        
        # Measure inference time
        start_time = time.time()
        results = model(image_path, conf=0.25, iou=0.7, verbose=False)
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_inference_time += inference_time
        
        # Calculate FPS
        fps = 1.0 / inference_time
        fps_measurements.append(fps)
        
        # Count detections
        for r in results:
            total_detections += len(r.boxes)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_images)} images...")
    
    # Calculate metrics
    avg_inference_time = total_inference_time / len(test_images)
    avg_fps = sum(fps_measurements) / len(fps_measurements)
    avg_detections_per_image = total_detections / len(test_images)
    
    # Print benchmark results
    print(f"\n{'='*60}")
    print("YOLO11 BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total images processed: {len(test_images)}")
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(f"Average inference time: {avg_inference_time*1000:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections_per_image:.2f}")
    print(f"Min FPS: {min(fps_measurements):.2f}")
    print(f"Max FPS: {max(fps_measurements):.2f}")
    
    return {
        'avg_inference_time': avg_inference_time,
        'avg_fps': avg_fps,
        'total_detections': total_detections,
        'avg_detections_per_image': avg_detections_per_image,
        'fps_measurements': fps_measurements
    }

# Run benchmark if model exists
if os.path.exists(best_model_path):
    test_images_path = f"{dataset_path}/test/images"
    if os.path.exists(test_images_path):
        benchmark_results = benchmark_yolo11_model(best_model_path, test_images_path)

# ============================================================================
# SECTION 14: YOLO11 MODEL COMPARISON AND ANALYSIS
# ============================================================================

def compare_yolo_models(dataset_path):
    """Compare YOLO11 with YOLOv8 performance"""
    if not os.path.exists(dataset_path):
        print("Dataset not found for model comparison")
        return
    
    print(f"\n{'='*60}")
    print("YOLO MODEL COMPARISON (YOLO11 vs YOLOv8)")
    print(f"{'='*60}")
    
    models_to_compare = [
        {'name': 'YOLO11n', 'model': 'yolo11n.pt'},
        {'name': 'YOLOv8n', 'model': 'yolov8n.pt'}
    ]
    
    comparison_results = {}
    
    for model_info in models_to_compare:
        print(f"\nTesting {model_info['name']}...")
        
        try:
            model = YOLO(model_info['model'])
            
            # Quick validation on a subset
            results = model.val(
                data=os.path.join(dataset_path, "data.yaml"),
                imgsz=640,
                batch=8,
                conf=0.25,
                iou=0.7,
                device=0,
                half=True,
                plots=False,
                verbose=False,
                save_json=False
            )
            
            comparison_results[model_info['name']] = {
                'mAP50': results.box.map50,
                'mAP50_95': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr
            }
            
            print(f"{model_info['name']} Results:")
            print(f"  mAP50: {results.box.map50:.4f}")
            print(f"  mAP50-95: {results.box.map:.4f}")
            print(f"  Precision: {results.box.mp:.4f}")
            print(f"  Recall: {results.box.mr:.4f}")
            
        except Exception as e:
            print(f"Error testing {model_info['name']}: {e}")
            comparison_results[model_info['name']] = None
    
    # Print comparison summary
    if len(comparison_results) >= 2:
        print(f"\n{'='*50}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*50}")
        
        yolo11_results = comparison_results.get('YOLO11n')
        yolov8_results = comparison_results.get('YOLOv8n')
        
        if yolo11_results and yolov8_results:
            print(f"mAP50 Improvement: {((yolo11_results['mAP50'] - yolov8_results['mAP50']) / yolov8_results['mAP50'] * 100):+.2f}%")
            print(f"mAP50-95 Improvement: {((yolo11_results['mAP50_95'] - yolov8_results['mAP50_95']) / yolov8_results['mAP50_95'] * 100):+.2f}%")
            print(f"Precision Improvement: {((yolo11_results['precision'] - yolov8_results['precision']) / yolov8_results['precision'] * 100):+.2f}%")
            print(f"Recall Improvement: {((yolo11_results['recall'] - yolov8_results['recall']) / yolov8_results['recall'] * 100):+.2f}%")
    
    return comparison_results

# Run model comparison
if dataset_path and os.path.exists(dataset_path):
    model_comparison = compare_yolo_models(dataset_path)

# ============================================================================
# SECTION 15: YOLO11 DEPLOYMENT PREPARATION
# ============================================================================

def prepare_deployment_package(model_path, output_dir="deployment_package"):
    """Prepare deployment package with optimized YOLO11 model"""
    if not os.path.exists(model_path):
        print("Model not found for deployment preparation")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy best model
    best_model_deployment = os.path.join(output_dir, "best_yolo11_model.pt")
    shutil.copy2(model_path, best_model_deployment)
    
    # Create inference script
    inference_script = f"""
# YOLO11 Bangladesh Street Food Detection - Inference Script
import cv2
import numpy as np
from ultralytics import YOLO
import argparse

class YOLO11Detector:
    def __init__(self, model_path="best_yolo11_model.pt", conf_threshold=0.25, iou_threshold=0.7):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = {SELECTED_CLASSES}
        
    def detect(self, image_path):
        results = self.model(image_path, conf=self.conf_threshold, iou=self.iou_threshold)
        return results
    
    def detect_batch(self, image_paths):
        results = self.model(image_paths, conf=self.conf_threshold, iou=self.iou_threshold)
        return results
    
    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            if output_path:
                out.write(annotated_frame)
            else:
                cv2.imshow('YOLO11 Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO11 Bangladesh Street Food Detection')
    parser.add_argument('--input', type=str, required=True, help='Input image/video path')
    parser.add_argument('--output', type=str, help='Output path (optional)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold')
    
    args = parser.parse_args()
    
    detector = YOLO11Detector(conf_threshold=args.conf, iou_threshold=args.iou)
    
    if args.input.endswith(('.mp4', '.avi', '.mov')):
        detector.process_video(args.input, args.output)
    else:
        results = detector.detect(args.input)
        
        # Save result
        if args.output:
            annotated = results[0].plot()
            cv2.imwrite(args.output, annotated)
        else:
            results[0].show()
"""
    
    with open(os.path.join(output_dir, "inference.py"), 'w') as f:
        f.write(inference_script)
    
    # Create requirements.txt
    requirements = """ultralytics>=8.3.0
opencv-python>=4.5.0
numpy>=1.21.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=8.3.0
"""
    
    with open(os.path.join(output_dir, "requirements.txt"), 'w') as f:
        f.write(requirements)
    
    # Create README for deployment
    readme_content = f"""
# YOLO11 Bangladesh Street Food Detection Model

## Model Information
- Architecture: YOLO11n
- Classes: {SELECTED_CLASSES}
- Input Size: 640x640
- Framework: Ultralytics YOLO11

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Image Detection
```bash
python inference.py --input image.jpg --output result.jpg
```

### Video Detection
```bash
python inference.py --input video.mp4 --output result.mp4
```

### Batch Processing
```python
from inference import YOLO11Detector

detector = YOLO11Detector()
results = detector.detect_batch(['image1.jpg', 'image2.jpg'])
```

## Model Performance
- mAP50: Check training results
- Average inference time: Check benchmark results (T4 GPU optimized)
- Supported formats: ONNX, TensorRT, CoreML

## Classes
{chr(10).join([f"{i}: {cls}" for i, cls in enumerate(SELECTED_CLASSES)])}
"""
    
    with open(os.path.join(output_dir, "README.md"), 'w') as f:
        f.write(readme_content)
    
    print(f"Deployment package created at: {output_dir}")
    print("Contents:")
    for item in os.listdir(output_dir):
        print(f"  - {item}")
    
    return output_dir

# Prepare deployment package
if os.path.exists(best_model_path):
    deployment_package_path = prepare_deployment_package(best_model_path)

# ============================================================================
# SECTION 16: CLEANUP AND FINAL SUMMARY
# ============================================================================

def cleanup_temporary_files():
    """Clean up temporary files to free up space"""
    temp_dirs = [
        "/tmp",
        "/content/sample_data"
    ]
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                # Only remove specific temporary files, not entire directories
                for file in os.listdir(temp_dir):
                    if file.startswith(('yolo', 'temp_', 'tmp_')):
                        file_path = os.path.join(temp_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
            except Exception as e:
                print(f"Warning: Could not clean {temp_dir}: {e}")

def generate_final_summary():
    """Generate final training summary"""
    print(f"\n{'='*80}")
    print("YOLO11 BANGLADESH STREET FOOD DETECTION - FINAL SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📊 EXPERIMENT CONFIGURATION:")
    print(f"   Model Architecture: {TRAINING_CONFIG['model_size']}")
    print(f"   Selected Classes: {SELECTED_CLASSES}")
    print(f"   Total Classes: {len(SELECTED_CLASSES)}")
    print(f"   Training Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"   Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"   Image Size: {TRAINING_CONFIG['imgsz']}")
    print(f"   GPU: {'T4' if torch.cuda.is_available() else 'CPU'}")
    
    print(f"\n📁 OUTPUT FILES:")
    project_path = f"{TRAINING_CONFIG['project']}/{TRAINING_CONFIG['name']}"
    if os.path.exists(project_path):
        print(f"   Training Results: {project_path}")
        print(f"   Best Model: {project_path}/weights/best.pt")
        print(f"   Last Model: {project_path}/weights/last.pt")
        
        # Check for exported models
        for export_format in ['onnx', 'engine', 'coreml']:
            export_file = f"{project_path}/weights/best.{export_format}"
            if os.path.exists(export_file):
                print(f"   Exported ({export_format.upper()}): {export_file}")
    
    print(f"\n🎯 PERFORMANCE SUMMARY:")
    if 'validation_results' in locals() and validation_results:
        print(f"   mAP50: {validation_results.box.map50:.4f}")
        print(f"   mAP50-95: {validation_results.box.map:.4f}")
    
    if 'benchmark_results' in locals() and benchmark_results:
        print(f"   Average Inference Time: {benchmark_results['avg_inference_time']*1000:.1f}ms")
        print(f"   Average FPS: {benchmark_results['avg_fps']:.1f}")
        print(f"   Avg Detections/Image: {benchmark_results['avg_detections_per_image']:.1f}")
    
    print(f"\n🚀 DEPLOYMENT:")
    if 'deployment_package_path' in locals():
        print(f"   Deployment Package: {deployment_package_path}")
        print(f"   Ready for production deployment!")
    
    print(f"\n💾 MEMORY USAGE:")
    if torch.cuda.is_available():
        print(f"   Final GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"   Peak GPU Memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    
    print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"   Your YOLO11 model is ready for deployment!")
    print(f"   Use the deployment package for production inference.")
    
    # W&B summary
    if wandb_enabled:
        print(f"\n📈 EXPERIMENT TRACKING:")
        print(f"   W&B Project: bangladesh-street-food-yolo11-3class")
        print(f"   View results at: https://wandb.ai")
        
        # Log final metrics to W&B
        if 'validation_results' in locals() and validation_results:
            wandb.log({
                "final_mAP50": validation_results.box.map50,
                "final_mAP50_95": validation_results.box.map,
                "training_completed": True
            })
        
        wandb.finish()
    
    print(f"\n{'='*80}")

# Final cleanup and summary
if torch.cuda.is_available():
    torch.cuda.empty_cache()

cleanup_temporary_files()
generate_final_summary()

print("\n🎉 YOLO11 Training Pipeline Completed Successfully! 🎉")
print("Your Bangladesh Street Food Detection model is ready to use!")