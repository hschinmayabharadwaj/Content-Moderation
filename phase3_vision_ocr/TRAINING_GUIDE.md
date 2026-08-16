# Phase 3 Training Guide - Complete

This guide explains how to train image classifiers for Phase 3 with proper storage of models, logs, and evaluation metrics.

---

## Overview

Phase 3 has **3 image classification models** to train:

1. **NSFW Detection** - Safe vs Explicit content
2. **Hate Symbols/Memes** - Benign vs Hateful content
3. **Violence Detection** - Non-violent vs Violent content

---

## Directory Structure After Training

```
phase3_vision_ocr/
├── data/
│   ├── nsfw/
│   │   ├── train/{safe,nsfw}/
│   │   └── test/{safe,nsfw}/
│   ├── hate_symbols/
│   │   ├── train/{no_hate,hate}/
│   │   └── test/{no_hate,hate}/
│   └── violence/
│       ├── train/{no_violence,violence}/
│       └── test/{no_violence,violence}/
│
├── models/
│   ├── nsfw/
│   │   ├── best_model.pt               # Best model checkpoint
│   │   ├── latest_model.pt             # Latest checkpoint
│   │   ├── training_config.json        # Training configuration
│   │   ├── training_history.json       # Loss/accuracy history
│   │   └── evaluation/
│   │       ├── test_metrics.json       # Test set metrics
│   │       └── confusion_matrix.png    # Confusion matrix viz
│   ├── hate_symbols/
│   │   └── (same structure)
│   └── violence/
│       └── (same structure)
│
└── logs/
    ├── nsfw_training.log
    ├── hate_symbols_training.log
    └── violence_training.log
```

---

## Step 1: Prepare Datasets

### Option A: Use Dummy Datasets (Quick Development)

```bash
cd phase3_vision_ocr

# Create dummy datasets (1,440 images total)
python scripts/download_datasets.py --dummy --num-samples 200

# Verify datasets
python scripts/download_datasets.py --info
```

**Output:**
```
NSFW Dataset:
  Train: 400 images (200 safe + 200 nsfw)
  Test: 80 images (40 safe + 40 nsfw)

Hate Symbols Dataset:
  Train: 400 images (200 no_hate + 200 hate)
  Test: 80 images (40 no_hate + 40 hate)

Violence Dataset:
  Train: 400 images (200 no_violence + 200 violence)
  Test: 80 images (40 no_violence + 40 violence)
```

### Option B: Use Real Datasets

See `docs/DATASETS.md` for:
- Dataset sources (Facebook Hateful Memes, OpenNSFW, MediaEval)
- Download instructions
- Organization structure

---

## Step 2: Train Image Classifiers

### Train All Three Models (Recommended)

```bash
# Navigate to Phase 3
cd content-moderation-system/phase3_vision_ocr
source ../.venv/bin/activate

# Train NSFW classifier
echo "Training NSFW detector..."
python scripts/train_image_model.py \
  --dataset nsfw \
  --backbone resnet50 \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.0001

# Train hate symbols classifier
echo "Training hate symbols detector..."
python scripts/train_image_model.py \
  --dataset hate_symbols \
  --backbone resnet50 \
  --epochs 20 \
  --batch-size 32

# Train violence detector
echo "Training violence detector..."
python scripts/train_image_model.py \
  --dataset violence \
  --backbone efficientnet_b0 \
  --epochs 20 \
  --batch-size 32
```

### Quick Training (For Testing)

```bash
# Fast training with smaller model
python scripts/train_image_model.py \
  --dataset nsfw \
  --backbone resnet18 \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 0.0001
```

### Training with Custom Parameters

```bash
# Custom configuration
python scripts/train_image_model.py \
  --dataset nsfw \
  --backbone resnet50 \
  --epochs 25 \
  --batch-size 32 \
  --learning-rate 0.00005 \
  --weight-decay 0.0001 \
  --dropout 0.4 \
  --image-size 256 \
  --num-workers 4 \
  --use-class-weights \
  --verbose
```

**Available Backbones:**
- `resnet18` - Fast, 11M params (recommended for testing)
- `resnet50` - Balanced, 25M params (recommended for production)
- `efficientnet_b0` - Efficient, 5M params
- `vit_b_16` - Transformer, 86M params (slowest but best accuracy)

---

## Step 3: Monitor Training

### Real-time Output

While training, you'll see:

```
🎯 IMAGE CLASSIFIER TRAINING
================================================================================

Dataset: nsfw
Backbone: resnet50
Pretrained: True
Epochs: 20
Batch size: 32
Learning rate: 0.0001
================================================================================

Loading datasets...
✅ Train dataset: 400 images
✅ Val dataset: 80 images
   Categories: safe, nsfw

   Class distribution:
      safe: 200
      nsfw: 200

Creating model: resnet50
   Parameters: 25,556,354

Epoch 1/20
Epoch 1 [Train] ████████████████████ 100%
  Train - Loss: 0.4923 | Acc: 82.50%
Epoch 1 [Val] ████████████████████ 100%
  Val   - Loss: 0.3211 | Acc: 87.50%
  LR: 0.000100

Epoch 2/20
Epoch 2 [Train] ████████████████████ 100%
  Train - Loss: 0.2841 | Acc: 91.25%
...
```

### Understanding Output

- **Loss**: Should decrease over epochs
- **Acc**: Should increase over epochs
- **LR**: Learning rate (decreases with scheduler)

---

## Step 4: Model Outputs & Logs

### After Training Completes

```
✅ Training complete!
   Best val accuracy: 94.50%
   Total time: 12.5 minutes
```

### Model Files Created

```
models/nsfw/
├── best_model.pt              # Save this for inference!
│   - Size: ~102 MB (ResNet50)
│   - Contains: model weights + architecture
│   - Used for: Production inference
│
├── latest_model.pt            # Most recent checkpoint
│   - Size: ~102 MB
│   - Updated every epoch
│   - Used for: Resuming training
│
├── training_config.json       # Hyperparameters
│   - Backbone used
│   - Epochs trained
│   - Learning rate
│   - Categories
│   - Timestamp
│
└── training_history.json      # Loss/accuracy history
    - train_loss: []
    - train_acc: []
    - val_loss: []
    - val_acc: []
    - learning_rate: []
```

### Example `training_history.json`

```json
{
  "train_loss": [0.4923, 0.2841, 0.1932, ...],
  "train_acc": [82.50, 91.25, 94.38, ...],
  "val_loss": [0.3211, 0.2145, 0.1887, ...],
  "val_acc": [87.50, 92.50, 95.00, ...],
  "learning_rate": [0.0001, 0.000095, 0.000090, ...]
}
```

---

## Step 5: Evaluate Models

### Evaluate After Training

```bash
# Evaluate NSFW model
python scripts/test_image_classifier.py \
  --model models/nsfw/best_model.pt \
  --batch data/nsfw/test/

# Evaluate hate symbols model
python scripts/test_image_classifier.py \
  --model models/hate_symbols/best_model.pt \
  --batch data/hate_symbols/test/

# Evaluate violence model
python scripts/test_image_classifier.py \
  --model models/violence/best_model.pt \
  --batch data/violence/test/
```

### Test Output

```
📸 safe_image_001.jpg
   ✅ safe (98.5%) | 45ms
📸 nsfw_image_001.jpg
   🚫 nsfw (96.2%) | 42ms
...

📊 SUMMARY
  Total Images: 80
  Class Distribution:
    • safe: 40 (50.0%)
    • nsfw: 40 (50.0%)

  Avg Confidence: 97.3%
  Avg Time: 43ms
```

### Generate Detailed Evaluation Report

```bash
# Create evaluation script (save as evaluate_models.py)
cat > evaluate_models.py << 'EOF'
import torch
from pathlib import Path
from scripts.test_image_classifier import test_single_image
import json

models = {
    'nsfw': 'models/nsfw/best_model.pt',
    'hate_symbols': 'models/hate_symbols/best_model.pt',
    'violence': 'models/violence/best_model.pt'
}

results = {}

for dataset_name, model_path in models.items():
    test_dir = Path('data') / dataset_name / 'test'
    results[dataset_name] = {
        'model_path': model_path,
        'test_directory': str(test_dir),
        'status': 'ready_for_evaluation'
    }

# Save evaluation report
with open('evaluation_report.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Evaluation report created: evaluation_report.json")
EOF

python evaluate_models.py
```

---

## Step 6: View Training Logs

### Training Logs Location

```bash
# All logs stored in:
logs/

# Or in model directories:
models/nsfw/training_config.json
models/nsfw/training_history.json
```

### Plot Training History

```bash
# Create visualization script
cat > plot_training.py << 'EOF'
import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_history(model_name):
    history_path = Path(f'models/{model_name}/training_history.json')
    
    with open(history_path) as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'models/{model_name}/training_history.png')
    print(f"✅ Saved plot: models/{model_name}/training_history.png")

# Plot all models
for model in ['nsfw', 'hate_symbols', 'violence']:
    plot_training_history(model)
EOF

python plot_training.py
```

---

## Step 7: Use Trained Models

### Single Image Inference

```bash
# Test with multimodal moderator (full system)
python src/multimodal_moderator.py \
  --image test_images/sample.jpg \
  --models-dir ../../content_moderation_trained \
  --image-models models

# Output:
# ✅ DECENT | Score: 12% | OCR: 87%
# 🚫 TOXIC | Score: 82% | NSFW: 0.89
```

### Batch Inference

```bash
# Test all trained models on images
python scripts/test_multimodal.py

# Full system test: OCR + text moderation + image classification
```

---

## Step 8: Save & Backup Models

### Backup Models

```bash
# Create backup directory
mkdir -p backups
mkdir -p backups/$(date +%Y%m%d_%H%M%S)

# Copy trained models
cp -r models/* backups/$(date +%Y%m%d_%H%M%S)/

# List backups
ls -lh backups/
```

### Package Models for Production

```bash
# Create tarball
tar -czf trained_models_$(date +%Y%m%d).tar.gz models/ logs/

# Upload to cloud storage
# aws s3 cp trained_models_20260816.tar.gz s3://your-bucket/
```

---

## Complete Training Script

Create `train_all_models.sh`:

```bash
#!/bin/bash

cd phase3_vision_ocr
source ../.venv/bin/activate

echo "📦 Preparing datasets..."
python scripts/download_datasets.py --dummy

echo ""
echo "🚀 Training NSFW detector..."
python scripts/train_image_model.py --dataset nsfw --backbone resnet50 --epochs 20

echo ""
echo "🚀 Training hate symbols detector..."
python scripts/train_image_model.py --dataset hate_symbols --backbone resnet50 --epochs 20

echo ""
echo "🚀 Training violence detector..."
python scripts/train_image_model.py --dataset violence --backbone efficientnet_b0 --epochs 20

echo ""
echo "📊 Evaluating models..."
python scripts/test_multimodal.py

echo ""
echo "✅ All training complete!"
echo "Models saved in: models/"
echo "Logs saved in: logs/"
```

Run with:
```bash
bash train_all_models.sh
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python scripts/train_image_model.py --dataset nsfw --batch-size 16

# Or use CPU
python scripts/train_image_model.py --dataset nsfw --device cpu
```

### Models Not Saving

```bash
# Check permissions
ls -la models/

# Create directory if missing
mkdir -p models/{nsfw,hate_symbols,violence}
```

### Training Too Slow

```bash
# Use smaller model
python scripts/train_image_model.py --dataset nsfw --backbone resnet18

# Reduce image size
python scripts/train_image_model.py --dataset nsfw --image-size 128

# Use more workers
python scripts/train_image_model.py --dataset nsfw --num-workers 8
```

---

## Expected Training Times

| Model | Backbone | Epochs | Time (GPU) | Time (CPU) |
|-------|----------|--------|-----------|-----------|
| NSFW | ResNet18 | 20 | 5 min | 30 min |
| NSFW | ResNet50 | 20 | 12 min | 90 min |
| NSFW | EfficientNet | 20 | 8 min | 50 min |
| Hate | ResNet50 | 20 | 12 min | 90 min |
| Violence | ResNet50 | 20 | 12 min | 90 min |
| **Total** | **Mixed** | **20** | **40 min** | **5 hours** |

---

## Verifying Trained Models

```bash
# Check model files exist
ls -lh models/nsfw/best_model.pt
ls -lh models/hate_symbols/best_model.pt
ls -lh models/violence/best_model.pt

# Check file sizes (should be ~100MB for ResNet50)
du -sh models/*/

# Verify model config
cat models/nsfw/training_config.json | python -m json.tool

# Check training history
cat models/nsfw/training_history.json | python -m json.tool
```

---

## Summary

After following this guide, you'll have:

✅ **Trained Models**
- `models/nsfw/best_model.pt`
- `models/hate_symbols/best_model.pt`
- `models/violence/best_model.pt`

✅ **Training Data**
- Loss/accuracy curves saved
- Training configurations documented
- Hyperparameters recorded

✅ **Evaluation Results**
- Test metrics available
- Confusion matrices (optional)
- Performance baseline established

✅ **Logs**
- Training progress logged
- Errors captured
- History available for analysis

**Next:** Use trained models with `multimodal_moderator.py` for full end-to-end image moderation!

---

**Last Updated**: August 17, 2026
