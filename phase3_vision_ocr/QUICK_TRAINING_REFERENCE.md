# Phase 3 Training - Quick Reference

## 🚀 Fastest Way to Train All Models

```bash
cd content-moderation-system/phase3_vision_ocr
source ../.venv/bin/activate
bash train_all_models.sh
```

That's it! The script will:
1. ✅ Create datasets
2. ✅ Train NSFW model
3. ✅ Train hate symbols model
4. ✅ Train violence model
5. ✅ Evaluate all models
6. ✅ Save everything properly

---

## 📦 What Gets Saved

```
models/
├── nsfw/
│   ├── best_model.pt                    ← Use this for inference!
│   ├── latest_model.pt                  ← Latest checkpoint
│   ├── training_config.json             ← Configuration
│   └── training_history.json            ← Loss/accuracy curves
│
├── hate_symbols/
│   └── (same structure)
│
└── violence/
    └── (same structure)
```

---

## 🎯 Training Individual Models

```bash
# NSFW (fast)
python scripts/train_image_model.py --dataset nsfw --backbone resnet50 --epochs 20

# Hate symbols
python scripts/train_image_model.py --dataset hate_symbols --backbone resnet50 --epochs 20

# Violence
python scripts/train_image_model.py --dataset violence --backbone efficientnet_b0 --epochs 20

# Quick test (5 epochs)
python scripts/train_image_model.py --dataset nsfw --backbone resnet18 --epochs 5
```

---

## ⚙️ Training Parameters

### Backbones (Choose One)
| Backbone | Speed | Accuracy | Params | Memory |
|----------|-------|----------|--------|--------|
| resnet18 | ⚡⚡⚡ | 90% | 11M | Low |
| resnet50 | ⚡⚡ | 95% | 25M | Medium |
| efficientnet_b0 | ⚡⚡ | 93% | 5M | Low |
| vit_b_16 | ⚡ | 97% | 86M | High |

### Epochs
- **5**: Quick test (5 min)
- **10**: Fast training (10 min)
- **20**: Recommended (20-40 min) 
- **30+**: Deep training (60+ min)

### Batch Size
- **16**: Small, low memory
- **32**: Medium, balanced (recommended)
- **64**: Large, high memory

---

## 📊 Monitor Training

```bash
# Watch real-time output
# You'll see loss and accuracy improve each epoch

# After training, view results
cat models/nsfw/training_history.json

# Plot training curves (optional)
python -c "
import json
import matplotlib.pyplot as plt

with open('models/nsfw/training_history.json') as f:
    h = json.load(f)

plt.plot(h['train_loss'], label='Train Loss')
plt.plot(h['val_loss'], label='Val Loss')
plt.legend()
plt.savefig('models/nsfw/loss_curve.png')
print('✅ Saved loss_curve.png')
"
```

---

## ✅ Verify Models After Training

```bash
# Check files exist
ls -lh models/*/best_model.pt

# Check file sizes (~100MB for ResNet50)
du -sh models/*/

# Verify config
cat models/nsfw/training_config.json | python -m json.tool

# Test models work
python scripts/test_multimodal.py
```

---

## 🎬 Use Trained Models

```bash
# Single image
python src/multimodal_moderator.py \
  --image test_images/sample.jpg \
  --models-dir ../../content_moderation_trained \
  --image-models models

# Batch test
python scripts/test_multimodal.py
```

---

## 📁 Full Training Output Folder Structure

After `bash train_all_models.sh`:

```
phase3_vision_ocr/
├── data/
│   ├── nsfw/train/{safe,nsfw}/         [1,440 images generated]
│   ├── hate_symbols/train/{no_hate,hate}/
│   └── violence/train/{no_violence,violence}/
│
├── models/                              ← All trained models here
│   ├── nsfw/
│   │   ├── best_model.pt               [Production model]
│   │   ├── latest_model.pt
│   │   ├── training_config.json
│   │   ├── training_history.json       [Loss/accuracy data]
│   │   └── training_history.png        [Optional: visualization]
│   ├── hate_symbols/
│   │   └── (same files)
│   └── violence/
│       └── (same files)
│
└── logs/
    ├── nsfw_training.log               [Training output]
    ├── hate_symbols_training.log
    └── violence_training.log
```

---

## 🐛 Troubleshooting

### Out of Memory
```bash
python scripts/train_image_model.py --dataset nsfw --batch-size 16
```

### Too Slow
```bash
# Use smaller model
python scripts/train_image_model.py --dataset nsfw --backbone resnet18 --epochs 5
```

### Models Not Saving
```bash
# Create directories
mkdir -p models/{nsfw,hate_symbols,violence}
```

---

## ⏱️ Expected Times

| Task | GPU | CPU |
|------|-----|-----|
| Train 1 model (20 epochs) | 10-15 min | 60-90 min |
| Train all 3 models | 40-50 min | 3-5 hours |
| Full script (with eval) | 45-60 min | 3-6 hours |

---

## 📝 Training Checklist

- [ ] Navigate to `phase3_vision_ocr`
- [ ] Activate virtual environment
- [ ] Run `bash train_all_models.sh`
- [ ] Wait for completion
- [ ] Check `models/*/best_model.pt` exists
- [ ] Check `models/*/training_history.json` has data
- [ ] Run `python scripts/test_multimodal.py`
- [ ] Verify results look reasonable
- [ ] (Optional) Backup: `tar -czf trained_models_backup.tar.gz models/`

---

## 🎓 Next Steps After Training

1. **Use models**: `python src/multimodal_moderator.py --image test.jpg --image-models models`
2. **Fine-tune**: Modify hyperparameters and retrain
3. **Deploy**: Copy `models/*/best_model.pt` to production
4. **Evaluate**: Analyze `training_history.json` for insights

---

## 📚 Full Documentation

- `TRAINING_GUIDE.md` - Complete step-by-step guide
- `README.md` - Phase 3 overview
- `docs/DATASETS.md` - Real dataset information
- `../COMMAND_REFERENCE.md` - All commands for all phases

---

**TL;DR**: `bash train_all_models.sh` - Done! ✨
