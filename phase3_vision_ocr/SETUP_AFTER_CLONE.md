# Phase 3 Setup Guide After Cloning

## Issue: Test Images Not Found

Test images were excluded from git (via `.gitignore`) because they are generated files. After cloning, you need to recreate them.

## Quick Fix

### 1. Navigate to Phase 3
```bash
cd content-moderation-system/phase3_vision_ocr
```

### 2. Activate Virtual Environment
```bash
source ../.venv/bin/activate
```

### 3. Create Test Images (1 command)
```bash
python scripts/create_test_images.py
```

This creates 6 test images:
- `clean_text.jpg` - Clean, readable text
- `meme_style.jpg` - Meme format with top/bottom text
- `noisy_text.jpg` - Text with noise artifacts
- `multiline_text.jpg` - Multiple lines of text
- `dark_bg.jpg` - White text on dark background
- `positive_text.jpg` - Positive sentiment text

### 4. Test OCR
```bash
# Single image
python scripts/test_ocr.py --image test_images/clean_text.jpg

# Batch test
python scripts/test_ocr.py --batch test_images/
```

### 5. Test Multimodal System
```bash
python scripts/test_multimodal.py
```

## Expected Results

### OCR Test Output
```
✅ 6/6 images processed successfully (100%)
✅ Average confidence: 86.5%
✅ Average processing time: 136ms per image
```

### Multimodal Test Output
```
✅ 7/7 images processed
✅ DECENT: 6 images (85.7%)
✅ TOXIC: 1 image (14.3%)
✅ Average OCR confidence: 86.5%
```

## Also Missing: Training Datasets

If you want to train image classifiers, create dummy datasets:

```bash
# Create dummy datasets (1,440 images total)
python scripts/download_datasets.py --dummy
```

This creates:
- `data/nsfw/` - NSFW classification dataset
- `data/hate_symbols/` - Hate symbols/memes dataset
- `data/violence/` - Violence detection dataset

Then train:
```bash
# Train NSFW classifier
python scripts/train_image_model.py --dataset nsfw --epochs 5

# Train hate symbols
python scripts/train_image_model.py --dataset hate_symbols --epochs 5

# Train violence
python scripts/train_image_model.py --dataset violence --epochs 5
```

## What's Already Available

✅ Source code (all .py files)
✅ Configuration files (*.yaml)
✅ Documentation (*.md)
✅ Requirements and setup scripts

❌ NOT included (recreate as needed):
- Test images (run `create_test_images.py`)
- Training datasets (run `download_datasets.py --dummy`)
- Trained models (*.pt files) - need to train them

## Complete Setup (one-liner for everything)

```bash
cd content-moderation-system/phase3_vision_ocr && \
source ../.venv/bin/activate && \
python scripts/create_test_images.py && \
python scripts/download_datasets.py --dummy && \
echo "✅ Setup complete!"
```

Then test:
```bash
python scripts/test_multimodal.py
```

## Troubleshooting

### "EasyOCR not available"
```bash
pip install easyocr
```

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "Out of memory"
Use smaller models or batch sizes:
```bash
# Small model
python scripts/train_image_model.py --dataset nsfw --backbone resnet18 --batch-size 16

# Or use CPU
python scripts/train_image_model.py --dataset nsfw --backbone resnet18 --device cpu
```

## Need More Help?

See:
- `README.md` - Full Phase 3 guide
- `docs/DATASETS.md` - Dataset information
- `../COMMAND_REFERENCE.md` - All commands for all phases
- `../QUICK_COMMAND_CHEATSHEET.md` - Quick reference

---

**TL;DR**: Run `python scripts/create_test_images.py` to fix the "Image not found" error!
