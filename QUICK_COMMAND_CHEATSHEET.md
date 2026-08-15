# Content Moderation System - Quick Command Cheatsheet

**One-page reference for common tasks**

---

## 🚀 Quick Start

```bash
# Activate environment (do this first!)
cd content-moderation-system
source .venv/bin/activate  # macOS/Linux

# Test text moderation (simplest)
cd /Users/chinmayabharadwajhs/Content
python text_moderator.py --text "Your text here"
```

---

## 📝 Phase 1: Text Moderation

| Task | Command |
|------|---------|
| **Train model** | `cd phase1_text_baseline && python train_classifier.py --config configs/baseline.yaml` |
| **Quick train (sample)** | `python train_classifier.py --config configs/baseline.yaml --use-sample` |
| **Calibrate thresholds** | `python calibrate_thresholds.py --config configs/baseline.yaml` |
| **Full pipeline** | `bash run_phase1.sh` |
| **Download data** | `python download_data.py --output-dir data --analyze` |

---

## 🌍 Phase 2: Multilingual

| Task | Command |
|------|---------|
| **Train multilingual** | `cd phase2_multilingual && python train_multilingual.py --config configs/multilingual.yaml` |
| **Detect language** | `python language_identifier.py --text "Hello world"` |
| **Evaluate model** | `python evaluate_multilingual.py --config configs/multilingual.yaml --model-path models/best_model.pt` |
| **Unified inference** | `python unified_inference.py --text "Your text" --phase1-model ../phase1_text_baseline/models/best_model.pt --phase2-model models/best_model.pt` |
| **Full pipeline** | `bash run_phase2.sh` |

---

## 🖼️ Phase 3: Vision & OCR

### Setup
```bash
cd phase3_vision_ocr
bash setup.sh
```

### OCR Testing

| Task | Command |
|------|---------|
| **Test OCR** | `python scripts/test_ocr.py --image test_images/meme.jpg` |
| **Batch OCR** | `python scripts/test_ocr.py --batch test_images/` |
| **With visualization** | `python scripts/test_ocr.py --image test.jpg --visualize` |

### Dataset Preparation

| Task | Command |
|------|---------|
| **Create dummy data** | `python scripts/download_datasets.py --dummy` |
| **Show dataset info** | `python scripts/download_datasets.py --info` |
| **Create test images** | `python scripts/create_test_images.py` |

### Image Classifier Training

| Task | Command |
|------|---------|
| **Train NSFW** | `python scripts/train_image_model.py --dataset nsfw --backbone resnet50 --epochs 20` |
| **Train hate symbols** | `python scripts/train_image_model.py --dataset hate_symbols --backbone resnet18 --epochs 15` |
| **Train violence** | `python scripts/train_image_model.py --dataset violence --backbone efficientnet_b0 --epochs 20` |
| **Test classifier** | `python scripts/test_image_classifier.py --model models/nsfw/best_model.pt --image test.jpg` |

### Image Moderation

| Task | Command |
|------|---------|
| **OCR + Text mod** | `python src/image_text_moderator.py --image test.jpg --models-dir ../../content_moderation_trained` |
| **Full multimodal** | `python src/multimodal_moderator.py --image test.jpg --models-dir ../../content_moderation_trained --image-models models` |
| **Batch test** | `python scripts/test_multimodal.py` |

---

## 🎯 Text Moderator (Root Level)

| Task | Command |
|------|---------|
| **Interactive mode** | `python text_moderator.py` |
| **Single text** | `python text_moderator.py --text "Your text here"` |
| **Batch from file** | `python text_moderator.py --file texts.txt` |
| **Web interface** | `python text_moderator_web.py` |
| **Custom threshold** | `python text_moderator.py --text "Your text" --threshold 0.7` |

---

## 🔥 Most Common Use Cases

### 1. Moderate a single text
```bash
python text_moderator.py --text "Your text here"
```

### 2. Moderate an image with text
```bash
python content-moderation-system/phase3_vision_ocr/src/image_text_moderator.py \
  --image your_image.jpg \
  --models-dir content_moderation_trained
```

### 3. Train Phase 1 quickly
```bash
cd content-moderation-system/phase1_text_baseline
python download_data.py --output-dir data --create-sample --sample-size 5000
python train_classifier.py --config configs/baseline.yaml --use-sample
```

### 4. Setup Phase 3 and test
```bash
cd content-moderation-system/phase3_vision_ocr
bash setup.sh
python scripts/download_datasets.py --dummy
python scripts/test_multimodal.py
```

---

## 🛠️ Fusion Strategies (Phase 3)

```bash
# Weighted (default: 60% text, 40% image)
python src/multimodal_moderator.py --image test.jpg --fusion weighted

# Max score across modalities
python src/multimodal_moderator.py --image test.jpg --fusion max

# Adaptive (adjusts based on OCR confidence)
python src/multimodal_moderator.py --image test.jpg --fusion adaptive

# Custom weights
python src/multimodal_moderator.py --image test.jpg --fusion weighted --text-weight 0.7 --image-weight 0.3
```

---

## 📊 Model Backbones

### Phase 1 & 2 (Text)
- `distilbert-base-uncased` (fast, 66M params)
- `roberta-base` (better, 125M params)
- `xlm-roberta-base` (multilingual, 270M params)

### Phase 3 (Images)
- `resnet18` (fast, 11M params)
- `resnet50` (balanced, 25M params)
- `efficientnet_b0` (efficient, 5M params)
- `vit_b_16` (vision transformer, 86M params)

**Usage**:
```bash
# Text models
python train_classifier.py --config configs/baseline.yaml --model-name distilbert-base-uncased

# Image models
python scripts/train_image_model.py --dataset nsfw --backbone resnet50
```

---

## 🐛 Quick Troubleshooting

### Model not found
```bash
# Check Phase 1 model
ls content_moderation_trained/phase1_text_baseline/models/best_model.pt

# Check Phase 2 model
ls content_moderation_trained/phase2_multilingual/models/best_model.pt
```

### GPU/Memory issues
```bash
# Use smaller batch size
python train_classifier.py --config configs/baseline.yaml --batch-size 16

# Force CPU
python train_classifier.py --config configs/baseline.yaml --device cpu
```

### OCR not working
```bash
# Try Tesseract fallback
python scripts/test_ocr.py --image test.jpg --engine tesseract

# Check Tesseract installation
tesseract --version
```

### Import errors
```bash
# Reinstall dependencies
pip install --upgrade torch transformers

# Verify installation
pip show transformers torch easyocr
```

---

## 📂 Important File Locations

```
content-moderation-system/
├── text_moderator.py                              # Main text CLI
├── text_moderator_web.py                          # Web interface
├── COMPLETE_COMMAND_REFERENCE.md                  # Full docs (712 lines)
├── QUICK_COMMAND_CHEATSHEET.md                    # This file
│
├── phase1_text_baseline/
│   ├── train_classifier.py
│   ├── calibrate_thresholds.py
│   ├── configs/baseline.yaml
│   └── models/best_model.pt
│
├── phase2_multilingual/
│   ├── train_multilingual.py
│   ├── evaluate_multilingual.py
│   ├── configs/multilingual.yaml
│   └── models/best_model.pt
│
└── phase3_vision_ocr/
    ├── src/
    │   ├── ocr_worker.py
    │   ├── image_text_moderator.py
    │   ├── multimodal_moderator.py
    │   └── image_classifier.py
    ├── scripts/
    │   ├── train_image_model.py
    │   ├── test_ocr.py
    │   ├── test_multimodal.py
    │   └── download_datasets.py
    └── configs/default.yaml
```

---

## ⚡ Performance Tips

1. **Use sample data for quick tests**
   ```bash
   python download_data.py --create-sample --sample-size 5000
   ```

2. **Reduce batch size if out of memory**
   ```bash
   --batch-size 16  # Instead of 32
   ```

3. **Use lighter models for faster training**
   ```bash
   --model-name distilbert-base-uncased  # Text
   --backbone resnet18                   # Images
   ```

4. **Enable mixed precision (if supported)**
   ```bash
   --mixed-precision  # Faster training on supported GPUs
   ```

---

## 📞 Need More Help?

- **Full documentation**: `COMPLETE_COMMAND_REFERENCE.md` (712 lines)
- **Phase 1 guide**: `docs/getting_started_phase1.md`
- **Phase 2 guide**: `docs/getting_started_phase2.md`
- **Phase 3 guide**: `phase3_vision_ocr/README.md`
- **Dataset guide**: `phase3_vision_ocr/docs/DATASETS.md`

---

**Last Updated**: August 16, 2026  
**Quick Reference Version**: 1.0
