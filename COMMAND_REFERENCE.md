# Content Moderation System - Complete Command Reference

**Last Updated**: August 16, 2026  
**System Status**: Phase 1-3 Complete (6/9 Phase 3 tasks done)

---

## Quick Navigation

- [Phase 1: Text Baseline](#phase-1-text-baseline-classifier)
- [Phase 2: Multilingual](#phase-2-multilingual-support)
- [Phase 3: Vision & OCR](#phase-3-vision--ocr)
- [Text Moderator CLI](#text-moderator-unified)
- [Troubleshooting](#troubleshooting)

---

## Setup & Environment

```bash
# Navigate to project
cd content-moderation-system

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers datasets pyyaml pandas
pip install easyocr pytesseract opencv-python scikit-image
pip install torchvision timm albumentations gradio
```

---

## Phase 1: Text Baseline Classifier

**Location**: `phase1_text_baseline/`  
**Purpose**: Multi-label toxic comment classification

### Training

```bash
cd phase1_text_baseline

# Download dataset
python download_data.py --output-dir data --analyze

# Create sample (quick test)
python download_data.py --output-dir data --create-sample --sample-size 5000

# Train model
python train_classifier.py --config configs/baseline.yaml

# Train on sample
python train_classifier.py --config configs/baseline.yaml --use-sample

# Calibrate thresholds
python calibrate_thresholds.py --config configs/baseline.yaml
```

### Full Pipeline

```bash
bash run_phase1.sh
```

---

## Phase 2: Multilingual Support

**Location**: `phase2_multilingual/`  
**Purpose**: Cross-lingual toxic comment detection

### Training

```bash
cd phase2_multilingual

# Prepare datasets
python prepare_datasets.py --config configs/multilingual.yaml

# Train multilingual model
python train_multilingual.py --config configs/multilingual.yaml

# Evaluate
python evaluate_multilingual.py --config configs/multilingual.yaml --model-path models/best_model.pt
```

### Language Detection & Inference

```bash
# Detect language
python language_identifier.py --text "Hello world"

# Unified inference
python unified_inference.py \
  --text "Your text" \
  --phase1-model ../phase1_text_baseline/models/best_model.pt \
  --phase2-model models/best_model.pt

# Batch inference
python unified_inference.py \
  --file texts.txt \
  --phase1-model ../phase1_text_baseline/models/best_model.pt \
  --phase2-model models/best_model.pt \
  --output results.json
```

### Full Pipeline

```bash
bash run_phase2.sh
```

---

## Phase 3: Vision & OCR

**Location**: `phase3_vision_ocr/`  
**Purpose**: Multimodal content moderation (text-in-images, NSFW, hate, violence)

### Setup

```bash
cd phase3_vision_ocr
bash setup.sh
```

### Dataset Preparation

```bash
# Create dummy datasets (development)
python scripts/download_datasets.py --dummy

# Show dataset info
python scripts/download_datasets.py --info

# Create test images
python scripts/create_test_images.py
```

### OCR Testing

```bash
# Single image
python scripts/test_ocr.py --image test_images/meme.jpg

# With visualization
python scripts/test_ocr.py --image test.jpg --visualize

# Batch
python scripts/test_ocr.py --batch test_images/

# Use Tesseract
python scripts/test_ocr.py --image test.jpg --engine tesseract
```

### Image Classifier Training

```bash
# Train NSFW
python scripts/train_image_model.py \
  --dataset nsfw \
  --backbone resnet50 \
  --epochs 20 \
  --batch-size 32

# Train hate symbols
python scripts/train_image_model.py \
  --dataset hate_symbols \
  --backbone resnet18 \
  --epochs 15

# Train violence
python scripts/train_image_model.py \
  --dataset violence \
  --backbone efficientnet_b0 \
  --epochs 20

# Available backbones: resnet18, resnet50, efficientnet_b0, vit_b_16
```

### Image Text Moderation

```bash
# OCR + Text analysis
python src/image_text_moderator.py \
  --image test_images/meme.jpg \
  --models-dir ../../content_moderation_trained
```

### Multimodal Moderation (Full System)

```bash
# Weighted fusion (text 60%, image 40%)
python src/multimodal_moderator.py \
  --image test.jpg \
  --models-dir ../../content_moderation_trained \
  --image-models models \
  --fusion weighted

# Max score fusion
python src/multimodal_moderator.py \
  --image test.jpg \
  --fusion max

# Adaptive fusion (adjusts based on OCR confidence)
python src/multimodal_moderator.py \
  --image test.jpg \
  --fusion adaptive

# Custom weights
python src/multimodal_moderator.py \
  --image test.jpg \
  --fusion weighted \
  --text-weight 0.7 \
  --image-weight 0.3

# Batch testing
python scripts/test_multimodal.py
```

---

## Text Moderator (Unified CLI)

**Location**: Root directory  
**Purpose**: Simple interface using Phase 1 & 2 models

### Usage

```bash
cd /Users/chinmayabharadwajhs/Content

# Interactive mode (recommended)
python text_moderator.py

# Single text
python text_moderator.py --text "Your text here"

# Custom threshold
python text_moderator.py --text "Your text" --threshold 0.7

# Batch from file
python text_moderator.py --file texts.txt

# Web interface
python text_moderator_web.py
```

---

## Troubleshooting

### Model Loading Error
```bash
# Verify Phase 1 model
ls content_moderation_trained/phase1_text_baseline/models/best_model.pt

# Verify Phase 2 model
ls content_moderation_trained/phase2_multilingual/models/best_model.pt
```

### Memory Issues
```bash
# Reduce batch size
python train_classifier.py --config configs/baseline.yaml --batch-size 16

# Force CPU
python train_classifier.py --config configs/baseline.yaml --device cpu
```

### OCR Not Working
```bash
# Try Tesseract fallback
python scripts/test_ocr.py --image test.jpg --engine tesseract

# Verify Tesseract
tesseract --version
```

### Install Tesseract (System)
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

---

## Most Common Commands

### Moderate text
```bash
python text_moderator.py --text "Your text here"
```

### Moderate image with text
```bash
cd content-moderation-system/phase3_vision_ocr
python src/image_text_moderator.py --image test.jpg --models-dir ../../content_moderation_trained
```

### Full multimodal
```bash
cd content-moderation-system/phase3_vision_ocr
python src/multimodal_moderator.py --image test.jpg --models-dir ../../content_moderation_trained --image-models models
```

### Quick training (Phase 1)
```bash
cd content-moderation-system/phase1_text_baseline
python download_data.py --create-sample --sample-size 5000
python train_classifier.py --config configs/baseline.yaml --use-sample
```

---

## Project Structure

```
content-moderation-system/
├── text_moderator.py                    # Main CLI
├── text_moderator_web.py                # Web interface
├── QUICK_COMMAND_CHEATSHEET.md          # Quick reference
├── COMMAND_REFERENCE.md                 # This file
│
├── phase1_text_baseline/
│   ├── train_classifier.py
│   ├── calibrate_thresholds.py
│   ├── configs/baseline.yaml
│   └── run_phase1.sh
│
├── phase2_multilingual/
│   ├── train_multilingual.py
│   ├── evaluate_multilingual.py
│   ├── configs/multilingual.yaml
│   └── run_phase2.sh
│
└── phase3_vision_ocr/
    ├── src/
    │   ├── ocr_worker.py
    │   ├── image_classifier.py
    │   ├── image_text_moderator.py
    │   └── multimodal_moderator.py
    ├── scripts/
    │   ├── train_image_model.py
    │   ├── test_ocr.py
    │   ├── test_multimodal.py
    │   ├── download_datasets.py
    │   └── test_image_classifier.py
    ├── configs/default.yaml
    └── README.md
```

---

## Documentation

- **Phase 1**: `phase1_text_baseline/docs/getting_started_phase1.md`
- **Phase 2**: `phase2_multilingual/docs/getting_started_phase2.md`
- **Phase 3**: `phase3_vision_ocr/README.md`
- **Datasets**: `phase3_vision_ocr/docs/DATASETS.md`

---

**For detailed information, see QUICK_COMMAND_CHEATSHEET.md**
