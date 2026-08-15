# Phase 3: Vision & OCR Content Moderation

## Overview

Phase 3 extends the text moderation system to handle **multimodal content** - images containing embedded text, memes, screenshots, and image-text combinations common in social media.

### Key Capabilities

1. **OCR Text Extraction** - Extract text from images using EasyOCR and Tesseract
2. **Image Classification** - Detect NSFW content, hate symbols, and violence
3. **Multimodal Fusion** - Combine text and image signals for unified moderation
4. **Integration** - Seamlessly connect with Phase 1 and Phase 2 text classifiers

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Image                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
      ┌────▼─────┐         ┌──────▼────────┐
      │   OCR    │         │    Image      │
      │ Extractor│         │  Classifier   │
      └────┬─────┘         └──────┬────────┘
           │                      │
    ┌──────▼──────┐        ┌──────▼─────────┐
    │ Extracted   │        │  NSFW: 0.8     │
    │ Text: "..."│        │  Hate: 0.3     │
    └──────┬──────┘        │  Violence: 0.1 │
           │               └──────┬─────────┘
           │                      │
    ┌──────▼────────┐             │
    │  Phase 1/2    │             │
    │ Text Moderator│             │
    │ Toxic: 0.7    │             │
    └──────┬────────┘             │
           │                      │
           └──────────┬───────────┘
                      │
              ┌───────▼────────┐
              │ Multimodal     │
              │ Fusion Layer   │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │ Final Verdict: │
              │ TOXIC (0.85)   │
              │ [Auto-Remove]  │
              └────────────────┘
```

---

## Directory Structure

```
phase3_vision_ocr/
├── README.md                      # This file
├── requirements.txt               # Phase 3 dependencies
├── configs/
│   └── default.yaml              # Configuration
├── src/
│   ├── ocr_worker.py             # OCR extraction
│   ├── image_classifier.py       # Image classification
│   ├── multimodal_moderator.py   # Fusion logic
│   └── preprocessing.py          # Image preprocessing utilities
├── scripts/
│   ├── download_datasets.py      # Download training data
│   ├── train_image_model.py      # Train image classifier
│   └── test_ocr.py               # Test OCR accuracy
├── evaluation/
│   ├── evaluate_ocr.py           # OCR benchmarks
│   └── evaluate_image.py         # Image classifier metrics
├── data/
│   ├── nsfw/                     # NSFW dataset
│   ├── hate_symbols/             # Hate symbols/memes
│   └── violence/                 # Violent content
├── test_images/                  # Sample test images
├── models/                       # Saved models
└── notebooks/
    └── phase3_demo.ipynb         # Interactive demo
```

---

## Installation

### 1. Install Dependencies

```bash
cd phase3_vision_ocr
pip install -r requirements.txt
```

### 2. Install Tesseract (System Dependency)

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Windows:**
Download installer from https://github.com/UB-Mannheim/tesseract/wiki

### 3. Download EasyOCR Models

EasyOCR will automatically download models on first use (~100MB).

---

## Quick Start

### 1. Test OCR Extraction

```bash
python scripts/test_ocr.py --image test_images/meme.jpg
```

Output:
```
🔍 OCR Extraction Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Image: meme.jpg
Engine: EasyOCR

Extracted Text:
"WHEN YOU SEE A TERRIBLE MEME
BUT IT HAS 1000 UPVOTES"

Confidence: 0.92
Processing Time: 1.2s
```

### 2. Test Image Classification

```bash
python scripts/test_image_classifier.py --image test_images/example.jpg
```

### 3. Run Full Multimodal Moderation

```bash
python src/multimodal_moderator.py --image test_images/meme.jpg
```

Output:
```
================================================================================
🛡️ MULTIMODAL CONTENT MODERATION REPORT
================================================================================

📄 Input: meme.jpg

📊 OCR TEXT ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extracted: "WHEN YOU SEE A TERRIBLE MEME..."
Text Toxicity: 0.12 ✅ DECENT

🖼️  IMAGE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NSFW Score: 0.05 ✅
Hate Symbols: 0.03 ✅
Violence: 0.02 ✅

🎯 FINAL VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Score: 0.15
Status: ✅ DECENT
Action: AUTO-APPROVE
```

---

## Training Custom Image Classifier

### 1. Prepare Dataset

First, download or create datasets:

```bash
# Create dummy datasets for testing
python scripts/download_datasets.py --dummy

# Or manually organize real images in this structure:
data/nsfw/
├── train/
│   ├── safe/
│   │   ├── image1.jpg
│   │   └── ...
│   └── nsfw/
│       ├── image1.jpg
│       └── ...
└── test/
    ├── safe/
    └── nsfw/
```

See `docs/DATASETS.md` for real dataset sources.

### 2. Train Model

```bash
# Train NSFW classifier
python scripts/train_image_model.py \
  --dataset nsfw \
  --backbone resnet50 \
  --epochs 20 \
  --batch-size 32

# Train hate symbols classifier
python scripts/train_image_model.py \
  --dataset hate_symbols \
  --backbone resnet50 \
  --epochs 20

# Train violence detector
python scripts/train_image_model.py \
  --dataset violence \
  --backbone efficientnet_b0 \
  --epochs 15
```

**Available backbones**: `resnet18`, `resnet50`, `efficientnet_b0`, `vit_b_16`

### 3. Test Model

```bash
# Test on single image
python scripts/test_image_classifier.py \
  --model models/nsfw/best_model.pt \
  --image test_images/sample.jpg

# Test on batch
python scripts/test_image_classifier.py \
  --model models/nsfw/best_model.pt \
  --batch test_images/
```

### 4. Monitor Training

Training outputs are saved to `models/<dataset>/`:
- `best_model.pt` - Best model checkpoint
- `latest_model.pt` - Latest checkpoint
- `training_config.json` - Training configuration
- `training_history.json` - Loss and accuracy history

---

## API Usage

### Python API

```python
from src.multimodal_moderator import MultimodalModerator

# Initialize
moderator = MultimodalModerator(
    ocr_engine="easyocr",
    image_model_path="models/nsfw_best.pt",
    text_moderator_phase1="../../content_moderation_trained/phase1_text_baseline",
    text_moderator_phase2="../../content_moderation_trained/phase2_multilingual"
)

# Moderate image
result = moderator.moderate_image("path/to/image.jpg")

print(f"Verdict: {result['verdict']}")
print(f"Overall Score: {result['overall_score']:.2f}")
print(f"Text Toxicity: {result['text_analysis']['toxicity']:.2f}")
print(f"NSFW Score: {result['image_analysis']['nsfw']:.2f}")
print(f"Action: {result['action']}")  # AUTO-APPROVE, HUMAN-REVIEW, AUTO-REMOVE
```

### CLI Usage

```bash
# Single image
python src/multimodal_moderator.py --image path/to/image.jpg

# Batch processing
python src/multimodal_moderator.py --batch path/to/images/

# With custom threshold
python src/multimodal_moderator.py --image test.jpg --threshold 0.7

# Output to JSON
python src/multimodal_moderator.py --image test.jpg --output results.json
```

---

## Configuration

Edit `configs/default.yaml` to customize:

### OCR Settings
```yaml
ocr:
  primary_engine: "easyocr"  # or "tesseract"
  gpu: true
  min_confidence: 0.3
  preprocessing:
    contrast_enhancement: true
    denoise: true
```

### Image Classification
```yaml
image_classification:
  model:
    backbone: "resnet50"  # or "efficientnet_b0", "vit_base_patch16_224"
  inference:
    threshold_nsfw: 0.7
    threshold_hate: 0.6
```

### Fusion Strategy
```yaml
fusion:
  weights:
    text_score: 0.6  # 60% weight to text
    image_score: 0.4  # 40% weight to image
```

---

## Performance Benchmarks

### OCR Accuracy (Target)
- Clean Text: > 95% character accuracy
- Noisy Images: > 80% character accuracy
- Processing Time: < 2s per image (GPU)

### Image Classification (Target)
- NSFW Detection: AUC-ROC > 0.93
- Hate Symbols: Recall > 0.90
- Violence Detection: AUC-ROC > 0.88

### End-to-End
- Multimodal Processing: < 3s per image (GPU)
- Throughput: > 20 images/second (batch mode, GPU)

---

## Datasets

### Recommended Sources

**NSFW Detection:**
- OpenNSFW: https://github.com/yahoo/open_nsfw
- NSFW Data Scraper: https://github.com/alex000kim/nsfw_data_scraper

**Hate Symbols/Memes:**
- Hateful Memes Challenge: https://ai.facebook.com/tools/hatefulmemes/
- Fine-Grained Hate Speech: https://github.com/binny1024/Labelling-of-Hate-Speech

**Violence Detection:**
- VSD2014: https://gitlab.com/volzotan/violentscenesdataset
- MediaEval: https://multimediaeval.github.io/

---

## Troubleshooting

### Issue: EasyOCR crashes or gives poor results
**Solution:** Try Tesseract as fallback
```bash
python src/multimodal_moderator.py --image test.jpg --ocr-engine tesseract
```

### Issue: Out of memory during training
**Solution:** Reduce batch size
```bash
python scripts/train_image_model.py --batch-size 16
```

### Issue: Slow OCR processing
**Solution:** Enable GPU and reduce image size
```yaml
ocr:
  gpu: true
  preprocessing:
    resize_max: 1280  # Reduce from 1920
```

---

## Integration with Phase 1 & 2

Phase 3 automatically integrates with existing text moderators:

```python
# OCR extracts text
extracted_text = ocr_worker.extract_text(image)

# Text goes through Phase 1 (detailed categories)
phase1_result = phase1_moderator.check_text(extracted_text)

# Text goes through Phase 2 (multilingual)
phase2_result = phase2_moderator.check_text(extracted_text)

# Combine with image classification
final_score = fusion_layer.combine(
    text_scores=[phase1_result, phase2_result],
    image_scores=[nsfw_score, hate_score, violence_score]
)
```

---

## Next Steps

After completing Phase 3:
- **Phase 4**: Advanced multimodal fusion with attention mechanisms
- **Phase 5**: Human-in-the-loop interface and production deployment

---

## References

- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **Tesseract OCR**: https://github.com/tesseract-ocr/tesseract
- **PyTorch Image Models (timm)**: https://github.com/huggingface/pytorch-image-models
- **Hateful Memes Paper**: https://arxiv.org/abs/2005.04790

---

**Status**: 🚧 In Development  
**Last Updated**: August 15, 2026
