# Image Classification Datasets Guide

## Overview

This document provides information about datasets used for training image classification models in Phase 3 of the content moderation system.

---

## Dataset Categories

### 1. NSFW Detection
**Purpose**: Classify images as safe-for-work or not-safe-for-work

**Categories**:
- `safe`: Safe content appropriate for all audiences
- `nsfw`: Adult/explicit content

**Recommended Real Datasets**:

#### Open NSFW (Yahoo)
- **Source**: https://github.com/yahoo/open_nsfw
- **Size**: ~60K images (train) + ~10K (test)
- **Format**: Caffe model + training data
- **License**: BSD-2-Clause
- **Notes**: Industry-standard NSFW detector, well-balanced dataset

#### NSFW Data Scraper
- **Source**: https://github.com/alex000kim/nsfw_data_scraper
- **Size**: Configurable (scrape your own)
- **Format**: Custom scraper with multiple sources
- **Notes**: Allows creation of custom NSFW datasets with proper filtering

#### Imagenet Subsets
- **Source**: http://www.image-net.org/
- **Notes**: Can use specific NSFW categories from ImageNet with proper filtering

**Current Status**: ✅ Dummy dataset created (400 train + 80 test)

---

### 2. Hateful Memes / Hate Symbols
**Purpose**: Detect hate speech, symbols, and harmful memes

**Categories**:
- `no_hate`: Benign memes and images
- `hate`: Hateful content, symbols, or harmful memes

**Recommended Real Datasets**:

#### Facebook Hateful Memes Challenge
- **Source**: https://ai.facebook.com/tools/hatefulmemes/
- **Size**: ~10K memes (8.5K train + 1K dev + 0.5K test)
- **Format**: JSON annotations + images
- **License**: Requires registration and agreement
- **Notes**: Gold standard for multimodal hate speech detection
- **Paper**: https://arxiv.org/abs/2005.04790

**How to download**:
1. Visit https://ai.facebook.com/tools/hatefulmemes/
2. Register and agree to terms of use
3. Download dataset (~2GB)
4. Extract to `data/hate_symbols/facebook_hateful_memes/`

#### Fine-Grained Hateful Memes
- **Source**: https://github.com/facebookresearch/fine_grained_hateful_memes
- **Size**: Subset of Hateful Memes with fine-grained labels
- **Format**: Extended annotations on hateful memes
- **Notes**: Provides attack type, target, and rationale annotations

#### Hate Symbol Database (ADL)
- **Source**: https://www.adl.org/hate-symbols
- **Notes**: Reference database of hate symbols, requires manual curation
- **Use**: Create training set by collecting images containing these symbols

**Current Status**: ✅ Dummy dataset created (400 train + 80 test)

---

### 3. Violence Detection
**Purpose**: Identify violent or graphic content

**Categories**:
- `no_violence`: Non-violent imagery
- `violence`: Violent scenes, weapons, blood, etc.

**Recommended Real Datasets**:

#### MediaEval Violent Scenes Detection (VSD)
- **Source**: https://multimediaeval.github.io/
- **Size**: ~18K images from movies
- **Format**: Keyframes from video with violence annotations
- **Notes**: Academic dataset, requires registration

#### VSD2014 Dataset
- **Source**: https://gitlab.com/volzotan/violentscenesdataset
- **Size**: Subset of MediaEval dataset
- **Format**: Organized image folders
- **Notes**: More accessible version of MediaEval data

#### UCF Crime Dataset
- **Source**: https://www.crcv.ucf.edu/projects/real-world/
- **Size**: ~1900 video clips (extract frames)
- **Format**: Videos with temporal annotations
- **Categories**: 13 anomaly types including violence
- **Notes**: Real-world CCTV footage

**Current Status**: ✅ Dummy dataset created (400 train + 80 test)

---

## Dataset Structure

All datasets follow this standard structure:

```
data/
├── nsfw/
│   ├── train/
│   │   ├── safe/
│   │   │   ├── image_0001.jpg
│   │   │   └── ...
│   │   └── nsfw/
│   │       ├── image_0001.jpg
│   │       └── ...
│   ├── test/
│   │   ├── safe/
│   │   └── nsfw/
│   └── metadata.json
│
├── hate_symbols/
│   ├── train/
│   │   ├── no_hate/
│   │   └── hate/
│   ├── test/
│   │   ├── no_hate/
│   │   └── hate/
│   └── metadata.json
│
└── violence/
    ├── train/
    │   ├── no_violence/
    │   └── violence/
    ├── test/
    │   ├── no_violence/
    │   └── violence/
    └── metadata.json
```

### Metadata Format

Each dataset includes a `metadata.json` file:

```json
{
  "dataset": "NSFW Dataset",
  "categories": ["safe", "nsfw"],
  "num_classes": 2,
  "splits": {
    "train": {
      "safe": 200,
      "nsfw": 200
    },
    "test": {
      "safe": 40,
      "nsfw": 40
    }
  },
  "type": "dummy",
  "description": "Safe vs NSFW image classification"
}
```

---

## Using Real Datasets

### Step 1: Download
Follow the instructions above for each dataset

### Step 2: Organize
Place downloaded data in the correct structure:
```bash
# Example for Hateful Memes
data/hate_symbols/
├── train/
│   ├── no_hate/    # Copy benign images here
│   └── hate/       # Copy hateful images here
└── test/
    ├── no_hate/
    └── hate/
```

### Step 3: Verify
```bash
python scripts/download_datasets.py --info
```

### Step 4: Update Metadata
If using custom datasets, update `metadata.json`:
```bash
# Count images and update metadata
cd data/nsfw
echo '{
  "dataset": "Custom NSFW",
  "categories": ["safe", "nsfw"],
  "num_classes": 2,
  "type": "real",
  "source": "open_nsfw"
}' > metadata.json
```

---

## Dataset Statistics (Current)

| Dataset | Train Images | Test Images | Categories |
|---------|-------------|-------------|------------|
| NSFW | 400 | 80 | safe, nsfw |
| Hate Symbols | 400 | 80 | no_hate, hate |
| Violence | 400 | 80 | no_violence, violence |

**Total**: 1,200 training images, 240 test images

---

## Data Augmentation

During training, the following augmentations are applied:

- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Random crop and resize
- Normalization (ImageNet stats)

See `configs/default.yaml` for augmentation settings.

---

## Ethical Considerations

### NSFW Content
- Handle with care and proper access controls
- Ensure compliance with local laws
- Use content filtering during collection
- Anonymize any identifiable information

### Hateful Content
- Review and validate annotations carefully
- Context matters - not all controversial content is hateful
- Consider cultural and regional differences
- Document annotation guidelines

### Violent Content
- Set clear boundaries for what constitutes violence
- Consider impact on annotators (use breaks, support)
- Distinguish between news/documentary vs glorified violence
- Age-gate and access-control trained models

---

## Data Quality Guidelines

### Image Requirements
- **Resolution**: Minimum 224x224 pixels
- **Format**: JPEG or PNG
- **Quality**: No heavily compressed or corrupted images
- **Diversity**: Vary lighting, angles, contexts

### Annotation Quality
- **Consistency**: Use clear annotation guidelines
- **Inter-annotator Agreement**: Aim for >80% agreement
- **Validation**: Have multiple annotators for ambiguous cases
- **Documentation**: Keep records of edge cases and decisions

### Balancing
- Aim for roughly equal samples per class
- Use weighted loss functions if imbalanced
- Consider oversampling minority class
- Monitor per-class performance

---

## Extending Datasets

### Adding New Categories

1. Create new directory structure:
```bash
mkdir -p data/new_category/{train,test}/{class1,class2}
```

2. Add images to appropriate folders

3. Create metadata.json:
```json
{
  "dataset": "New Category",
  "categories": ["class1", "class2"],
  "num_classes": 2
}
```

4. Update `configs/default.yaml` to include new category

5. Retrain model:
```bash
python scripts/train_image_model.py --dataset new_category
```

---

## References

### Papers
- **Hateful Memes**: Kiela et al., "The Hateful Memes Challenge" (2020)
- **Open NSFW**: Yahoo's NSFW detector technical report
- **Violence Detection**: Demarty et al., "MediaEval Violent Scenes Detection" (2014)

### Benchmarks
- **NSFW**: AUC-ROC > 0.93 (industry standard)
- **Hateful Memes**: AUROC > 0.70 (challenging baseline)
- **Violence**: Average Precision > 0.80

---

## Troubleshooting

### Issue: Dataset download fails
**Solution**: Many datasets require manual registration and download. Follow links above.

### Issue: Images in wrong format
**Solution**: Convert using:
```bash
# Convert PNG to JPEG
mogrify -format jpg *.png
```

### Issue: Unbalanced classes
**Solution**: 
1. Use weighted loss in training
2. Oversample minority class
3. Data augmentation on minority class

### Issue: Low quality images
**Solution**: Filter by:
```python
from PIL import Image
img = Image.open(path)
if img.size[0] < 224 or img.size[1] < 224:
    # Skip or resize
```

---

## Next Steps

1. ✅ Create dummy datasets for development
2. 🔲 Download real datasets for production
3. 🔲 Validate data quality
4. 🔲 Train baseline models
5. 🔲 Evaluate and iterate

---

**Last Updated**: August 15, 2026  
**Status**: Dummy datasets ready for development
