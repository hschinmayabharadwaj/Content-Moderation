# Phase 2: Multilingual Routing & Code-Mix Support

## Overview

Phase 2 extends the content moderation system to support 100+ languages and code-mixed text (like Hinglish, Kanglish, Tanglish).

### Key Features

- ✅ **Language Identification**: FastText-based detection for 176 languages
- ✅ **Code-Mix Detection**: Identifies mixed-language text (e.g., "Aaj main bahut khush hoon yaar")
- ✅ **XLM-RoBERTa**: Multilingual transformer supporting 100+ languages
- ✅ **Intelligent Routing**: Automatic model selection based on language
- ✅ **Unified Pipeline**: Single API for all languages

## Quick Start

```bash
# Run complete Phase 2 pipeline
bash run_phase2.sh
```

## Components

### 1. Language Identification (`language_identifier.py` - 511 lines)

**Features:**
- FastText lid.176.bin model (176 languages)
- Script mixing detection (Latin + Devanagari/Tamil/etc.)
- Code-mix pattern recognition (Hinglish, Kanglish, Tanglish, Tenglish)
- Fallback to langdetect for edge cases
- Confidence thresholding

**Usage:**
```python
from language_identifier import LanguageIdentifier, LanguageRouter

identifier = LanguageIdentifier(
    fasttext_model_path='models/lid.176.bin',
    confidence_threshold=0.5
)

result = identifier.identify("Aaj main bahut khush hoon")
# {'language': 'hi', 'confidence': 0.89, 'is_code_mixed': True, 'code_mix_type': 'hinglish'}

router = LanguageRouter(identifier)
route = router.route("Your text here")
# {'model': 'phase2', 'language_info': {...}, 'reasoning': '...'}
```

### 2. Dataset Preparation (`prepare_datasets.py` - 433 lines)

**Features:**
- Synthetic code-mixed dataset generation
- HASOC dataset loader (Hindi-English hate speech)
- TRAC dataset loader (Aggression detection)
- Combines Phase 1 English data with multilingual data
- Stratified train/val splits by language

**Usage:**
```bash
# With sample data
python prepare_datasets.py --output-dir data --sample-size 5000

# Include Phase 1 English data
python prepare_datasets.py --output-dir data --include-phase1

# Check real dataset availability
python prepare_datasets.py --check-availability
```

**Output:**
- `data/multilingual_train.csv`
- `data/multilingual_train_split.csv`
- `data/multilingual_val_split.csv`

### 3. XLM-RoBERTa Training (`train_multilingual.py` - 571 lines)

**Features:**
- Fine-tune XLM-RoBERTa-base (270M params, 100 languages)
- Binary classification (toxic vs safe)
- Optional language-specific adapter layers
- Mixed precision training
- Per-language evaluation
- Early stopping and checkpointing

**Usage:**
```bash
python train_multilingual.py --config configs/xlm_roberta.yaml
```

**Model Architecture:**
```
Input Text → XLM-RoBERTa Encoder (768-dim)
           ↓
     [CLS] Token
           ↓
  Optional Language Adapter
           ↓
     Classification Head
     (768 → 256 → 1)
           ↓
    Binary Prediction
```

**Configuration (`configs/xlm_roberta.yaml`):**
```yaml
model:
  name: "xlm-roberta-base"
  max_length: 512
  dropout: 0.1
  hidden_size: 256

training:
  batch_size: 16
  learning_rate: 2.0e-5
  num_epochs: 5
  mixed_precision: true
```

### 4. Unified Inference (`unified_inference.py` - 416 lines)

**Features:**
- Combines Phase 1 (English) and Phase 2 (Multilingual) models
- Automatic language detection and routing
- Applies calibrated thresholds
- Returns enforcement tier decisions
- Batch processing support

**Usage:**
```python
from unified_inference import UnifiedContentModerator

moderator = UnifiedContentModerator(
    phase1_model_path='../phase1_text_baseline/models/best_model.pt',
    phase2_model_path='models/best_model.pt',
    language_model_path='models/lid.176.bin'
)

result = moderator.moderate("Aaj tu bahut pagal hai yaar")
# {
#   'action': 'human_review',
#   'predictions': {'toxic': 0.745},
#   'language_info': {...},
#   'model_used': 'phase2',
#   'reasoning': 'Code-mixed text (hinglish)'
# }
```

### 5. Evaluation (`evaluate_multilingual.py` - 401 lines)

**Features:**
- Per-language performance metrics
- Code-mix detection accuracy
- Routing accuracy evaluation
- Visual comparisons (plots)
- Markdown report generation

**Usage:**
```bash
python evaluate_multilingual.py \
    --predictions models/val_predictions.npy \
    --labels models/val_labels.npy \
    --languages models/val_languages.json \
    --output-dir evaluation
```

**Outputs:**
- `evaluation/evaluation_report.md`
- `evaluation/per_language_performance.png`
- `evaluation/language_comparison.png`

## File Structure

```
phase2_multilingual/
├── language_identifier.py        (511 lines) - Language detection & routing
├── prepare_datasets.py           (433 lines) - Dataset preparation
├── train_multilingual.py         (571 lines) - XLM-RoBERTa training
├── unified_inference.py          (416 lines) - Unified pipeline
├── evaluate_multilingual.py      (401 lines) - Evaluation benchmarks
├── run_phase2.sh                 (229 lines) - Automation script
├── configs/
│   └── xlm_roberta.yaml          (63 lines)  - Training config
├── models/                       - Model checkpoints
├── data/                         - Datasets
└── evaluation/                   - Results & reports
```

**Total: 2,624 lines of code**

## Performance Expectations

### Language Detection
- Pure language: ~99% accuracy (major languages)
- Code-mix detection: ~85-90% accuracy
- Code-mix type: ~80-85% accuracy

### Classification Performance (Full Dataset)

| Language Type | F1 Score | AUC-ROC |
|---------------|----------|---------|
| English | 0.75-0.80 | 0.95-0.97 |
| Hindi (Devanagari) | 0.70-0.75 | 0.90-0.93 |
| Hinglish (Romanized) | 0.65-0.72 | 0.88-0.91 |
| Tamil/Telugu/Kannada | 0.68-0.73 | 0.89-0.92 |
| Code-mixed (avg) | 0.65-0.70 | 0.87-0.90 |

### Training Time
- Sample (10K): ~30 min (GPU), ~2 hours (CPU)
- Full (50K+): ~2-3 hours (GPU), ~10 hours (CPU)

### Inference Speed
- Single prediction: ~50-70ms (GPU)
- Batch (32): ~10-15 items/sec (GPU)

## Supported Languages

### Major Languages (with training data)
- English, Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Punjabi, Gujarati, Marathi

### Code-Mixed Varieties
- **Hinglish**: Hindi + English (most common)
- **Tanglish**: Tamil + English
- **Tenglish**: Telugu + English  
- **Kanglish**: Kannada + English

### All XLM-RoBERTa Languages (100+)
See full list: https://huggingface.co/xlm-roberta-base

## Datasets

### Included
- ✅ Synthetic code-mixed data (demonstration)
- ✅ Phase 1 English data (optional)

### Recommended (Manual Download)

1. **HASOC (Hate Speech)**
   - URL: https://hasocfire.github.io/hasoc/
   - Languages: Hindi, English, Hinglish
   - Size: ~15K examples

2. **TRAC (Aggression)**
   - URL: https://sites.google.com/view/trac2/
   - Languages: Hindi, Bengali, Tamil
   - Size: ~10K examples per language

3. **Custom Collection**
   - Twitter API for code-mixed content
   - Reddit multilingual communities
   - News comments

## Workflows

### Training Workflow

```
1. Download FastText model
   ↓
2. Prepare datasets
   ↓
3. Train XLM-RoBERTa
   ↓
4. Evaluate per-language
   ↓
5. Test unified inference
```

### Inference Workflow

```
Input Text
   ↓
Language Identification
   ├─ English (high conf) → Phase 1 Model
   ├─ Code-mixed → Phase 2 Model
   ├─ Regional lang → Phase 2 Model
   └─ Unknown → Phase 2 Model (fallback)
   ↓
Get Predictions
   ↓
Apply Thresholds
   ↓
Return Action (auto-remove/review/approve)
```

## Customization

### Add New Language

1. **Get training data** in your language
2. **Add to dataset prep**:
   ```python
   df_new['language'] = 'your_lang'
   ```
3. **Retrain**:
   ```bash
   python train_multilingual.py
   ```

### Add Code-Mix Pattern

In `language_identifier.py`:
```python
self.code_mix_markers['your_mix'] = [
    r'\b(pattern1|pattern2)\b',
]
```

### Use Different Model

In `configs/xlm_roberta.yaml`:
```yaml
model:
  name: "xlm-roberta-large"  # 550M params, better accuracy
  # or
  name: "distilbert-base-multilingual-cased"  # Smaller, faster
```

## Troubleshooting

### Issue: FastText model not downloading
**Solution:** Download manually:
```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
mv lid.176.bin phase2_multilingual/models/
```

### Issue: Out of memory during training
**Solution:** Reduce batch size:
```yaml
training:
  batch_size: 8  # or 4
```

### Issue: Poor code-mix performance
**Solutions:**
1. Collect more code-mixed training data
2. Add transliteration normalization
3. Use data augmentation
4. Increase training epochs

### Issue: Language misidentification
**Solution:** Lower confidence threshold:
```python
identifier = LanguageIdentifier(confidence_threshold=0.3)
```

## Next Steps

After Phase 2:
- ✅ 100+ language support
- ✅ Code-mixed text handling
- ✅ Intelligent routing

**Move to Phase 3**: Vision & OCR Pipelines
- Image classification (NSFW, hate symbols)
- OCR for text-in-images
- Multimodal content understanding

## Resources

- **FastText Language ID**: https://fasttext.cc/docs/en/language-identification.html
- **XLM-RoBERTa**: https://huggingface.co/xlm-roberta-base
- **HASOC**: http://hasocfire.github.io/hasoc/
- **TRAC**: https://sites.google.com/view/trac2/
- **Code-Mixing Research**: https://aclanthology.org/

## Citation

If using this for research:
```bibtex
@article{conneau2019unsupervised,
  title={Unsupervised Cross-lingual Representation Learning at Scale},
  author={Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1911.02116},
  year={2019}
}
```

---

**Last Updated**: August 2026  
**Status**: ✅ IMPLEMENTATION COMPLETE  
**Ready For**: Training and Deployment
