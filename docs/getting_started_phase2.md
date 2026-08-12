# Getting Started with Phase 2: Multilingual Routing

This guide walks you through extending your content moderation system to support multiple languages and code-mixed text (like Hinglish, Kanglish, etc.).

## What Phase 2 Adds

- **100+ Language Support**: XLM-RoBERTa handles English, Hindi, Tamil, Telugu, Kannada, Bengali, and many more
- **Code-Mix Detection**: Automatically identifies and handles mixed-language text (Hinglish: "Aaj main bahut khush hoon yaar")
- **Intelligent Routing**: Directs English text to Phase 1 model, regional/code-mixed to Phase 2
- **Unified Pipeline**: Single inference API that handles all languages

## Prerequisites

- ✅ Phase 1 completed (baseline English model trained)
- Python 3.8+
- CUDA GPU recommended (XLM-RoBERTa is larger than BERT)
- 12GB+ disk space (for models and datasets)

## Quick Start (15-30 minutes)

### Option A: Automated Pipeline

```bash
cd phase2_multilingual
bash run_phase2.sh
```

This will:
1. Download FastText language identification model (131 MB)
2. Prepare sample multilingual dataset
3. Train XLM-RoBERTa (or use existing model)
4. Evaluate per-language performance
5. Test unified inference

### Option B: Step-by-Step

Follow along to understand each component.

---

## Step 1: Language Identification Setup

### Download FastText Model

```bash
cd phase2_multilingual
python language_identifier.py
```

This downloads `lid.176.bin` (131 MB) which identifies 176 languages.

**Alternative manual download:**
```bash
mkdir -p models
wget -O models/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

### Test Language Detection

```python
from language_identifier import LanguageIdentifier, LanguageRouter

# Initialize
identifier = LanguageIdentifier(
    fasttext_model_path='models/lid.176.bin',
    confidence_threshold=0.5
)

# Test
texts = [
    "This is English text",
    "Aaj main bahut khush hoon yaar",  # Hinglish
    "यह हिंदी है",  # Hindi (Devanagari)
]

for text in texts:
    result = identifier.identify(text)
    print(f"Text: {text}")
    print(f"  Language: {result['language']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Code-mixed: {result['is_code_mixed']}")
    if result['code_mix_type']:
        print(f"  Type: {result['code_mix_type']}")
```

**Expected Output:**
```
Text: This is English text
  Language: en
  Confidence: 0.998
  Code-mixed: False

Text: Aaj main bahut khush hoon yaar
  Language: hi
  Confidence: 0.892
  Code-mixed: True
  Type: hinglish
```

---

## Step 2: Prepare Multilingual Datasets

### Option A: Use Sample Data (Quick Testing)

```bash
python prepare_datasets.py \
    --output-dir data \
    --include-phase1 \
    --sample-size 5000
```

This creates:
- 5,000 synthetic code-mixed examples (Hinglish)
- 5,000 English examples from Phase 1 (if available)

**Output:**
```
data/
├── multilingual_train.csv           # Combined dataset
├── multilingual_train_split.csv     # Training split
└── multilingual_val_split.csv       # Validation split
```

### Option B: Use Real Datasets (Production)

**Download HASOC (Hate Speech and Offensive Content)**

1. Visit: https://hasocfire.github.io/hasoc/2019/dataset.html
2. Request access and download
3. Place files in: `data/hasoc2019/`
   - `hindi_train.csv`
   - `hinglish_train.csv`
   - `english_train.csv`

**Download TRAC (Trolling, Aggression, Cyberbullying)**

1. Visit: https://sites.google.com/view/trac2/shared-task
2. Download datasets
3. Place in: `data/trac2020/`

**Prepare combined dataset:**
```bash
python prepare_datasets.py \
    --output-dir data \
    --include-phase1 \
    --check-availability  # Check which datasets are found
```

### Dataset Format

Your CSV should have these columns:
- `text`: The content text
- `label`: Binary label (0 = safe, 1 = toxic)
- `language`: Language code (e.g., 'english', 'hindi', 'hinglish')
- `is_code_mixed`: Boolean indicating code-mixing

---

## Step 3: Train XLM-RoBERTa

### Configure Training

Edit `configs/xlm_roberta.yaml`:

```yaml
model:
  name: "xlm-roberta-base"  # 270M parameters
  max_length: 512
  dropout: 0.1
  hidden_size: 256

training:
  batch_size: 16        # Reduce to 8 if OOM
  learning_rate: 2.0e-5
  num_epochs: 5
  mixed_precision: true # Faster on modern GPUs
```

### Train the Model

```bash
python train_multilingual.py --config configs/xlm_roberta.yaml
```

**Training Progress:**
```
Epoch 1/5
Training: 100%|████████| 625/625 [12:34<00:00, 1.20s/it, loss=0.342]
Train Loss: 0.3812
Evaluating: 100%|████████| 70/70 [00:52<00:00, 1.34it/s]
Val F1: 0.7234, Val AUC: 0.9145
Val Precision: 0.7812, Val Recall: 0.6723

Per-language performance:
  english         - F1: 0.7615, Count: 500
  hinglish        - F1: 0.6821, Count: 450
  hindi           - F1: 0.7102, Count: 50

✓ Saved best model (F1: 0.7234)
```

**Training Time Estimates:**
- Sample dataset (10K): ~30 min (GPU), ~2 hours (CPU)
- Full dataset (50K+): ~2-3 hours (GPU), ~10 hours (CPU)

**Output Files:**
```
models/
├── best_model.pt                  # Model checkpoint
├── val_predictions.npy            # Validation predictions
├── val_labels.npy                # Validation labels
└── per_language_metrics.json     # Performance per language
```

---

## Step 4: Evaluate Multilingual Performance

```bash
python evaluate_multilingual.py \
    --predictions models/val_predictions.npy \
    --labels models/val_labels.npy \
    --languages models/val_languages.json \
    --output-dir evaluation
```

**Generates:**
- `evaluation/evaluation_report.md` - Detailed metrics report
- `evaluation/per_language_performance.png` - Visual comparison
- `evaluation/language_comparison.png` - Side-by-side analysis

**Example Report:**
```markdown
# Multilingual Content Moderation - Evaluation Report

## Per-Language Performance
| Language | Count | F1    | Precision | Recall | AUC   |
|----------|-------|-------|-----------|--------|-------|
| english  | 5000  | 0.761 | 0.812     | 0.715  | 0.967 |
| hinglish | 4500  | 0.682 | 0.721     | 0.647  | 0.891 |
| hindi    | 500   | 0.710 | 0.745     | 0.678  | 0.908 |

**Average F1**: 0.718 (±0.041)
**Average AUC**: 0.922 (±0.042)

## Code-Mix Detection Performance
- Detection Accuracy: 0.893
- Detection F1: 0.867
- Type Accuracy: 0.812 (hinglish vs tanglish vs kanglish)
```

---

## Step 5: Unified Inference Pipeline

Combine Phase 1 (English) and Phase 2 (Multilingual) models:

```python
from unified_inference import UnifiedContentModerator

# Initialize
moderator = UnifiedContentModerator(
    phase1_model_path='../phase1_text_baseline/models/best_model.pt',
    phase2_model_path='models/best_model.pt',
    language_model_path='models/lid.176.bin',
    phase1_calibration_path='../phase1_text_baseline/models/calibration/calibration_results.json'
)

# Test texts
texts = [
    "You are an idiot!",                      # English → Phase 1
    "Aaj tu bahut pagal hai yaar",           # Hinglish → Phase 2
    "This is a great post!",                  # English positive
    "Kya bakwas hai ye, stop this nonsense",  # Hinglish negative
]

# Moderate
for text in texts:
    result = moderator.moderate(text, return_details=True)
    
    print(f"\nText: {text}")
    print(f"  Language: {result['language_info']['language']}")
    print(f"  Model: {result['model_used'].upper()}")
    print(f"  Predictions: {result['predictions']}")
    print(f"  ACTION: {result['action'].upper()}")
```

**Output:**
```
Text: You are an idiot!
  Language: en
  Model: PHASE1
  Predictions: {'toxic': 0.912, 'insult': 0.887, ...}
  ACTION: AUTO_REMOVE

Text: Aaj tu bahut pagal hai yaar
  Language: hi
  Model: PHASE2
  Predictions: {'toxic': 0.745}
  ACTION: HUMAN_REVIEW

Text: This is a great post!
  Language: en
  Model: PHASE1
  Predictions: {'toxic': 0.023, 'insult': 0.012, ...}
  ACTION: AUTO_APPROVE
```

---

## Understanding the Results

### Language Detection Accuracy

The system can:
- ✅ Detect 176 languages with FastText
- ✅ Identify script mixing (Latin + Devanagari)
- ✅ Recognize romanized code-mix patterns
- ✅ Fall back to langdetect for edge cases

**Typical Accuracy:**
- Pure language detection: ~99% for major languages
- Code-mix detection: ~85-90% (depends on mixing ratio)
- Code-mix type identification: ~80-85%

### Model Performance by Language

**Expected Performance (Full Dataset):**

| Language Type | F1 Score | Notes |
|---------------|----------|-------|
| English | 0.75-0.80 | Best (Phase 1 specialized) |
| Hindi (Devanagari) | 0.70-0.75 | Good with sufficient data |
| Hinglish (Roman) | 0.65-0.72 | Challenging due to spelling variations |
| Tamil/Telugu/Kannada | 0.68-0.73 | Depends on training data size |
| Code-mixed (any) | 0.65-0.70 | Most challenging |

### Routing Accuracy

The router decides which model to use:
- English (high confidence) → Phase 1
- Code-mixed → Phase 2
- Regional languages → Phase 2
- Unknown/low confidence → Phase 2 (fallback)

**Typical Routing Stats:**
- Phase 1 usage: 60-70% (if dataset is mostly English)
- Phase 2 usage: 30-40%
- Routing accuracy: 95%+ (rarely misroutes)

---

## Customization

### Add Support for New Languages

1. **Get Training Data**
   - Find toxic comment datasets in your target language
   - Ensure CSV format with `text`, `label` columns

2. **Add to Dataset Preparation**
   ```python
   # In prepare_datasets.py
   df_new_lang = pd.read_csv('your_language_data.csv')
   df_new_lang['language'] = 'your_language_code'
   df_new_lang['is_code_mixed'] = False
   ```

3. **Update Configuration**
   ```yaml
   # In configs/xlm_roberta.yaml
   languages:
     supported: [..., "your_language_code"]
   ```

4. **Retrain**
   ```bash
   python train_multilingual.py
   ```

### Improve Code-Mix Detection

1. **Add Patterns** in `language_identifier.py`:
   ```python
   self.code_mix_markers = {
       'your_mix': [
           r'\b(pattern1|pattern2)\b',
       ]
   }
   ```

2. **Collect More Data**
   - Code-mixed text is hardest to label
   - Consider active learning (Phase 5) to improve
   - Twitter/Reddit often have code-mixed content

### Fine-tune for Specific Domain

```bash
# Start from pretrained, continue on your domain data
python train_multilingual.py \
    --config configs/xlm_roberta.yaml \
    --resume models/best_model.pt \
    --domain-data your_domain_data.csv
```

---

## Troubleshooting

### Language Model Download Fails

**Manual download:**
```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
mv lid.176.bin phase2_multilingual/models/
```

### Out of Memory (OOM)

**Solution 1: Reduce batch size**
```yaml
# In configs/xlm_roberta.yaml
training:
  batch_size: 8  # or even 4
```

**Solution 2: Use gradient accumulation**
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size = 16
```

**Solution 3: Use smaller model**
```yaml
model:
  name: "xlm-roberta-base"  # Already smallest
  # Consider using distilbert-base-multilingual-cased
```

### Poor Performance on Code-Mixed Text

**Likely causes:**
1. **Insufficient training data**
   - Solution: Collect more code-mixed examples
   - Try data augmentation (word substitution)

2. **Transliteration variations**
   - "kar" vs "ker" vs "kr" (all mean "do")
   - Solution: Add transliteration normalization

3. **Imbalanced mixing ratios**
   - Some texts are 90% English, 10% Hindi
   - Solution: Sample by mixing ratio

### FastText Not Detecting Language Correctly

**Check confidence threshold:**
```python
# Lower threshold for more permissive detection
identifier = LanguageIdentifier(
    confidence_threshold=0.3  # Default is 0.5
)
```

**Enable fallback:**
```python
identifier = LanguageIdentifier(
    use_fallback=True  # Uses langdetect as backup
)
```

---

## Performance Optimization

### Batch Inference

```python
# Instead of one-by-one
for text in texts:
    result = moderator.moderate(text)

# Use batch processing
results = moderator.batch_moderate(texts)
```

**Speedup: 5-10x**

### Model Quantization (Future)

```python
# Convert to int8 for faster inference
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**Speedup: 2-4x, Accuracy loss: <1%**

---

## Next Steps

After completing Phase 2:

1. ✅ You support 100+ languages
2. ✅ You handle code-mixed text
3. ✅ You have intelligent routing

**Move to Phase 3**: Vision & OCR
- Image classification (NSFW, hate symbols)
- OCR for text-in-images
- Meme moderation

## Resources

- **FastText**: https://fasttext.cc/docs/en/language-identification.html
- **XLM-RoBERTa Paper**: https://arxiv.org/abs/1911.02116
- **HASOC**: http://hasocfire.github.io/hasoc/
- **TRAC**: https://sites.google.com/view/trac2/
- **Code-Mixing Research**: https://aclanthology.org/

## FAQ

**Q: Can I use only Phase 2 (skip Phase 1)?**
A: Yes! Phase 2 handles English too. Phase 1 is optional for better English-specific performance.

**Q: How many languages does XLM-RoBERTa support?**
A: 100 languages. See full list: https://huggingface.co/xlm-roberta-base

**Q: What's the best way to get code-mixed training data?**
A: Twitter API, Reddit, or competitions like HASOC/TRAC. Alternatively, synthetic generation with rules.

**Q: Can I deploy this to production?**
A: Phase 2 is research-ready. For production, see Phase 5 (HITL, monitoring, APIs).

**Q: How do I add my custom language?**
A: If XLM-RoBERTa supports it (check HuggingFace), just add training data and retrain!
