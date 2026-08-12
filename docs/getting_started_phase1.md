# Getting Started with Phase 1: Baseline Text Classification

This guide walks you through training your first toxic comment classifier and calibrating confidence thresholds.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- 10GB+ disk space

## Installation

### 1. Create a virtual environment

```bash
cd content-moderation-system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- PyTorch (for deep learning)
- Transformers (for pre-trained models)
- scikit-learn (for metrics)
- And other necessary packages

### 3. Verify GPU availability (optional)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Phase 1 Workflow

### Step 1: Download the Dataset

```bash
cd phase1_text_baseline
python download_data.py --output-dir data --analyze
```

This will:
- Download the Jigsaw Toxic Comment dataset from HuggingFace
- Save it as CSV files in `phase1_text_baseline/data/`
- Display dataset statistics

**Expected output:**
```
Dataset loaded successfully!
Train set shape: (159571, 8)
Columns: ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

Label Distribution in Training Set
==================================================
toxic                :  15294 ( 9.58%)
severe_toxic         :   1595 ( 1.00%)
obscene              :   8449 ( 5.29%)
threat               :    478 ( 0.30%)
insult               :   7877 ( 4.94%)
identity_hate        :   1405 ( 0.88%)
```

**Optional: Create a sample dataset for quick experiments**

```bash
python download_data.py --output-dir data --create-sample --sample-size 5000
```

This creates `data/train_sample.csv` with 5000 examples for rapid prototyping.

### Step 2: Configure Your Training

Edit `configs/baseline.yaml` to customize:

- **Model**: Choose from `bert-base-uncased`, `roberta-base`, `distilbert-base-uncased`
- **Batch size**: Adjust based on GPU memory (16 for 8GB GPU, 32 for 16GB+)
- **Learning rate**: Default 2e-5 works well
- **Epochs**: 3-5 epochs typical
- **Mixed precision**: Set to `true` for faster training on modern GPUs

**Quick start configuration (for testing):**
```yaml
training:
  use_sample: true  # Use small sample dataset
  batch_size: 32
  num_epochs: 2
  mixed_precision: true
```

### Step 3: Train the Classifier (Step 1.1)

```bash
python train_classifier.py --config configs/baseline.yaml
```

**What happens:**
1. Loads the dataset
2. Initializes a pre-trained transformer (BERT/RoBERTa)
3. Fine-tunes on toxic comment classification
4. Evaluates after each epoch
5. Saves the best model based on F1 score

**Training progress:**
```
Epoch 1/5
Training: 100%|████████| 4987/4987 [15:32<00:00, 5.35it/s, loss=0.0823]
Train Loss: 0.0965
Evaluating: 100%|████████| 554/554 [01:23<00:00, 6.62it/s]
Val F1: 0.7234, Val AUC: 0.9567

  toxic           - F1: 0.7615, AUC: 0.9673
  severe_toxic    - F1: 0.5821, AUC: 0.9821
  obscene         - F1: 0.8134, AUC: 0.9758
  threat          - F1: 0.4532, AUC: 0.9612
  insult          - F1: 0.7234, AUC: 0.9701
  identity_hate   - F1: 0.6012, AUC: 0.9745

✓ Saved best model (F1: 0.7234)
```

**Training time estimates:**
- Sample dataset (5K): ~5 minutes/epoch (GPU), ~20 minutes/epoch (CPU)
- Full dataset (160K): ~15 minutes/epoch (GPU), ~2 hours/epoch (CPU)

**Output files:**
```
phase1_text_baseline/models/
├── best_model.pt              # Model checkpoint
├── val_predictions.npy        # Validation set predictions
└── val_labels.npy            # Validation set labels
```

### Step 4: Calibrate Thresholds (Step 1.2)

```bash
python calibrate_thresholds.py --config configs/baseline.yaml
```

**What happens:**
1. Loads validation predictions from Step 1.1
2. Applies temperature scaling to calibrate probabilities
3. Finds optimal classification thresholds for each label
4. Defines enforcement tier boundaries:
   - **Auto-remove**: Very high confidence (P > 0.95)
   - **Human review**: Medium confidence (0.70 < P < 0.95)
   - **Auto-approve**: Very high confidence safe (P < 0.70)
5. Generates visualizations

**Expected output:**
```
Processing label: toxic
==================================================
Before calibration - ECE: 0.0823, MCE: 0.1452
Optimal temperature: 1.3421
After calibration  - ECE: 0.0234, MCE: 0.0612

Optimal threshold: 0.4532
  Precision: 0.8123
  Recall:    0.7845
  F1:        0.7982

Enforcement Tier Thresholds:
  Auto-remove (T_high):  0.9200
    - Volume: 3.45%
    - Precision: 0.9612
    - Recall: 0.3421
  
  Human-review (T_low to T_high):
    - Volume: 12.34%
    - Positive rate: 0.4521
  
  Auto-approve (T_low):   0.6800
    - Volume: 84.21%
    - False negative rate: 0.0123
```

**Output files:**
```
phase1_text_baseline/models/calibration/
├── calibration_results.json           # All calibration metrics
├── toxic_reliability.png              # Calibration plots
├── toxic_thresholds.png              # Threshold analysis
├── severe_toxic_reliability.png
├── ... (one per label)
```

### Step 5: Analyze Results

**View calibration results:**
```bash
python -c "import json; print(json.dumps(json.load(open('models/calibration/calibration_results.json')), indent=2))"
```

**Interpret the visualizations:**

1. **Reliability Diagrams** (`*_reliability.png`):
   - Perfect calibration: dots on the diagonal line
   - Before vs After shows calibration improvement
   - ECE (Expected Calibration Error) should decrease

2. **Threshold Analysis** (`*_thresholds.png`):
   - Precision-Recall curve shows tradeoffs
   - F1 vs Threshold helps choose optimal cutoff
   - Tier boundaries marked for enforcement

### Step 6: Test Inference (Optional)

Create a simple inference script to test your model:

```bash
python -c "
import torch
import numpy as np
from transformers import AutoTokenizer
from train_classifier import ToxicCommentClassifier
import json

# Load model
checkpoint = torch.load('models/best_model.pt')
config = checkpoint['config']

model = ToxicCommentClassifier(
    model_name=config['model']['name'],
    num_labels=config['model']['num_labels'],
    dropout=config['model']['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

# Load thresholds
with open('models/calibration/calibration_results.json') as f:
    calibration = json.load(f)

# Test text
text = 'You are an idiot and I hate you!'

# Tokenize
encoding = tokenizer(text, return_tensors='pt', max_length=512, 
                     padding='max_length', truncation=True)

# Predict
with torch.no_grad():
    outputs = model(**encoding)
    probs = torch.sigmoid(outputs['logits']).numpy()[0]

# Show results
print(f'Text: {text}\n')
for i, label in enumerate(config['labels']['names']):
    prob = probs[i]
    thresh = calibration['labels'][label]['optimal_threshold']
    pred = 'TOXIC' if prob > thresh else 'SAFE'
    print(f'{label:15s}: {prob:.4f} [{pred}]')
"
```

## Understanding the Results

### Key Metrics

- **AUC-ROC**: Measures overall classification performance (higher is better, >0.90 is good)
- **F1 Score**: Balance between precision and recall (0-1, higher is better)
- **Precision**: Of items flagged toxic, how many are actually toxic?
- **Recall**: Of all toxic items, how many did we catch?

### Calibration Quality

- **ECE (Expected Calibration Error)**: Average difference between confidence and accuracy
  - < 0.05: Well calibrated
  - 0.05-0.10: Reasonably calibrated
  - \> 0.10: Poorly calibrated

### Enforcement Tiers

The three-tier system balances automation and human oversight:

1. **Auto-remove** (high confidence toxic):
   - Automatically removed content
   - Target: 95%+ precision (minimize false positives)
   - Typically 2-5% of all content

2. **Human review** (uncertain):
   - Sent to moderators for decision
   - Contains mix of toxic and safe content
   - Typically 10-20% of all content

3. **Auto-approve** (high confidence safe):
   - Automatically approved content
   - Target: <2% false negative rate
   - Typically 75-90% of all content

## Troubleshooting

### Out of Memory Error

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size in config (try 8 or 4)
- Use `use_sample: true` for testing
- Enable mixed precision training
- Use a smaller model (distilbert-base-uncased)

### Poor Performance

If F1 < 0.60 or AUC < 0.85:
- Train for more epochs (try 5-10)
- Increase learning rate (try 3e-5 or 5e-5)
- Use focal loss for class imbalance: `focal_loss: true`
- Check data quality (missing values, text encoding issues)

### Slow Training

- Enable mixed precision: `mixed_precision: true`
- Increase batch size (if GPU memory allows)
- Reduce number of workers: `num_workers: 2`
- Use sample dataset for debugging

## Next Steps

Once you complete Phase 1:

1. ✅ You have a working toxic comment classifier
2. ✅ You understand confidence calibration
3. ✅ You have enforcement tier thresholds

**Move to Phase 2**: Multilingual routing and code-mix handling
- Train XLM-RoBERTa for regional languages
- Implement language identification
- Handle Hinglish, Kanglish, and other code-mixed text

## Additional Resources

- **Jigsaw Competition**: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **Calibration Paper**: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
- **Multi-label Classification**: https://scikit-learn.org/stable/modules/multiclass.html

## Questions?

Common questions:

**Q: Why calibration?**
A: Raw model outputs are poorly calibrated. A model might output 0.90 confidence but only be correct 70% of the time. Calibration fixes this.

**Q: Why three tiers?**
A: Balances automation (cost savings) with accuracy (safety). Pure automation would either miss too much (low recall) or over-flag (low precision).

**Q: Can I use my own dataset?**
A: Yes! Format it as CSV with columns: `text`, `label1`, `label2`, etc. Update the config accordingly.

**Q: How do I deploy this?**
A: Phase 5 covers production deployment with APIs, monitoring, and HITL interfaces.
