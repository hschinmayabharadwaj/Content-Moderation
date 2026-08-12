# Quick Start Guide

Welcome to the **Multilingual Content Moderation System**! This guide will get you started with Phase 1.

## 🚀 Installation (5 minutes)

### 1. Navigate to the project

```bash
cd content-moderation-system
```

### 2. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⏱️ **Time**: ~3-5 minutes depending on internet speed

## 🎯 Phase 1: Baseline Classifier (Your First Model!)

### Option A: Automated Pipeline (Recommended)

Run everything with one command:

```bash
cd phase1_text_baseline
bash run_phase1.sh
```

This will:
1. ✅ Download Jigsaw dataset (~160K toxic comments)
2. ✅ Train a RoBERTa classifier (~15 min/epoch on GPU)
3. ✅ Calibrate confidence thresholds
4. ✅ Generate visualizations

### Option B: Step-by-Step (For Learning)

**Step 1: Download Dataset**
```bash
cd phase1_text_baseline
python download_data.py --output-dir data --analyze
```

**Step 2: Train Model (Step 1.1)**
```bash
python train_classifier.py --config configs/baseline.yaml
```

**Step 3: Calibrate (Step 1.2)**
```bash
python calibrate_thresholds.py --config configs/baseline.yaml
```

### Option C: Quick Test (Sample Data)

For rapid experimentation with 5K examples:

```bash
cd phase1_text_baseline

# Download and create sample
python download_data.py --output-dir data --create-sample --sample-size 5000

# Edit configs/baseline.yaml: set use_sample: true

# Train on sample (~5 min/epoch)
python train_classifier.py --config configs/baseline.yaml

# Calibrate
python calibrate_thresholds.py --config configs/baseline.yaml
```

## 📊 Expected Results

After training, you should see:

```
Val F1: ~0.75-0.80
Val AUC: ~0.95-0.97

Per-label AUC:
  toxic:         0.96-0.97
  severe_toxic:  0.98-0.99
  obscene:       0.97-0.98
  threat:        0.96-0.98
  insult:        0.97-0.98
  identity_hate: 0.97-0.98
```

## 🎨 Visualizations

After calibration, check out:

```bash
open phase1_text_baseline/models/calibration/*.png
```

You'll see:
- **Reliability diagrams**: Shows calibration quality
- **Threshold analysis**: Precision-recall tradeoffs
- **Enforcement tiers**: Auto-remove, Human-review, Auto-approve

## 🔍 Interactive Exploration

Launch the Jupyter notebook:

```bash
cd phase1_text_baseline
jupyter notebook notebooks/phase1_tutorial.ipynb
```

Features:
- Dataset exploration
- Training monitoring
- Interactive inference
- Custom text testing

## 🛠️ Customization

### Change the Model

Edit `configs/baseline.yaml`:

```yaml
model:
  name: "bert-base-uncased"  # Options: roberta-base, distilbert-base-uncased
```

### Adjust Training

```yaml
training:
  batch_size: 16      # Increase for faster training (if GPU allows)
  learning_rate: 3e-5 # Try 2e-5 to 5e-5
  num_epochs: 5       # More epochs = better performance (but watch overfitting)
  mixed_precision: true  # Faster on modern GPUs
```

### Handle Class Imbalance

```yaml
training:
  use_class_weights: true   # Weight loss by class frequency
  focal_loss: true          # Use focal loss instead of BCE
  focal_gamma: 2.0          # Higher = focus more on hard examples
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Solution 1: Reduce batch size
# In configs/baseline.yaml, set batch_size: 8 or 4

# Solution 2: Use sample data
python download_data.py --create-sample --sample-size 5000
# Set use_sample: true in config

# Solution 3: Use smaller model
# Change model.name to "distilbert-base-uncased"
```

### Slow Training on CPU

```bash
# Enable on Apple Silicon
# In config: device: "mps"

# Use sample data for prototyping
# This reduces training time from hours to minutes
```

### Dataset Download Fails

If HuggingFace fails, download manually from Kaggle:

1. Go to: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2. Download `train.csv` and `test.csv`
3. Place in `phase1_text_baseline/data/`

## 📚 Understanding the System

### What is Calibration?

Raw model outputs are overconfident. A 90% prediction might only be correct 70% of the time.

**Temperature Scaling** fixes this by finding a "temperature" parameter that makes:
- Predicted probability ≈ Actual accuracy

### Why Three Tiers?

| Tier | Threshold | Action | Purpose |
|------|-----------|--------|---------|
| Auto-remove | P > 0.92 | Delete | High precision (few false positives) |
| Human review | 0.70 < P < 0.92 | Queue | Catch uncertain cases |
| Auto-approve | P < 0.70 | Allow | High recall (few false negatives) |

This balances:
- ⚡ Automation (reduces human workload)
- 🎯 Accuracy (catches toxic content)
- 💰 Cost (minimizes review queue)

### Multi-label Classification

Comments can have multiple labels:
- "You're an ugly idiot!" → `toxic=1, insult=1`
- "I hope you die!" → `toxic=1, threat=1`

The model predicts 6 independent probabilities (one per label).

## 📈 Performance Tips

### GPU Optimization

```yaml
training:
  batch_size: 32           # Larger batches
  mixed_precision: true    # FP16 training
  pin_memory: true         # Faster data loading
  num_workers: 4           # Parallel data loading
```

### Data Augmentation

```yaml
training:
  use_augmentation: true
  augmentation:
    synonym_replacement: true  # Replace words with synonyms
```

### Learning Rate Scheduling

```yaml
training:
  scheduler: "cosine"      # Options: linear, cosine, constant
  warmup_steps: 500        # Gradual learning rate increase
```

## 🎓 Learning Objectives

By completing Phase 1, you'll understand:

✅ **Fine-tuning transformers** (BERT/RoBERTa) for text classification  
✅ **Multi-label classification** with imbalanced data  
✅ **Confidence calibration** (temperature scaling)  
✅ **Threshold optimization** (precision-recall tradeoffs)  
✅ **Enforcement systems** (automated moderation tiers)  
✅ **PyTorch training loops** (with mixed precision)  
✅ **HuggingFace ecosystem** (datasets, transformers)  

## 🚀 Next Steps

After Phase 1:

### Phase 2: Multilingual Routing
- Language identification
- XLM-RoBERTa for code-mixed text (Hinglish, Kanglish)
- Regional language support

### Phase 3: Vision & OCR
- Image classification (NSFW, hate symbols)
- Text extraction from images
- Meme moderation

### Phase 4: Multimodal Fusion
- Cross-attention between text and images
- NLI-based defamation detection
- Contextual understanding

### Phase 5: Production & HITL
- Human moderator interface
- Drift monitoring
- Automated retraining
- A/B testing

## 📖 Documentation

- **Architecture**: `docs/architecture.md`
- **Getting Started**: `docs/getting_started_phase1.md`
- **API Reference**: Coming in Phase 5

## 🤝 Contributing

This is a research/educational project. Feel free to:
- Experiment with different models
- Try new calibration methods
- Add data augmentation techniques
- Optimize hyperparameters

## 📝 Citation

Dataset:
```
@misc{jigsaw-toxic,
  title={Toxic Comment Classification Challenge},
  author={Jigsaw/Conversation AI},
  howpublished={\url{https://kaggle.com/c/jigsaw-toxic-comment-classification-challenge}},
  year={2018}
}
```

## 💡 Tips for Success

1. **Start small**: Use sample data (5K examples) for quick iteration
2. **Monitor training**: Watch for overfitting after 3-5 epochs
3. **Calibrate always**: Raw probabilities are unreliable
4. **Visualize results**: Reliability diagrams tell the story
5. **Test interactively**: Use the notebook to understand model behavior

## ⏱️ Time Estimates

| Task | Sample (5K) | Full (160K) |
|------|-------------|-------------|
| Download | 2 min | 5 min |
| Training (GPU) | 10 min | 1 hour |
| Training (CPU) | 40 min | 4 hours |
| Calibration | 1 min | 3 min |
| **Total** | **~15 min** | **~1.2 hours** |

## 🎉 You're Ready!

Run your first model:

```bash
cd phase1_text_baseline
bash run_phase1.sh
```

Happy moderating! 🛡️
