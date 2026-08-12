# System Architecture Documentation

## Overview

This document describes the end-to-end architecture of the multilingual content moderation system, including data flow, model interactions, and decision-making logic.

## System Components

### 1. Ingestion Layer
- **Input Types**: Text posts, images, videos (future), mixed content
- **Preprocessing**: Text normalization, image resizing, format standardization
- **Rate Limiting**: Queue-based processing for scalability

### 2. Language Detection & Routing
- **FastText Language Identifier**: Quick language detection (99+ languages)
- **Router Logic**: 
  - English → Phase 1 classifier
  - Dravidian/Code-mix → Phase 2 specialized models
  - Unsupported → Fallback to multilingual XLM-RoBERTa

### 3. Text Classification Pipeline

#### Phase 1: Baseline (English)
```
Input Text
    ↓
Tokenization (BERT/RoBERTa tokenizer)
    ↓
Feature Extraction (768-dim embeddings)
    ↓
Classification Head (6 sigmoid outputs)
    ↓
Calibration Layer (temperature scaling)
    ↓
[toxic, severe_toxic, obscene, threat, insult, identity_hate] probabilities
```

**Model Architecture**:
- Base: `bert-base-uncased` or `roberta-base`
- Fine-tuning: Last 2-4 transformer layers
- Head: Dense(768 → 256 → 6) with dropout
- Loss: Binary Cross-Entropy with class weights
- Optimization: AdamW with warm-up schedule

**Calibration**:
- Method: Temperature scaling on validation set
- Goal: P(predicted) ≈ P(actual) across confidence bins
- Output: Calibrated confidence scores for threshold application

#### Threshold Enforcement Tiers
```
Tier 1 (Auto-Remove):     P > T_high (e.g., 0.95)
Tier 2 (Human Review):    T_low < P < T_high (e.g., 0.70 - 0.95)
Tier 3 (Auto-Approve):    P < T_low (e.g., 0.70)
```

Thresholds optimized via:
- Precision-Recall curves
- Cost-sensitive analysis (false positive vs false negative costs)
- Human review capacity constraints

### 4. Multilingual Pipeline (Phase 2)

#### Language Identification
- **Model**: FastText lid.176.bin
- **Fallback**: langdetect library for ambiguous cases
- **Code-mix Detection**: Character set analysis + n-gram patterns

#### XLM-RoBERTa Fine-tuning
- **Base Model**: `xlm-roberta-base` (270M params)
- **Training Data**: 
  - Dravidian-CodeMix datasets (HASOC, TRAC, custom)
  - Transliteration augmentation
  - Back-translation for data expansion
- **Domain Adaptation**: Continued pre-training on domain corpus

### 5. Vision & OCR Pipeline (Phase 3)

#### OCR Extraction
```
Image Input
    ↓
Preprocessing (grayscale, contrast, deskew)
    ↓
OCR Engine (EasyOCR/Tesseract)
    ↓
Text Regions + Confidence Scores
    ↓
Post-processing (spell-check, filtering)
    ↓
Extracted Text → Text Pipeline
```

**OCR Stack**:
- **EasyOCR**: Primary (neural, multi-language)
- **Tesseract**: Fallback (rule-based, fast)
- **PaddleOCR**: For complex layouts

#### Image Classification
```
Image Input
    ↓
ResNet50/EfficientNet Feature Extractor
    ↓
Multi-head Classifier
    ├── NSFW Detection (Safe/Unsafe)
    ├── Hate Symbol Detection (20+ symbols)
    └── Violence/Gore Detection
    ↓
Ensemble Predictions
```

**Training Strategy**:
- Transfer learning from ImageNet
- Fine-tune on:
  - NSFW datasets (e.g., OpenNSFW, custom)
  - Hate symbol datasets (ADL database, custom)
- Data augmentation: rotation, flip, color jitter

### 6. Multimodal Fusion (Phase 4)

#### Late-Fusion Architecture
```
Text Embedding (768-dim) ────┐
                              ├──→ Cross-Attention Layer ──→ Fusion Vector (512-dim)
Image Features (2048-dim) ───┤
                              ├──→ MLP Projection
OCR Text Embedding (768-dim)─┘

Fusion Vector
    ↓
Classification Head
    ↓
Final Decision + Confidence
```

**Cross-Attention Mechanism**:
```python
# Pseudo-code
Q = Linear(text_emb)      # Query
K = Linear(image_feat)    # Key
V = Linear(image_feat)    # Value

attention_scores = softmax(Q @ K^T / sqrt(d_k))
attended_features = attention_scores @ V
fused = concat(text_emb, attended_features, ocr_emb)
```

#### NLI-based Defamation Scoring
- **Model**: Fine-tuned RoBERTa-NLI
- **Method**: 
  1. Extract claims from content
  2. Compare against known facts (knowledge base)
  3. Score: Entailment, Contradiction, Neutral
- **Output**: Defamation risk score (0-1)

### 7. Human-in-the-Loop (Phase 5)

#### Review Queue Management
```
Incoming Content
    ↓
Model Predictions
    ↓
Routing Logic:
    - Auto-remove (high confidence harmful)
    - Queue for review (medium confidence)
    - Auto-approve (high confidence safe)
    ↓
Moderator Interface
    ↓
Human Decision + Feedback
    ↓
Update Training Data + Retrain Trigger
```

**Queue Prioritization**:
1. Viral content (high engagement)
2. High uncertainty (entropy)
3. Novel patterns (OOD detection)
4. User reports

#### Drift Monitoring
- **Statistical Tests**: 
  - KS-test on prediction distributions
  - Population Stability Index (PSI)
- **Performance Metrics**:
  - Rolling window accuracy
  - Moderator override rate
- **Retraining Triggers**:
  - PSI > 0.25 (significant drift)
  - Override rate > 15%
  - Weekly scheduled retraining

### 8. Feedback Loop

```
Moderator Overrides
    ↓
Label Corrections
    ↓
Active Learning Selection
    ↓
Incremental Fine-tuning
    ↓
A/B Testing (champion vs challenger)
    ↓
Model Deployment (if improved)
```

## Data Flow Diagram

```
┌─────────────┐
│   Content   │
│   Ingress   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Preprocessing  │
│  & Validation   │
└──────┬──────────┘
       │
       ├────────────────┐
       │                │
       ▼                ▼
┌──────────┐    ┌──────────────┐
│   Text   │    │    Image     │
│ Pipeline │    │   Pipeline   │
└────┬─────┘    └───────┬──────┘
     │                  │
     │  ┌───────────────┘
     │  │
     ▼  ▼
┌─────────────────┐
│ Multimodal      │
│ Fusion Layer    │
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  Threshold       │
│  Application     │
└────────┬─────────┘
         │
    ┌────┼────┐
    │    │    │
    ▼    ▼    ▼
┌──────┬──────┬──────┐
│Remove│Review│Approve│
└──────┴───┬──┴──────┘
           │
           ▼
    ┌────────────┐
    │ Moderator  │
    │  Interface │
    └──────┬─────┘
           │
           ▼
    ┌────────────┐
    │  Feedback  │
    │    Loop    │
    └────────────┘
```

## Model Specifications

### Baseline Text Classifier (Phase 1)
- **Parameters**: ~110M (BERT-base)
- **Input**: 512 tokens max
- **Output**: 6 class probabilities
- **Inference Time**: ~50ms (GPU), ~200ms (CPU)
- **Memory**: ~500MB

### Multilingual Model (Phase 2)
- **Parameters**: ~270M (XLM-RoBERTa-base)
- **Languages**: 100+ (focus on English + Dravidian)
- **Inference Time**: ~70ms (GPU)

### Vision Models (Phase 3)
- **OCR**: Variable (depends on image size)
- **Image Classifier**: ~25M params (ResNet50)
- **Inference Time**: ~100ms (GPU)

### Fusion Model (Phase 4)
- **Total Parameters**: ~400M
- **Inference Time**: ~150ms (GPU) for full pipeline
- **Batch Processing**: Recommended for efficiency

## Performance Targets (Research Context)

### Accuracy Metrics
- **Text Classification**: 
  - AUC-ROC > 0.95 per class
  - F1-score > 0.80 (balanced)
- **Calibration**: 
  - ECE (Expected Calibration Error) < 0.05
- **Image Classification**: 
  - NSFW detection > 0.93 AUC
  - Hate symbol detection > 0.90 recall @ 0.95 precision

### Throughput (Single GPU)
- **Text-only**: ~100 items/sec
- **Multimodal**: ~20 items/sec
- **Batch optimization**: 5-10x speedup possible

## Technology Decisions

### Why PyTorch?
- Better research flexibility
- Strong ecosystem (HuggingFace, torchvision)
- Dynamic computation graphs for experimentation
- TorchScript for production deployment

### Why HuggingFace Transformers?
- Pre-trained models across modalities
- Unified API for different architectures
- Active community and updates
- Integration with datasets library

### Why Calibration Matters
- Raw model outputs are poorly calibrated
- Threshold optimization requires reliable probabilities
- Human trust depends on accurate confidence
- Cost optimization (review queue sizing)

## Future Enhancements

1. **Video Understanding**: Frame-level + temporal analysis
2. **Audio Moderation**: Speech + background sound
3. **Context Awareness**: Thread-level reasoning
4. **Explainability**: LIME/SHAP for moderator insights
5. **Adversarial Robustness**: Defense against evasion attacks
6. **Privacy**: Differential privacy in training
7. **Fairness**: Bias detection and mitigation across demographics

## References

- Jigsaw Toxic Comment Classification Challenge
- "Attention Is All You Need" (Vaswani et al.)
- "XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning at Scale"
- "On Calibration of Modern Neural Networks" (Guo et al.)
- "Multimodal Deep Learning" (Ngiam et al.)
