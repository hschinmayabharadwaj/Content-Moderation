# Multilingual Content Moderation System with Human-in-the-Loop

A comprehensive, research-oriented content moderation system that progressively builds from baseline text classification to multimodal fusion with human oversight.

## Project Overview

This system is designed to moderate social media content across multiple languages, modalities (text, images, video), and contexts. It incorporates state-of-the-art deep learning techniques with human-in-the-loop capabilities for continuous improvement.

## Architecture

### Phase 1: Core Text Foundation & Threshold Calibration
- **Step 1.1**: Baseline Multi-Label Text Classifier (BERT/RoBERTa on Jigsaw)
- **Step 1.2**: Confidence Calibration & Threshold Optimization
- **Output**: Calibrated text classifier with enforcement tier boundaries

### Phase 2: Multilingual Routing (Local Adaptation)
- **Step 2.1**: Language Identification & Routing Layer
- **Step 2.2**: Fine-tuned XLM-RoBERTa for Dravidian Code-Mix (Hinglish, Kanglish, etc.)
- **Output**: Multi-language capable text moderation pipeline

### Phase 3: OCR & Computer Vision Pipelines
- **Step 3.1**: OCR Extraction Worker (Tesseract/EasyOCR + preprocessing)
- **Step 3.2**: Image Classification (NSFW/Hate Symbol Detection)
- **Output**: Parallel image and text-in-image processing capabilities

### Phase 4: Multimodal Fusion & Advanced Reasoning
- **Step 4.1**: PyTorch Late-Fusion Attention Layer
- **Step 4.2**: NLI-based Defamation Risk Scoring
- **Output**: Unified multimodal understanding with contextual reasoning

### Phase 5: Production Enforcement, HITL, & Continuous Retraining
- **Step 5.1**: Human-in-the-Loop Moderator Interface
- **Step 5.2**: Drift Monitor & Automated Retraining Pipeline
- **Output**: Production-ready system with continuous learning

## Project Structure

```
content-moderation-system/
├── phase1_text_baseline/          # Baseline text classification
│   ├── models/                    # Trained model checkpoints
│   ├── data/                      # Jigsaw dataset and preprocessed data
│   ├── notebooks/                 # Training and analysis notebooks
│   ├── configs/                   # Model and training configurations
│   ├── train_classifier.py        # Training script
│   ├── calibrate_thresholds.py    # Threshold calibration
│   └── evaluate.py                # Evaluation utilities
├── phase2_multilingual/           # Multilingual routing
│   ├── models/                    # XLM-RoBERTa checkpoints
│   ├── data/                      # Code-mix datasets
│   ├── notebooks/
│   ├── configs/
│   ├── language_router.py         # Language identification
│   └── train_multilingual.py     # Fine-tuning scripts
├── phase3_vision_ocr/             # Vision and OCR pipelines
│   ├── models/                    # Vision model checkpoints
│   ├── data/                      # Image datasets
│   ├── notebooks/
│   ├── configs/
│   ├── ocr_worker.py              # OCR extraction
│   └── image_classifier.py        # NSFW/hate symbol detection
├── phase4_multimodal_fusion/      # Fusion layer
│   ├── models/
│   ├── data/
│   ├── notebooks/
│   ├── configs/
│   ├── fusion_layer.py            # Cross-modal attention
│   └── nli_defamation.py          # Defamation scoring
├── phase5_hitl_production/        # HITL and orchestration
│   ├── models/
│   ├── data/
│   ├── notebooks/
│   ├── configs/
│   ├── moderator_ui.py            # Web interface for human review
│   ├── drift_monitor.py           # Model drift detection
│   └── orchestrator.py            # End-to-end pipeline
├── utils/                         # Shared utilities
│   ├── data_utils.py
│   ├── model_utils.py
│   ├── metrics.py
│   └── visualization.py
├── tests/                         # Unit and integration tests
├── docs/                          # Documentation
│   ├── architecture.md
│   ├── api.md
│   └── deployment.md
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
└── README.md                      # This file
```

## Technology Stack

- **Deep Learning**: PyTorch (primary), TensorFlow/Keras (secondary)
- **Transformers**: HuggingFace Transformers library
- **OCR**: Tesseract, EasyOCR, PaddleOCR
- **Vision**: torchvision, timm (PyTorch Image Models)
- **Data**: HuggingFace datasets, pandas, numpy
- **Visualization**: matplotlib, seaborn, wandb
- **Web UI**: Streamlit/Gradio for HITL interface

## Key Features

1. **Multi-label Classification**: Toxic, severe_toxic, obscene, threat, insult, identity_hate
2. **Confidence Calibration**: Temperature scaling and Platt scaling for reliable probabilities
3. **Multilingual Support**: English + Dravidian code-mix languages
4. **Multimodal Understanding**: Text + Image + OCR fusion
5. **Human-in-the-Loop**: Active learning with moderator feedback
6. **Continuous Learning**: Automated retraining based on drift detection

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download datasets:
```bash
python phase1_text_baseline/download_data.py
```

3. Train baseline classifier:
```bash
python phase1_text_baseline/train_classifier.py --config configs/baseline.yaml
```

4. Follow phase-by-phase implementation guide in `docs/architecture.md`

## Research Goals

- Explore state-of-the-art techniques in content moderation
- Understand confidence calibration and threshold optimization
- Learn multimodal fusion architectures
- Implement production-grade ML systems with HITL
- Gain hands-on experience with transformer models across modalities

## License

Research/Educational Use

## Acknowledgments

- Jigsaw/Conversation AI for the Toxic Comment Classification dataset
- HuggingFace for transformers and datasets
- PyTorch and TensorFlow communities

- TEST 
