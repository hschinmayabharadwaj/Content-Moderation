#!/bin/bash

# Phase 2: Multilingual Routing - End-to-End Pipeline
# This script runs the complete Phase 2 workflow:
# 1. Downloads FastText language model
# 2. Prepares multilingual datasets
# 3. Trains XLM-RoBERTa
# 4. Evaluates multilingual performance

set -e  # Exit on error

echo "=========================================="
echo "Content Moderation System - Phase 2"
echo "Multilingual Routing & Code-Mix Support"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR=${DATA_DIR:-"data"}
CONFIG_FILE=${CONFIG_FILE:-"configs/xlm_roberta.yaml"}
INCLUDE_PHASE1=${INCLUDE_PHASE1:-true}

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Data directory: $DATA_DIR"
echo "  Config file: $CONFIG_FILE"
echo "  Include Phase 1 English data: $INCLUDE_PHASE1"

# Step 1: Download FastText Language Model
echo -e "\n${GREEN}Step 1: Setting up Language Identification${NC}"
echo "=========================================="

if [ -f "models/lid.176.bin" ]; then
    echo "✓ FastText language model already exists"
else
    echo "Downloading FastText language identification model (131 MB)..."
    python language_identifier.py
    
    if [ ! -f "models/lid.176.bin" ]; then
        echo -e "${YELLOW}FastText model not downloaded automatically.${NC}"
        echo "Downloading manually..."
        mkdir -p models
        wget -O models/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    fi
fi

echo -e "${GREEN}✓ Language identification ready${NC}"

# Step 2: Test Language Detection
echo -e "\n${GREEN}Step 2: Testing Language Detection${NC}"
echo "=========================================="

python -c "
from language_identifier import LanguageIdentifier, LanguageRouter

# Initialize
identifier = LanguageIdentifier(
    fasttext_model_path='models/lid.176.bin',
    confidence_threshold=0.5
)

router = LanguageRouter(identifier)

# Test cases
tests = [
    'This is English text',
    'Aaj main bahut khush hoon yaar',
    'यह हिंदी है',
]

print('\nLanguage Detection Test:')
for text in tests:
    result = identifier.identify(text)
    print(f'  {text[:30]:30s} → {result[\"language\"]:10s} ({result[\"confidence\"]:.2f})')
" || echo -e "${YELLOW}Warning: Language detection test had issues${NC}"

# Step 3: Prepare Datasets
echo -e "\n${GREEN}Step 3: Preparing Multilingual Datasets${NC}"
echo "=========================================="

if [ -f "$DATA_DIR/multilingual_train_split.csv" ]; then
    echo "Dataset already prepared at $DATA_DIR/multilingual_train_split.csv"
    read -p "Re-prepare dataset? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ "$INCLUDE_PHASE1" = true ]; then
            python prepare_datasets.py --output-dir "$DATA_DIR" --include-phase1
        else
            python prepare_datasets.py --output-dir "$DATA_DIR"
        fi
    fi
else
    echo "Preparing multilingual datasets..."
    echo ""
    echo "Note: This will create a sample code-mixed dataset for demonstration."
    echo "For production, download HASOC or TRAC datasets (see documentation)."
    echo ""
    
    if [ "$INCLUDE_PHASE1" = true ]; then
        python prepare_datasets.py --output-dir "$DATA_DIR" --include-phase1 --sample-size 5000
    else
        python prepare_datasets.py --output-dir "$DATA_DIR" --sample-size 5000
    fi
fi

# Check if datasets exist
if [ ! -f "$DATA_DIR/multilingual_train_split.csv" ]; then
    echo -e "${RED}✗ Dataset preparation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Datasets ready${NC}"

# Step 4: Train XLM-RoBERTa
echo -e "\n${GREEN}Step 4: Training XLM-RoBERTa (Step 2.2)${NC}"
echo "=========================================="

if [ -f "models/best_model.pt" ]; then
    echo "Model already exists at models/best_model.pt"
    read -p "Re-train? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping training..."
    else
        python train_multilingual.py --config "$CONFIG_FILE"
    fi
else
    echo "Training XLM-RoBERTa on multilingual data..."
    echo "This may take 1-3 hours depending on dataset size and hardware."
    echo ""
    python train_multilingual.py --config "$CONFIG_FILE"
fi

# Check if training completed
if [ ! -f "models/best_model.pt" ]; then
    echo -e "${RED}✗ Training failed - model not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Training complete${NC}"

# Step 5: Evaluate Performance
echo -e "\n${GREEN}Step 5: Evaluating Multilingual Performance${NC}"
echo "=========================================="

if [ -f "models/val_predictions.npy" ] && [ -f "models/val_labels.npy" ]; then
    echo "Generating evaluation report..."
    
    # Save languages to JSON for evaluation
    python -c "
import pandas as pd
import json
import numpy as np

# Load validation data to get languages
val_df = pd.read_csv('$DATA_DIR/multilingual_val_split.csv')
languages = val_df['language'].tolist()

with open('models/val_languages.json', 'w') as f:
    json.dump(languages, f)
"
    
    python evaluate_multilingual.py \
        --predictions models/val_predictions.npy \
        --labels models/val_labels.npy \
        --languages models/val_languages.json \
        --output-dir evaluation
    
    echo -e "${GREEN}✓ Evaluation complete${NC}"
else
    echo -e "${YELLOW}Warning: No predictions found for evaluation${NC}"
fi

# Step 6: Test Unified Inference (if Phase 1 model exists)
echo -e "\n${GREEN}Step 6: Testing Unified Inference Pipeline${NC}"
echo "=========================================="

PHASE1_MODEL="../phase1_text_baseline/models/best_model.pt"
PHASE1_CALIBRATION="../phase1_text_baseline/models/calibration/calibration_results.json"

if [ -f "$PHASE1_MODEL" ]; then
    echo "Testing unified inference with Phase 1 + Phase 2..."
    
    python unified_inference.py \
        --phase1-model "$PHASE1_MODEL" \
        --phase2-model "models/best_model.pt" \
        --language-model "models/lid.176.bin" \
        --calibration "$PHASE1_CALIBRATION" \
        --text "Aaj tu bahut pagal hai yaar" || echo -e "${YELLOW}Unified inference test skipped${NC}"
else
    echo -e "${YELLOW}Phase 1 model not found. Run Phase 1 first for full pipeline testing.${NC}"
fi

# Summary
echo -e "\n=========================================="
echo -e "${GREEN}Phase 2 Complete! 🎉${NC}"
echo "=========================================="

echo -e "\n${YELLOW}Generated Files:${NC}"
echo "  📁 models/lid.176.bin - FastText language model"
echo "  📁 models/best_model.pt - XLM-RoBERTa checkpoint"
echo "  📁 models/val_predictions.npy"
echo "  📁 models/per_language_metrics.json"
echo "  📁 evaluation/evaluation_report.md"
echo "  📁 evaluation/per_language_performance.png"

echo -e "\n${YELLOW}Model Capabilities:${NC}"
echo "  ✓ 100+ language support (XLM-RoBERTa)"
echo "  ✓ Code-mix detection (Hinglish, Kanglish, etc.)"
echo "  ✓ Automatic language routing"
echo "  ✓ Unified inference pipeline"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "  1. Review evaluation report:"
echo "     cat evaluation/evaluation_report.md"
echo ""
echo "  2. Test on your own text:"
echo "     python unified_inference.py \\"
echo "       --phase1-model ../phase1_text_baseline/models/best_model.pt \\"
echo "       --phase2-model models/best_model.pt \\"
echo "       --text 'Your text here'"
echo ""
echo "  3. Move to Phase 3: Vision & OCR pipelines"

echo -e "\n${GREEN}Happy moderating across languages! 🌍${NC}"
