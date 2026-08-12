#!/bin/bash

# Phase 1: End-to-End Pipeline
# This script runs the complete Phase 1 workflow:
# 1. Downloads dataset
# 2. Trains classifier
# 3. Calibrates thresholds

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Content Moderation System - Phase 1"
echo "Baseline Text Classification"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
USE_SAMPLE=${USE_SAMPLE:-false}
CONFIG_FILE=${CONFIG_FILE:-"configs/baseline.yaml"}
DATA_DIR=${DATA_DIR:-"data"}

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Config file: $CONFIG_FILE"
echo "  Data directory: $DATA_DIR"
echo "  Use sample: $USE_SAMPLE"

# Resolve Python interpreter (Windows venvs expose "python", not "python3")
if [[ -n "$PYTHON" && -x "$PYTHON" ]]; then
    :
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo -e "\n${RED}ERROR: Python not found${NC}"
    echo "Activate the project venv or set PYTHON to your interpreter path."
    exit 1
fi

# Prefer the repository venv when the script is launched from a shell without
# an activated environment. This keeps the phase launcher on the interpreter
# that already has the project dependencies installed.
if [[ -z "$VIRTUAL_ENV" ]]; then
    for candidate in "$PROJECT_ROOT/.venv-py311" "$PROJECT_ROOT/.venv" "$PROJECT_ROOT/venv"; do
        if [[ -f "$candidate/bin/activate" ]]; then
            # shellcheck disable=SC1090
            source "$candidate/bin/activate"
            break
        elif [[ -f "$candidate/Scripts/activate" ]]; then
            # shellcheck disable=SC1090
            source "$candidate/Scripts/activate"
            break
        fi
    done
fi

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "\n${RED}WARNING: Virtual environment not activated${NC}"
    echo "It's recommended to use a virtual environment."
    echo "Create one with: python -m venv venv"
    echo "Activate with: source venv/Scripts/activate  (Windows) or source venv/bin/activate  (Linux/macOS)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Download Dataset
echo -e "\n${GREEN}Step 1: Downloading Dataset${NC}"
echo "=========================================="

if [ -f "$DATA_DIR/train.csv" ]; then
    echo "Dataset already exists at $DATA_DIR/train.csv"
    read -p "Re-download? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        "$PYTHON" download_data.py --output-dir "$DATA_DIR" --analyze
    fi
else
    "$PYTHON" download_data.py --output-dir "$DATA_DIR" --analyze
fi

# Create sample dataset if requested
if [ "$USE_SAMPLE" = true ]; then
    echo -e "\n${YELLOW}Creating sample dataset...${NC}"
    "$PYTHON" download_data.py --output-dir "$DATA_DIR" --create-sample --sample-size 5000
    
    # Update config to use sample
    if command -v yq &> /dev/null; then
        yq eval '.training.use_sample = true' -i "$CONFIG_FILE"
    else
        echo -e "${YELLOW}Note: Install 'yq' to automatically update config, or manually set training.use_sample: true${NC}"
    fi
fi

echo -e "${GREEN}✓ Dataset ready${NC}"

# Step 2: Train Classifier
echo -e "\n${GREEN}Step 2: Training Classifier (Step 1.1)${NC}"
echo "=========================================="

if [ -f "models/best_model.pt" ]; then
    echo "Model already exists at models/best_model.pt"
    read -p "Re-train? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping training..."
    else
        "$PYTHON" train_classifier.py --config "$CONFIG_FILE"
    fi
else
    "$PYTHON" train_classifier.py --config "$CONFIG_FILE"
fi

# Check if training completed successfully
if [ ! -f "models/best_model.pt" ]; then
    echo -e "${RED}✗ Training failed - model not found${NC}"
    exit 1
fi

if [ ! -f "models/val_predictions.npy" ]; then
    echo -e "${RED}✗ Training failed - predictions not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Training complete${NC}"

# Step 3: Calibrate Thresholds
echo -e "\n${GREEN}Step 3: Calibrating Thresholds (Step 1.2)${NC}"
echo "=========================================="

"$PYTHON" calibrate_thresholds.py --config "$CONFIG_FILE"

# Check if calibration completed successfully
if [ ! -f "models/calibration/calibration_results.json" ]; then
    echo -e "${RED}✗ Calibration failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Calibration complete${NC}"

# Summary
echo -e "\n=========================================="
echo -e "${GREEN}Phase 1 Complete! 🎉${NC}"
echo "=========================================="

echo -e "\n${YELLOW}Generated Files:${NC}"
echo "  📁 models/best_model.pt"
echo "  📁 models/val_predictions.npy"
echo "  📁 models/val_labels.npy"
echo "  📁 models/calibration/calibration_results.json"
echo "  📁 models/calibration/*_reliability.png"
echo "  📁 models/calibration/*_thresholds.png"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "  1. Review calibration results:"
echo "     cat models/calibration/calibration_results.json | python -m json.tool"
echo ""
echo "  2. View visualizations:"
echo "     open models/calibration/*.png"
echo ""
echo "  3. Try the interactive notebook:"
echo "     jupyter notebook notebooks/phase1_tutorial.ipynb"
echo ""
echo "  4. Move to Phase 2: Multilingual routing"

echo -e "\n${GREEN}Happy moderating! 🛡️${NC}"
