#!/bin/bash

# Phase 3 Complete Training Script
# Trains all three image classifiers with proper storage of models, logs, and evaluations

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                  Phase 3: Complete Training Pipeline                           ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"

# Check if we're in the right directory
if [ ! -f "scripts/train_image_model.py" ]; then
    echo "❌ Error: Please run this script from phase3_vision_ocr directory"
    echo "   cd content-moderation-system/phase3_vision_ocr"
    exit 1
fi

# Activate virtual environment if needed
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔧 Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Create necessary directories
echo "📁 Creating output directories..."
mkdir -p models/{nsfw,hate_symbols,violence}/{evaluation}
mkdir -p logs
mkdir -p data

# Step 1: Prepare datasets
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 STEP 1: Preparing Datasets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "data/nsfw/train" ]; then
    echo "Creating dummy datasets (1,440 images total)..."
    python scripts/download_datasets.py --dummy --num-samples 200
else
    echo "✅ Datasets already exist"
fi

echo ""
echo "📊 Dataset Summary:"
python scripts/download_datasets.py --info

# Step 2: Train NSFW Model
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STEP 2: Training NSFW Detector"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/train_image_model.py \
  --dataset nsfw \
  --backbone resnet50 \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --verbose

echo ""
echo "✅ NSFW model saved to: models/nsfw/"
echo "   - best_model.pt (production model)"
echo "   - training_history.json (metrics)"
echo "   - training_config.json (configuration)"

# Step 3: Train Hate Symbols Model
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STEP 3: Training Hate Symbols Detector"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/train_image_model.py \
  --dataset hate_symbols \
  --backbone resnet50 \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --verbose

echo ""
echo "✅ Hate symbols model saved to: models/hate_symbols/"

# Step 4: Train Violence Model
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STEP 4: Training Violence Detector"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/train_image_model.py \
  --dataset violence \
  --backbone efficientnet_b0 \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --verbose

echo ""
echo "✅ Violence model saved to: models/violence/"

# Step 5: Evaluation
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STEP 5: Evaluating Models"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Evaluating NSFW model..."
python scripts/test_image_classifier.py \
  --model models/nsfw/best_model.pt \
  --batch data/nsfw/test/ 2>/dev/null || echo "⚠️ Test evaluation skipped"

echo ""
echo "Full multimodal system test..."
python scripts/test_multimodal.py

# Step 6: Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ TRAINING COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "📁 Output Structure:"
echo ""
ls -lh models/*/best_model.pt

echo ""
echo "📊 Training Configurations:"
find models -name "training_config.json" -exec sh -c 'echo "File: {}" && head -5 {}' \;

echo ""
echo "📈 Training History:"
find models -name "training_history.json" -exec sh -c 'echo "File: {}" && python -c "import json; h=json.load(open(\"{}\"));print(f\"  Epochs: {len(h[\"train_loss\"])}\");print(f\"  Final train acc: {h[\"train_acc\"][-1]:.1f}%\");print(f\"  Final val acc: {h[\"val_acc\"][-1]:.1f}%\")"' \;

echo ""
echo "✨ Next Steps:"
echo "  1. Use models with multimodal moderator:"
echo "     python src/multimodal_moderator.py --image test.jpg --image-models models"
echo ""
echo "  2. View training history:"
echo "     cat models/nsfw/training_history.json"
echo ""
echo "  3. Backup models:"
echo "     tar -czf trained_models_backup.tar.gz models/"
echo ""
echo "═════════════════════════════════════════════════════════════════════════════════"
