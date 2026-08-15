#!/bin/bash
# Phase 3 Setup Script

set -e  # Exit on error

echo "🚀 Phase 3: Vision & OCR Setup"
echo "================================"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: Please activate virtual environment first"
    echo "   Run: source ../.venv/bin/activate"
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt
echo "✅ Python packages installed"

# Check for Tesseract
echo ""
echo "🔍 Checking for Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    tesseract_version=$(tesseract --version 2>&1 | head -n1)
    echo "✅ Tesseract found: $tesseract_version"
else
    echo "⚠️  Tesseract not found"
    echo "   macOS: brew install tesseract"
    echo "   Ubuntu: sudo apt-get install tesseract-ocr"
    echo "   Windows: https://github.com/UB-Mannheim/tesseract/wiki"
fi

# Create necessary directories
echo ""
echo "📁 Creating directory structure..."
mkdir -p data/{nsfw/{train/safe,train/nsfw,test/safe,test/nsfw},hate_symbols,violence}
mkdir -p models
mkdir -p logs
mkdir -p visualizations
mkdir -p test_images
echo "✅ Directories created"

# Download EasyOCR models (will happen on first use)
echo ""
echo "🤖 EasyOCR models will download automatically on first use (~100MB)"

# Check for existing Phase 1/2 models
echo ""
echo "🔗 Checking for Phase 1/2 models..."
if [ -d "../../content_moderation_trained/phase1_text_baseline" ]; then
    echo "✅ Phase 1 models found"
else
    echo "⚠️  Phase 1 models not found at expected location"
fi

if [ -d "../../content_moderation_trained/phase2_multilingual" ]; then
    echo "✅ Phase 2 models found"
else
    echo "⚠️  Phase 2 models not found at expected location"
fi

echo ""
echo "✅ Phase 3 setup complete!"
echo ""
echo "Next steps:"
echo "  1. Review configs/default.yaml"
echo "  2. Download training datasets: python scripts/download_datasets.py"
echo "  3. Test OCR: python scripts/test_ocr.py --image test_images/sample.jpg"
echo "  4. Train models: python scripts/train_image_model.py"
echo ""
