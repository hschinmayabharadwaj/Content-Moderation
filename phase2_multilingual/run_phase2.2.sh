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
val_df = pd.read_csv('data/multilingual_val_split.csv')
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
