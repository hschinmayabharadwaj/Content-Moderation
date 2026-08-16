# Google Colab Training Guide

## Complete Notebook for Training All Phases

**File**: `COLAB_COMPLETE_TRAINING.ipynb`

This notebook trains all 3 phases of the content moderation system in Google Colab with GPU acceleration.

---

## ⚡ Quick Start (3 Steps)

### 1. Open in Colab
- Go to https://colab.research.google.com/
- Click "File" → "Open notebook"
- Select "GitHub" tab
- Enter: `hschinmayabharadwaj/Content-Moderation`
- Choose `COLAB_COMPLETE_TRAINING.ipynb`

### 2. Enable GPU
- Click "Runtime" in menu
- Select "Change runtime type"
- Choose "GPU" (T4 or better)
- Click "Save"

### 3. Run All Cells
- Press `Ctrl+F9` (or click "Runtime" → "Run all")
- Wait for completion (2-3 hours)
- Models automatically backup to Google Drive

---

## 📊 What Gets Trained

### Phase 1: Text Baseline
- **Model**: DistilBERT
- **Categories**: 6 (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- **Training time**: ~20 minutes
- **Output**: `phase1_text_baseline/models/best_model.pt`

### Phase 2: Multilingual
- **Model**: XLM-RoBERTa
- **Languages**: English + Multilingual support
- **Training time**: ~20 minutes
- **Output**: `phase2_multilingual/models/best_model.pt`

### Phase 3: Vision & OCR
- **NSFW Detector**: ResNet50 (~20 min)
- **Hate Symbols**: ResNet50 (~20 min)
- **Violence**: EfficientNet (~15 min)
- **Output**: `phase3_vision_ocr/models/{nsfw,hate_symbols,violence}/best_model.pt`

**Total time**: 2-3 hours on GPU (vs 3-6 hours on CPU)

---

## 🔄 Notebook Sections

1. **Setup & GPU Check** (1 min)
   - Verifies GPU is available
   - Shows GPU specs

2. **Clone Repository** (2 min)
   - Clones from GitHub
   - Mounts Google Drive for backup

3. **Install Dependencies** (5 min)
   - Core ML libraries
   - Phase 3 specific packages
   - EasyOCR models

4. **Phase 1 Training** (25 min)
   - Downloads dataset (Jigsaw Toxic Comments)
   - Creates sample data
   - Trains text classifier
   - Calibrates thresholds

5. **Phase 2 Training** (25 min)
   - Prepares multilingual data
   - Trains XLM-RoBERTa
   - Evaluates on test set

6. **Phase 3 Training** (60 min)
   - Creates dummy datasets
   - Trains NSFW detector
   - Trains hate symbols detector
   - Trains violence detector

7. **Evaluation** (10 min)
   - Tests OCR system
   - Runs full multimodal evaluation
   - Exports results

8. **Backup** (5 min)
   - Copies all models to Google Drive
   - Saves training summary

---

## 📦 Model Outputs

After training, you'll have in Google Drive:

```
ContentModeration_TrainedModels/
├── phase1_models/
│   ├── best_model.pt
│   ├── training_history.json
│   └── training_config.json
│
├── phase2_models/
│   ├── best_model.pt
│   ├── training_history.json
│   └── training_config.json
│
├── phase3_models/
│   ├── nsfw/best_model.pt
│   ├── hate_symbols/best_model.pt
│   ├── violence/best_model.pt
│   └── (config files for each)
│
└── TRAINING_SUMMARY.json
```

Each model includes:
- **best_model.pt** - Production model
- **training_history.json** - Loss/accuracy curves
- **training_config.json** - Exact configuration used

---

## 🚀 After Training

### 1. Download Models
- Open Google Drive
- Navigate to `ContentModeration_TrainedModels/`
- Download all model folders
- Place in your local `content-moderation-system/` directory

### 2. Use for Inference
```bash
# Test text moderation
python text_moderator.py --text "Your text here"

# Test multimodal moderation
cd phase3_vision_ocr
python src/multimodal_moderator.py \
  --image test.jpg \
  --image-models models
```

### 3. Deploy
- Copy model files to your production server
- Use in your application

---

## ⚠️ Common Issues

### GPU Not Available
**Problem**: Notebook shows "GPU Available: False"

**Solution**:
1. Go to Runtime → Change runtime type
2. Select "GPU"
3. Click "Save"
4. Re-run the GPU check cell

### Out of Memory During Phase 3
**Problem**: "RuntimeError: CUDA out of memory"

**Solution**: The notebook uses batch_size=32. For smaller GPUs:
- Edit `train_image_model.py` lines to use `--batch-size 16`
- Or skip Phase 3 (you have Phase 1 & 2)

### Very Slow Training
**Problem**: Still getting slow training times

**Solution**: Check you're on GPU (not CPU)
- Click Runtime → Current resource
- Should show "GPU" and model type (T4, P100, V100, etc.)

### Models Not Backing Up
**Problem**: Files not appearing in Google Drive

**Solution**:
1. Check permissions - grant Colab access to Drive
2. Manually copy from `/content/drive/My Drive/ContentModeration_TrainedModels/`

---

## 💡 Tips

1. **Keep Colab tab open** - Colab can timeout if left unattended
2. **Monitor progress** - Watch loss/accuracy improve in real-time
3. **Check GPU stats** - Click "Runtime" → "Resources" to see GPU usage
4. **Save checkpoints** - Models auto-save to Drive every phase
5. **Interrupt training** - Press Stop button if needed (models still saved)

---

## 🔍 Monitoring Training

During training, you'll see output like:

```
Epoch 1/20
Train - Loss: 0.4923 | Acc: 82.50%
Val   - Loss: 0.3211 | Acc: 87.50%
LR: 0.000100
```

**What this means**:
- ✅ Loss decreasing = good
- ✅ Accuracy increasing = good
- ✅ Val acc close to train acc = not overfitting
- ✅ LR decreasing = learning rate schedule working

---

## 📚 Related Guides

- `TRAINING_GUIDE.md` - Detailed training documentation
- `QUICK_TRAINING_REFERENCE.md` - Quick reference
- `COMMAND_REFERENCE.md` - All available commands
- `README.md` - System overview

---

## 🎓 Full Documentation

- **Phase 1**: `phase1_text_baseline/docs/getting_started_phase1.md`
- **Phase 2**: `phase2_multilingual/docs/getting_started_phase2.md`
- **Phase 3**: `phase3_vision_ocr/README.md`
- **Datasets**: `phase3_vision_ocr/docs/DATASETS.md`

---

## ⏱️ Timeline

| Phase | Time (GPU) | What Gets Trained |
|-------|-----------|------------------|
| Setup | 10 min | Dependencies, clone repo |
| Phase 1 | 25 min | DistilBERT text classifier |
| Phase 2 | 25 min | XLM-RoBERTa multilingual |
| Phase 3 | 60 min | 3 image classifiers |
| Export | 15 min | Backup, summary, evaluation |
| **Total** | **2-3 hours** | **All 3 phases + OCR** |

---

## 🎯 Success Criteria

After completion, you should have:
- ✅ 3 trained model directories (phase1, phase2, phase3)
- ✅ Models backed up to Google Drive
- ✅ Training summary showing all metrics
- ✅ Evaluation results showing models work
- ✅ OCR tested and working
- ✅ Multimodal system functional

---

**Ready to train? Open the notebook in Colab and run all cells!** 🚀
