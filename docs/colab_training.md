# Google Colab Training Guide

Train Phase 1 and Phase 2 on a free GPU — no load on your Mac.

## Quick start

1. Open [Google Colab](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU**
3. Upload `notebooks/colab_train_phase1_phase2.ipynb`  
   **or** open from GitHub after pushing this file
4. **Runtime → Run all**
5. Download `content_moderation_trained.zip` from the last cell

## Profiles

| Profile | Phase 1 data | Epochs | Est. time (T4) |
|---|---|---|---|
| `quick` | 2,000 samples | 1 | ~25–35 min |
| `standard` | 5,000 samples | 2 | ~45–70 min |
| `full` | Full Jigsaw (~159K) | 3 | ~2–3 hrs |

Change `PROFILE` in cell 1 before running.

## What gets trained

- **Phase 1:** DistilBERT multi-label classifier + temperature calibration + tier thresholds
- **Phase 2:** FastText language ID + XLM-RoBERTa on English + synthetic code-mix data

## Restore on your Mac

```bash
cd content-moderation-system
unzip ~/Downloads/content_moderation_trained.zip -d /tmp/artifacts
mkdir -p phase1_text_baseline/models phase2_multilingual/models
cp -r /tmp/artifacts/phase1_text_baseline/models/* phase1_text_baseline/models/
cp -r /tmp/artifacts/phase2_multilingual/models/* phase2_multilingual/models/
cp -r /tmp/artifacts/phase2_multilingual/evaluation phase2_multilingual/ 2>/dev/null || true

cd phase2_multilingual
source ../.venv-py311/bin/activate  # or your venv
python unified_inference.py \
  --phase1-model ../phase1_text_baseline/models/best_model.pt \
  --phase2-model models/best_model.pt \
  --language-model models/lid.176.bin \
  --calibration ../phase1_text_baseline/models/calibration/calibration_results.json \
  --text "Aaj tu bahut pagal hai yaar"
```

Inference on Mac uses CPU and stays cool.

## Files

- `notebooks/colab_train_phase1_phase2.ipynb` — one-click Colab pipeline
- `configs/colab/baseline_colab.yaml` — Phase 1 reference config
- `configs/colab/xlm_roberta_colab.yaml` — Phase 2 reference config
- `requirements-colab.txt` — minimal pip deps (notebook installs inline)
