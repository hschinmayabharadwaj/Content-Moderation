"""Write a manifest + summary into the single trained_hugging_face_models folder."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "trained_hugging_face_models"

def read_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

manifest = {
    "name": "trained_hugging_face_models",
    "description": "All 5-phase content-moderation parameters trained from HuggingFace datasets, stored in a single folder.",
    "datasets_used": {
        "text": [
            "hsbharadwaj/toxicity-multilingual-binary-classification-dataset (8 langs)",
            "hsbharadwaj/toxic-vs-clean-dataset (Russian)",
            "hsbharadwaj/multilingual_toxicity_dataset (14 langs)",
        ],
        "vision": ["hsbharadwaj/AI-vs-Real (AI vs real images)"],
        "ocr": ["hsbharadwaj/ocr_datasets (webdataset, OCR images + transcripts)"],
    },
    "phases": {
        "phase1": read_json(OUT / "phase1" / "config.json"),
        "phase2": read_json(OUT / "phase2" / "config.json"),
        "phase3": read_json(OUT / "phase3" / "config.json"),
        "phase3_ocr": read_json(OUT / "phase3_ocr" / "config.json"),
        "phase4": read_json(OUT / "phase4" / "config.json"),
        "phase5": read_json(OUT / "phase5" / "hitl_config.json"),
    },
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("manifest written to", OUT / "manifest.json")
