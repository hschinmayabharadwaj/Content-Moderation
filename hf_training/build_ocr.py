"""Standalone build of OCR image+text pairs from HF ocr_datasets (streamed)."""
import os
import io
import shutil
from pathlib import Path
from PIL import Image
from datasets import load_dataset

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

ROOT = Path(__file__).resolve().parent.parent
d = ROOT / "hf_training" / "_data" / "images" / "ocr"
shutil.rmtree(d, ignore_errors=True)
d.mkdir(parents=True, exist_ok=True)

N = 1500
ds = load_dataset("hsbharadwaj/ocr_datasets", split="train", streaming=True)
k = 0
for i, ex in enumerate(ds):
    if k >= N:
        break
    try:
        im = Image.open(io.BytesIO(ex["jpg"]["bytes"])).convert("RGB")
        im = im.resize((224, 224))
        im.save(d / f"{k}.jpg", quality=85)
        with open(d / f"{k}.txt", "w", encoding="utf-8") as f:
            f.write(str(ex["txt"]))
        k += 1
        if k % 200 == 0:
            print(f"[OCR build] {k}/{N}", flush=True)
    except Exception:
        continue
print(f"[OCR build] DONE wrote {k} pairs")
