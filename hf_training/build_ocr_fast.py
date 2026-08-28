"""Build OCR image+text pairs by range-downloading part of an ocr webdataset tar
and parsing the .jpg/.txt entries directly (fast, no datasets streaming overhead)."""
import io
import os
import shutil
import tarfile
import urllib.request
import time
from pathlib import Path
from PIL import Image

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "hf_training" / "_data" / "images" / "ocr"
out.mkdir(parents=True, exist_ok=True)

TAR_URL = "https://huggingface.co/datasets/hsbharadwaj/ocr_datasets/resolve/main/webdataset/train/ocr-shard-000000.tar"
TMP = ROOT / "_hf_downloads" / "ocr" / "ocr_part.tar"
N = 1200

t0 = time.time()
# Download a large leading range of the shard once
req = urllib.request.Request(TAR_URL, headers={"Range": "bytes=0-200000000"})  # first 200MB
data = b""
with urllib.request.urlopen(req, timeout=120) as r:
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            break
        data += chunk
print(f"[OCR build] downloaded {len(data)/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)

# Parse entries until we have N pairs
t0 = time.time()
k = 0
stream = io.BytesIO(data)
tf = tarfile.open(fileobj=stream, mode="r|")
cur_img_name = None
cur_img_bytes = None
for member in tf:
    fname = member.name
    if fname.endswith(".jpg"):
        cur_img_name = fname
        cur_img_bytes = tf.extractfile(member).read()
    elif fname.endswith(".txt") and cur_img_bytes is not None:
        txt = tf.extractfile(member).read().decode("utf-8", "ignore").strip()
        if txt:
            im = Image.open(io.BytesIO(cur_img_bytes)).convert("RGB")
            im = im.resize((224, 224))
            im.save(out / f"{k}.jpg", quality=85)
            (out / f"{k}.txt").write_text(txt, encoding="utf-8")
            k += 1
            if k % 200 == 0:
                print(f"[OCR build] {k}/{N}", flush=True)
            if k >= N:
                break
        cur_img_name = None
        cur_img_bytes = None
print(f"[OCR build] DONE wrote {k} pairs in {time.time()-t0:.0f}s")
