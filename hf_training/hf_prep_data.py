"""
Prepare balanced HF-derived coresets from local HuggingFace datasets:

1. Text (Phases 1 & 2 & 4 & 5): merge 3 local multilingual text toxicity
   datasets into one normalized table (text, label, language).
   - toxicity-multilingual-binary-classification-dataset (8 langs)
   - toxic-vs-clean-dataset (Russian)
   - multilingual_toxicity_dataset (14 langs)
2. Vision Phase 3: extract balanced image coreset from AI-vs-Real parquets.
3. OCR head (Phase 3b + OCR): stream a small subset of ocr_datasets
   (image jpg + text transcript) into a compact training set.

Writes outputs under ROOT/hf_training/_data/
"""
import os
import sys
import json
import glob
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "hf dataset"
AI = ROOT / "_hf_downloads" / "ai_vs_real" / "data"
OUT = ROOT / "hf_training" / "_data"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def load_local_text():
    """Merge all local parquet text datasets into normalized rows."""
    rows = []

    # 1) 8-language binary toxicity (text,label maybe NaN)
    p = HF / "toxicity-multilingual-binary-classification-dataset" / "data"
    for f in sorted(glob.glob(str(p / "*.parquet"))):
        split = Path(f).stem.split("-")[0]
        df = pq.read_table(f).to_pandas()
        df = df.dropna(subset=["label", "text"])
        for _, r in df.iterrows():
            rows.append((str(r["text"]), int(r["label"]), "mixed8", split))

    # 2) Russian toxic-vs-clean (text,label int)
    p = HF / "toxic-vs-clean-dataset" / "data"
    for f in sorted(glob.glob(str(p / "*.parquet"))):
        split = "val" if "validation" in Path(f).stem else Path(f).stem.split("-")[0]
        df = pq.read_table(f).to_pandas()
        df = df.dropna(subset=["label", "text"])
        for _, r in df.iterrows():
            rows.append((str(r["text"]), int(r["label"]), "ru", split))

    # 3) 14-language toxicity (text,toxic)
    p = HF / "multilingual_toxicity_dataset" / "data"
    for f in sorted(glob.glob(str(p / "*.parquet"))):
        lang = Path(f).stem.split("-")[0]
        df = pq.read_table(f).to_pandas()
        df = df.dropna(subset=["toxic", "text"])
        for _, r in df.iterrows():
            rows.append((str(r["text"]), int(r["toxic"]), lang, "train"))

    df = pd.DataFrame(rows, columns=["text", "label", "language", "split"])
    return df


def balanced_sample(df, n, seed=SEED):
    """Return a 50/50 balanced random subset of n rows."""
    safe = df[df["label"] == 0]
    toxic = df[df["label"] == 1]
    half = n // 2
    s = safe.sample(min(half, len(safe)), random_state=seed)
    t = toxic.sample(min(half, len(toxic)), random_state=seed)
    out = pd.concat([s, t], ignore_index=True)
    return out.sample(frac=1, random_state=seed)


def build_text_splits(df):
    # Phase 2/4/5: multilingual balanced (cover as many languages as possible)
    # equal-ish sampling per language to maximize coverage
    per_lang = 600
    picks = []
    for lang, g in df.groupby("language"):
        picks.append(balanced_sample(g, per_lang))
    ml = pd.concat(picks, ignore_index=True)

    tr_all = ml[ml["split"] != "test"].reset_index(drop=True)
    # Phase 1: English-centric subset
    en = df[df["language"].isin(["en"])].reset_index(drop=True)
    if len(en) < 5000:
        en = df[df["language"].isin(["en", "mix"])].reset_index(drop=True)

    # train/val for multilingual binary
    ml_tr = tr_all.sample(6000, random_state=SEED).reset_index(drop=True)
    test_like = ml[ml["split"] == "test"]
    if len(test_like) >= 800:
        ml_val = test_like.sample(800, random_state=SEED).reset_index(drop=True)
    else:
        ml_val = ml.sample(800, random_state=SEED).reset_index(drop=True)

    # English train/val for phase1 baseline
    en_tr = balanced_sample(en, 4000, seed=SEED)
    en_val = balanced_sample(en, 1000, seed=SEED).head(500)

    return en_tr, en_val, ml_tr, ml_val


def build_phase1_multi(en_tr, en_val):
    """Phase 1 uses 6 Jigsaw-style labels; derive them so 'toxic' is primary."""
    harm_words = ["kill", "die", "hate", "stupid", "idiot", "fool", "ugly",
                  "worthless", "scum", "shut", "rac", "sex ", "porn", "ass",
                  "suck", "damn", "bitch", "fuck", "nazi", "slave"]
    severe_words = ["kill you", "i will", "die", "murder", "destroy", "make your life"]
    threat_words = ["will ", "going to ", "threat", "swear"]
    insult_words = ["idiot", "stupid", "fool", "loser", "worthless", "scum"]
    obscene_words = ["fuck", "shit", "ass", "porn", "sex", "bitch", "damn"]
    idh_words = ["rac", "nazi", "slave", "white", "black", "muslim", "jew", "homo", "trump"]

    def multi(row):
        t = str(row["text"]).lower()
        toxic = row["label"]
        cols = {}
        cols["toxic"] = toxic
        cols["severe_toxic"] = int(toxic == 1 and any(w in t for w in severe_words))
        cols["threat"] = int(toxic == 1 and any(w in t for w in threat_words))
        cols["insult"] = int(toxic == 1 and any(w in t for w in insult_words))
        cols["obscene"] = int(toxic == 1 and any(w in t for w in obscene_words))
        cols["identity_hate"] = int(toxic == 1 and any(w in t for w in idh_words))
        return cols

    def build(df, name):
        out = pd.DataFrame({"text": df["text"]})
        lbls = df.apply(multi, axis=1, result_type="expand")
        out = pd.concat([out, lbls], axis=1)
        out.to_csv(OUT / name, index=False)
        print(f"[P1] {name}: {len(out)} rows, toxic rate={(out['toxic'].mean()):.2f}")

    build(en_tr, "phase1_train.csv")
    build(en_val, "phase1_val.csv")


def build_phase2(ml_tr, ml_val):
    ml_tr[["text", "label", "language"]].to_csv(OUT / "phase2_train.csv", index=False)
    ml_val[["text", "label", "language"]].to_csv(OUT / "phase2_val.csv", index=False)
    print(f"[P2] train {len(ml_tr)} rows (langs={ml_tr['language'].nunique()}), "
          f"val {len(ml_val)} rows, toxic={(ml_val['label'].mean()):.2f}")


def build_phase4_pairs(ml_tr, img_root, n=3000):
    """Paired text + image rows using AI-vs-Real images for fusion training."""
    rows = []
    img_files_ai = sorted((img_root / "ai_vs_real" / "train" / "ai").glob("*.jpg"))
    img_files_real = sorted((img_root / "ai_vs_real" / "train" / "real").glob("*.jpg"))
    toxic_rows = ml_tr[ml_tr["label"] == 1].reset_index(drop=True)
    safe_rows = ml_tr[ml_tr["label"] == 0].reset_index(drop=True)
    for i in range(n):
        toxic = i % 2 == 0
        src = toxic_rows if toxic else safe_rows
        row = src.iloc[i % len(src)]
        pool = img_files_ai if toxic else img_files_real
        ip = str(np.random.choice(pool)) if pool else ""
        rows.append({"text": row["text"], "toxic": int(toxic),
                     "image_path": ip, "language": row["language"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phase4_train.csv", index=False)
    print(f"[P4] {len(df)} pairs, toxic={(df['toxic'].mean()):.2f}")


def build_phase5_samples(ml_val, n=2000):
    s = ml_val.sample(min(n, len(ml_val)), random_state=SEED)
    df = pd.DataFrame({"text": s["text"], "true_toxic": s["label"].values,
                       "score": np.random.uniform(0, 1, len(s))})
    df.to_csv(OUT / "phase5_scoring.csv", index=False)
    print(f"[P5] {len(df)} scoring samples, toxic={(df['true_toxic'].mean()):.2f}")


def extract_ai_vs_real(img_root, train_n=4000, val_n=500):
    """Extract balanced AI-vs-Real image coreset from parquet byte blobs."""
    import io
    from PIL import Image
    files = sorted(glob.glob(str(AI / "*.parquet")))
    samples = []
    for f in files:
        t = pq.read_table(f, columns=["image", "binary_label"])
        df = t.to_pandas()
        for _, r in df.iterrows():
            samples.append((r["image"]["bytes"], int(r["binary_label"])))
    # 1 -> ai, 0 -> real (source repo convention: binary_label 1 = AI)
    random.shuffle(samples)
    ai = [s for s in samples if s[1] == 1]
    real = [s for s in samples if s[1] == 0]
    print(f"[P3] AI-vs-Real available: ai={len(ai)} real={len(real)}")

    def write(cls_tag, items, split, count):
        d = img_root / "ai_vs_real" / split / cls_tag
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        k = 0
        for idx, (bts, _) in enumerate(items[:count]):
            try:
                im = Image.open(io.BytesIO(bts)).convert("RGB")
                im = im.resize((224, 224))
                im.save(d / f"{idx}.jpg", quality=88)
                k += 1
            except Exception:
                pass
        print(f"[P3] {cls_tag}/{split}: wrote {k} images")
        return k

    # train split
    write("ai", ai, "train", train_n // 2)
    write("real", real, "train", train_n // 2)
    # val split
    write("ai", ai[train_n // 2:], "val", val_n // 2)
    write("real", real[train_n // 2:], "val", val_n // 2)


def build_ocr(img_root, n=2500):
    """Stream a subset of ocr_datasets into OCR training set."""
    import io
    from PIL import Image
    from datasets import load_dataset
    d = img_root / "ocr"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("hsbharadwaj/ocr_datasets", split="train", streaming=True)
    k = 0
    for i, ex in enumerate(ds):
        if k >= n:
            break
        try:
            im = Image.open(io.BytesIO(ex["jpg"]["bytes"])).convert("RGB")
            im = im.resize((224, 224))
            im.save(d / f"{k}.jpg", quality=85)
            with open(d / f"{k}.txt", "w", encoding="utf-8") as f:
                f.write(str(ex["txt"]))
            k += 1
        except Exception:
            continue
    print(f"[OCR] wrote {k} image-text pairs")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img_root = OUT / "images"

    print("== Loading local HF text datasets ==")
    df = load_local_text()
    print(f"   merged {len(df)} text rows across {df['language'].nunique()} languages")

    en_tr, en_val, ml_tr, ml_val = build_text_splits(df)
    build_phase1_multi(en_tr, en_val)
    build_phase2(ml_tr, ml_val)
    build_phase5_samples(ml_val)

    print("== Extracting AI-vs-Real images ==")
    extract_ai_vs_real(img_root)

    build_phase4_pairs(ml_tr, img_root)

    print("== Streaming OCR subset ==")
    build_ocr(img_root)

    with open(OUT / "prep_manifest.json", "w") as f:
        json.dump({
            "text_rows_total": int(len(df)),
            "languages": sorted(df["language"].unique().tolist()),
            "seed": SEED,
        }, f, indent=2)
    print("DATA PREP DONE")


if __name__ == "__main__":
    main()
