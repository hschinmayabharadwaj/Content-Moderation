"""
Generate a balanced 50% clean / 50% toxic dataset for all 5 phases.

Phase 1: Multi-label text (Jigsaw style) - balanced per toxic label
Phase 2: Multilingual text - balanced 50/50
Phase 3: Images (nsfw / hate_symbols / violence) - balanced 50/50 synthetic
Phase 4: Multimodal pairs (text + image) for fusion training
Phase 5: Scoring samples (for HITL thresholds + drift baseline)

Writes outputs under ROOT/content_moderation_dataset/
"""

import os
import sys
import random
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_SRC = ROOT / "phase1_text_baseline" / "data"
ML_SRC = ROOT / "phase2_multilingual" / "data"
OUT = ROOT / "content_moderation_dataset"

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def ensure_dirs():
    for split in ("train", "val", "test"):
        for dset in ("nsfw", "hate_symbols", "violence"):
            for cls_ in ("safe", "toxic"):
                (OUT / "images" / dset / split / cls_).mkdir(
                    parents=True, exist_ok=True
                )


def build_phase1_balanced(num_train=4000, num_val=500, num_test=500):
    """Balanced multi-label text dataset from real Jigsaw data."""
    full_path = DATA_SRC / "train.csv"
    sample_path = DATA_SRC / "train_sample.csv"

    if full_path.exists():
        df = pd.read_csv(full_path)
    else:
        df = pd.read_csv(sample_path)

    df = df.dropna(subset=["comment_text"])
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # Use 'toxic' as main binary split for 50/50 balance
    toxic = df[df["toxic"] == 1].copy()
    clean = df[df["toxic"] == 0].copy()

    def sample_balanced(target):
        n = target // 2
        t = toxic.sample(min(n, len(toxic)), random_state=SEED)
        c = clean.sample(min(n, len(clean)), random_state=SEED)
        out = pd.concat([t, c], ignore_index=True)
        return out.sample(frac=1, random_state=SEED)

    train = sample_balanced(num_train)
    val = sample_balanced(num_val)
    test = sample_balanced(num_test)

    def write(df, name, split):
        rec = {"text": [], "toxic": [], "severe_toxic": [], "obscene": [],
               "threat": [], "insult": [], "identity_hate": []}
        for _, row in df.iterrows():
            rec["text"].append(str(row["comment_text"]))
            for c in label_cols:
                rec[c].append(int(row[c]))
        out_df = pd.DataFrame(rec)
        out_df.to_csv(OUT / name, index=False)
        toxicity = out_df["toxic"].sum() / len(out_df)
        print(f"[P1] {split}: {len(out_df)} rows, toxic rate={toxicity:.2f}")

    write(train, "phase1_train.csv", "train")
    write(val, "phase1_val.csv", "val")
    write(test, "phase1_test.csv", "test")


def build_phase2_balanced(num_train=4000, num_val=500, num_test=500):
    """Balanced binary multilingual dataset (50/50)."""
    src = ML_SRC / "multilingual_train_split.csv"
    if not src.exists():
        src = ML_SRC / "multilingual_train.csv"
    df = pd.read_csv(src)
    df = df.dropna(subset=["text"])
    if "language" not in df.columns:
        df["language"] = "english"
        df["is_code_mixed"] = 0

    safe = df[df["label"] == 0].copy()
    toxic = df[df["label"] == 1].copy()

    def sample_balanced(n):
        half = n // 2
        s = safe.sample(min(half, len(safe)), random_state=SEED)
        t = toxic.sample(min(half, len(toxic)), random_state=SEED)
        out = pd.concat([s, t], ignore_index=True)
        return out.sample(frac=1, random_state=SEED)

    for name, n, split in (("phase2_train.csv", num_train, "train"),
                           ("phase2_val.csv", num_val, "val"),
                           ("phase2_test.csv", num_test, "test")):
        d = sample_balanced(n)
        d.to_csv(OUT / name, index=False)
        print(f"[P2] {split}: {len(d)} rows, toxic rate={d['label'].mean():.2f}")


def make_image(label, toxic, idx, path):
    """Create a synthetic image with a distinct visual fingerprint per class."""
    from PIL import Image, ImageDraw
    size = 224
    img = Image.new("RGB", (size, size))
    px = img.load()

    for y in range(size):
        for x in range(size):
            if toxic:
                # Red-dominant gradient grid (flagged content fingerprint)
                r = int(120 + 80 * math.sin(x / 14.0) * math.cos(y / 18.0))
                g = int(30 + 40 * math.sin(x / 9.0 + 2))
                b = int(30 + 40 * math.cos(y / 11.0 + 1))
                px[x, y] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
            else:
                # Blue/green calm gradient (clean content fingerprint)
                r = int(40 + 40 * math.cos(x / 20.0))
                g = int(120 + 60 * math.sin(y / 16.0))
                b = int(160 + 60 * math.cos((x + y) / 24.0))
                px[x, y] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    # Add central shape marker
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    if toxic:
        draw.text((cx - 40, cy - 10), "FLAG", fill=(255, 255, 255))
    else:
        draw.text((cx - 40, cy - 10), "OK", fill=(0, 0, 0))

    img.save(path)


def build_phase3():
    """Balanced synthetic image datasets (50/50 per dataset)."""
    per_split = {"train": 300, "val": 80, "test": 80}
    for dset in ("nsfw", "hate_symbols", "violence"):
        for split, n in per_split.items():
            for cls_ in ("safe", "toxic"):
                # Only produce the needed count per class
                pass
    for dset in ("nsfw", "hate_symbols", "violence"):
        counts = {}
        for split, n in per_split.items():
            for cls_ in ("safe", "toxic"):
                count = n // 2
                counts[(dset, split, cls_)] = count
                for i in range(count):
                    path = OUT / "images" / dset / split / cls_ / f"{i}.png"
                    if not path.exists():
                        make_image(dset, cls_ == "toxic", i, path)
        total = sum(c for (_, _, cls_), c in counts.items() if cls_ == "toxic") * 2
        print(f"[P3] {dset}: generated, safe/toxic balanced")


def build_phase4_pairs(n=4000):
    """Build text+image pairs for fusion training (50/50)."""
    p1 = pd.read_csv(OUT / "phase1_train.csv")
    p2 = pd.read_csv(OUT / "phase2_train.csv")

    rows = []
    for i in range(n):
        p2_row = p2.iloc[i % len(p2)]
        toxic = int(p2_row["label"])
        text = p2_row["text"]
        img_dset = random.choice(["nsfw", "hate_symbols", "violence"])
        split = "train"
        cls_ = "toxic" if toxic else "safe"
        img_files = list((OUT / "images" / img_dset / split / cls_).glob("*.png"))
        img_path = str(random.choice(img_files)) if img_files else ""
        rows.append({"text": text, "toxic": toxic, "image_dset": img_dset,
                     "image_path": img_path})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phase4_train.csv", index=False)
    print(f"[P4] fusion pairs: {len(df)} rows, toxic rate={df['toxic'].mean():.2f}")


def build_phase5_samples(n=2000):
    """Scoring samples for HITL thresholds + drift baseline."""
    p2 = pd.read_csv(OUT / "phase2_val.csv")
    s = p2.sample(min(n, len(p2)), random_state=SEED)
    df = pd.DataFrame({
        "text": s["text"],
        "true_toxic": s["label"].values,
        "score": np.random.uniform(0, 1, len(s)),
    })
    df.to_csv(OUT / "phase5_scoring.csv", index=False)
    print(f"[P5] scoring samples: {len(df)} rows, toxic rate={df['true_toxic'].mean():.2f}")


def write_manifest():
    manifest = {
        "name": "Content Moderation Balanced Dataset (50% clean / 50% toxic)",
        "balance": "each phase is seeded with equal safe and toxic samples",
        "seed": SEED,
        "phases": {
            "1": "text baseline multi-label (roberta)",
            "2": "multilingual binary (xlm-roberta)",
            "3": "image classification (resnet18) - synthetic",
            "4": "multimodal fusion pairs",
            "5": "HITL scoring + drift baseline samples",
        },
    }
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("[manifest] written")


if __name__ == "__main__":
    ensure_dirs()
    build_phase1_balanced()
    build_phase2_balanced()
    build_phase3()
    build_phase4_pairs()
    build_phase5_samples()
    write_manifest()
    print("\nDONE generating balanced dataset")
