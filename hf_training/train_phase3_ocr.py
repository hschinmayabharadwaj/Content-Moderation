"""
Phase 3b (OCR head): Train a compact image->text OCR model on a subset of
HF ocr_datasets (HuggingFace webdataset). resnet18 visual encoder + GRU decoder.

Exports to trained_hugging_face_models/phase3_ocr
"""
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "trained_hugging_face_models" / "phase3_ocr"
DATA = ROOT / "hf_training" / "_data" / "images" / "ocr"
IMG_SIZE = 224
MAX_LEN = 40
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CHARS = " abcdefghijklmnopqrstuvwxyz0123456789-.,:;!?'\"%()/+&@"
C2I = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = pad


def encode(s):
    s = str(s).lower()
    ids = [C2I.get(c, 0) for c in s]
    ids = ids[:MAX_LEN]
    ids = ids + [0] * (MAX_LEN - len(ids))
    return ids


class OCRDataset(Dataset):
    def __init__(self, transform):
        self.rows = []
        for p in sorted(DATA.glob("*.jpg")):
            t = p.with_suffix(".txt")
            self.rows.append((str(p), t.read_text(encoding="utf-8") if t.exists() else ""))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, text = self.rows[idx]
        img = self.transform(Image.open(path).convert("RGB"))
        return img, torch.tensor(encode(text), dtype=torch.long)


class OCRModel(nn.Module):
    def __init__(self, vocab_size=len(CHARS) + 1, hidden=256, embed=128):
        super().__init__()
        enc = torchvision.models.resnet18()
        enc.fc = nn.Identity()
        self.enc = enc
        self.proj = nn.Linear(512, hidden)
        self.embed = nn.Embedding(vocab_size, embed)
        self.gru = nn.GRU(input_size=embed + hidden, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, img, targets=None, teacher_forcing=True):
        x = self.enc(img)
        feat = torch.tanh(self.proj(x)).unsqueeze(1)  # [B,1,H]
        B = img.size(0)
        if targets is not None and teacher_forcing:
            seq = targets  # [B,L]
            emb = self.embed(seq)  # [B,L,E]
            emb = torch.cat([feat.expand(B, seq.size(1), -1), emb], dim=2)
            out, _ = self.gru(emb)
            return self.fc(out)
        # inference: greedy decode
        inp = torch.zeros(B, 1, dtype=torch.long, device=img.device)
        h = None
        logits_all = []
        cur = feat
        for _ in range(MAX_LEN):
            emb = self.embed(inp)  # [B,1,E]
            x_in = torch.cat([cur, emb], dim=2)
            out, h = self.gru(x_in, h)
            logits = self.fc(out)  # [B,1,V]
            logits_all.append(logits)
            inp = logits.argmax(-1)
            cur = out
        return torch.cat(logits_all, dim=1)


def char_acc(pred, target):
    pad = 0
    correct = total = 0
    for p, t in zip(pred, target):
        for a, b in zip(p[1:], t):
            if b == pad:
                continue
            total += 1
            correct += int(a == b)
    return correct / max(1, total)


def train():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[OCR] device={device}")

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = OCRDataset(tf)
    n = len(ds)
    n_val = int(0.15 * n)
    val_ds, train_ds = torch.utils.data.random_split(ds, [n_val, n - n_val],
                                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)
    print(f"[OCR] train={len(train_ds)} val={len(val_ds)}")

    model = OCRModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    epochs = 4
    best = 0.0
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for img, tgt in train_loader:
            img, tgt = img.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(img, tgt, teacher_forcing=True)
            loss = crit(out.reshape(-1, out.size(-1)), tgt.reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

        model.eval(); ca = tot = 0
        with torch.no_grad():
            for img, tgt in val_loader:
                img, tgt = img.to(device), tgt.to(device)
                out = model(img, None, teacher_forcing=False)
                pred = out.argmax(-1)
                ca += char_acc(pred, tgt); tot += img.size(0)
        acc = ca / tot
        print(f"[OCR] epoch {epoch+1}: val_char_acc={acc:.3f}")
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "arch": "resnet18-gru",
                        "vocab_size": len(CHARS) + 1, "val_char_acc": acc}, OUT / "best_model.pt")

    elapsed = time.time() - start
    print(f"[OCR] DONE in {elapsed/60:.1f} min, best_char_acc={best:.3f}")
    with open(OUT / "config.json", "w") as f:
        json.dump({"arch": "resnet18-gru-ocr", "vocab_size": len(CHARS) + 1,
                   "max_len": MAX_LEN, "charset": CHARS,
                   "best_char_acc": float(best), "train_time_min": round(elapsed/60, 2),
                   "data": "HF ocr_datasets (subset)"}, f, indent=2)


if __name__ == "__main__":
    train()
