"""
Phase 2: Train xlm-roberta-base multilingual binary classifier on all HF text data.
Exports to trained_hugging_face_models/phase2
"""
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "trained_hugging_face_models" / "phase2"
DATA = ROOT / "hf_training" / "_data"

MODEL_NAME = "xlm-roberta-base"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class MultiDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor([self.labels[idx]])}


class MultiClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1, hidden=256):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(self.config.hidden_size, hidden)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        hidden = self.relu(self.hidden_layer(pooled))
        return self.classifier(self.dropout(hidden)).squeeze(1)


def train():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUT)
    print(f"[P2] device={device}")

    train_df = pd.read_csv(DATA / "phase2_train.csv")
    val_df = pd.read_csv(DATA / "phase2_val.csv")

    def prep(df):
        return df["text"].fillna("").tolist(), df["label"].values.astype(np.float32)

    tr_texts, tr_labels = prep(train_df)
    va_texts, va_labels = prep(val_df)
    train_ds = MultiDataset(tr_texts, tr_labels, tokenizer)
    val_ds = MultiDataset(va_texts, va_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=48)

    model = MultiClassifier(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-5, weight_decay=0.01)
    epochs = 2
    best = 0.0
    start = time.time()
    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for b in train_loader:
            ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
            labels = b["labels"].to(device).squeeze(1)
            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item(); total += labels.size(0)
        train_acc = correct / total

        model.eval(); vc = vt = 0
        with torch.no_grad():
            for b in val_loader:
                ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
                labels = b["labels"].to(device).squeeze(1)
                preds = (torch.sigmoid(model(ids, mask)) > 0.5).float()
                vc += (preds == labels).sum().item(); vt += labels.size(0)
        val_acc = vc / vt
        print(f"[P2] epoch {epoch+1}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc > best:
            best = val_acc
            torch.save({"state_dict": model.state_dict(), "arch": MODEL_NAME,
                        "val_acc": val_acc}, OUT / "best_model.pt")

    elapsed = time.time() - start
    print(f"[P2] DONE in {elapsed/60:.1f} min, best_val_acc={best:.3f}")
    langs = sorted(train_df["language"].unique().tolist())
    with open(OUT / "config.json", "w") as f:
        json.dump({"arch": MODEL_NAME, "best_val_acc": float(best), "binary": True,
                   "languages": langs, "num_langs": len(langs),
                   "train_time_min": round(elapsed/60, 2),
                   "data": "HF multilingual toxicity (3 datasets)"}, f, indent=2)


if __name__ == "__main__":
    train()
