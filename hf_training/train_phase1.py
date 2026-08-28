"""
Phase 1: Train roberta-base multi-label text classifier on HF-derived English data.
Exports to trained_hugging_face_models/phase1
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
OUT = ROOT / "trained_hugging_face_models" / "phase1"
DATA = ROOT / "hf_training" / "_data"

MODEL_NAME = "roberta-base"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.FloatTensor(self.labels[idx])}


class TextClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1, hidden=256):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(self.config.hidden_size, hidden)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        hidden = self.relu(self.hidden_layer(pooled))
        return self.classifier(self.dropout(hidden))


def train():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUT)
    print(f"[P1] device={device}")

    train_df = pd.read_csv(DATA / "phase1_train.csv")
    val_df = pd.read_csv(DATA / "phase1_val.csv")

    def prep(df):
        return df["text"].fillna("").tolist(), df[LABELS].values.astype(np.float32)

    tr_texts, tr_label = prep(train_df)
    va_texts, va_label = prep(val_df)
    train_ds = TextDataset(tr_texts, tr_label, tokenizer)
    val_ds = TextDataset(va_texts, va_label, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = TextClassifier(MODEL_NAME, num_labels=len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    best = 0.0
    start = time.time()
    for epoch in range(epochs):
        model.train()
        total_c = total = 0.0
        for b in train_loader:
            ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
            labels = b["labels"].to(device)
            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_c += (preds == labels).float().mean().item() * labels.size(0); total += labels.size(0)
        train_acc = total_c / total

        model.eval(); vc = vt = 0.0
        with torch.no_grad():
            for b in val_loader:
                ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
                labels = b["labels"].to(device)
                preds = (torch.sigmoid(model(ids, mask)) > 0.5).float()
                vc += (preds == labels).float().mean().item() * labels.size(0); vt += labels.size(0)
        val_acc = vc / vt
        print(f"[P1] epoch {epoch+1}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc > best:
            best = val_acc
            torch.save({"state_dict": model.state_dict(), "arch": MODEL_NAME,
                        "labels": LABELS, "val_acc": val_acc}, OUT / "best_model.pt")

    elapsed = time.time() - start
    print(f"[P1] DONE in {elapsed/60:.1f} min, best_val_acc={best:.3f}")
    with open(OUT / "config.json", "w") as f:
        json.dump({"arch": MODEL_NAME, "labels": LABELS, "best_val_acc": float(best),
                   "num_classes": len(LABELS), "train_time_min": round(elapsed/60, 2),
                   "data": "HF toxicity-multilingual (en-centric)"}, f, indent=2)


if __name__ == "__main__":
    train()
