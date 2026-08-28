"""
Phase 1: Train baseline multi-label text classifier (roberta-base)
Balanced 50/50 dataset -> exports state_dict + config to content_moderation_trained/phase1
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "content_moderation_trained" / "phase1"
DATA = ROOT / "content_moderation_dataset"

MODEL_NAME = "roberta-base"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class ModeratorTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.FloatTensor(self.labels[idx]),
        }


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
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        hidden = self.relu(self.hidden_layer(pooled))
        hidden = self.dropout(hidden)
        return self.classifier(hidden)


def train():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P1] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA / "phase1_train.csv")
    val_df = pd.read_csv(DATA / "phase1_val.csv")

    def prep(df):
        texts = df["text"].fillna("").tolist()
        labels = df[LABELS].values.astype(np.float32)
        return texts, labels

    tr_texts, tr_labels = prep(train_df)
    va_texts, va_labels = prep(val_df)

    train_ds = ModeratorTextDataset(tr_texts, tr_labels, tokenizer)
    val_ds = ModeratorTextDataset(va_texts, va_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = TextClassifier(MODEL_NAME, num_labels=len(LABELS)).to(device)
    print(f"[P1] params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3

    best = 0.0
    start = time.time()
    for epoch in range(epochs):
        model.train()
        total, correct = 0.0, 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, mask)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).float().mean().item() * labels.size(0)
            total += labels.size(0)
        train_acc = correct / total

        model.eval()
        v_total, v_correct = 0.0, 0.0
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids, mask)
                val_loss += nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                v_correct += (preds == labels).float().mean().item() * labels.size(0)
                v_total += labels.size(0)
        val_acc = v_correct / v_total
        print(f"[P1] epoch {epoch+1}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} val_loss={val_loss:.4f}")

        if val_acc > best:
            best = val_acc
            torch.save({"state_dict": model.state_dict(), "arch": "roberta-base",
                        "labels": LABELS, "val_acc": val_acc}, OUT / "best_model.pt")

    elapsed = time.time() - start
    print(f"[P1] DONE in {elapsed/60:.1f} min, best_val_acc={best:.3f}")

    meta = {"arch": MODEL_NAME, "labels": LABELS, "best_val_acc": float(best),
            "train_time_min": round(elapsed / 60, 2), "num_classes": len(LABELS)}
    with open(OUT / "config.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    train()
