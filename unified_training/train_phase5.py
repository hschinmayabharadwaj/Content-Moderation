"""
Phase 5: HITL decision layer + drift monitor baseline.
Tunes 3-tier enforcement thresholds (auto-approve / human-review / auto-remove)
and computes a drift baseline from the trained Phase 2 model's validation scores.
Exports thresholds + baseline statistics to content_moderation_trained/phase5
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "content_moderation_trained" / "phase5"
P2 = ROOT / "content_moderation_trained" / "phase2"

TEXT_MODEL = "xlm-roberta-base"
LOW = 0.35   # below -> auto-approve
HIGH = 0.65  # above -> auto-remove; between -> human-review


class EvalDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len=128):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor([self.labels[idx]])}


class MultiClassifier(nn.Module):
    def __init__(self, model_name, hidden=256):
        super().__init__()
        self.config = AutoModel.from_pretrained(model_name).config
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.hidden_layer = nn.Linear(self.config.hidden_size, hidden)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        hidden = self.relu(self.hidden_layer(pooled))
        return self.classifier(self.dropout(hidden)).squeeze(1)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(P2 / "best_model.pt", map_location=device, weights_only=False)
    model = MultiClassifier(TEXT_MODEL).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    val_df = pd.read_csv(ROOT / "content_moderation_dataset" / "phase2_val.csv")
    ds = EvalDataset(val_df["text"].fillna("").tolist(),
                     val_df["label"].values.astype(np.float32), tok)
    loader = DataLoader(ds, batch_size=64)

    scores, trues = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            trues.extend(batch["label"].squeeze(1).cpu().numpy().tolist())
    scores = np.array(scores)
    trues = np.array(trues)

    # Tune thresholds by grid search over F1 for the operational decision
    best = -1
    best_low, best_high = LOW, HIGH
    grid = np.arange(0.05, 0.96, 0.05)
    for lo in grid:
        for hi in grid:
            if lo >= hi:
                continue
            preds = np.full_like(scores, -1)
            preds[scores < lo] = 0      # approve
            preds[(scores >= lo) & (scores <= hi)] = 1  # review
            preds[scores > hi] = 2      # remove
            # decision quality: approve==0 should match toxic==0, remove==2 toxic==1
            agree = np.zeros_like(trues)
            agree[(trues == 0) & (preds == 0)] = 1
            agree[(trues == 1) & (preds == 2)] = 1
            agree[(trues == 1) & (preds == 1)] = 0.5  # review catches undershoot partially
            f = agree.mean()
            if f > best:
                best = f
                best_low, best_high = lo, hi

    # Metrics at tuned thresholds
    preds = np.full_like(scores, 1)
    preds[scores < best_low] = 0
    preds[scores > best_high] = 2
    tp = ((preds == 2) & (trues == 1)).sum()
    fp = ((preds == 2) & (trues == 0)).sum()
    fn = ((preds != 2) & (trues == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    # Drift baseline: population statistics of the score distribution (reference)
    baseline = {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "p5": float(np.percentile(scores, 5)),
        "p50": float(np.percentile(scores, 50)),
        "p95": float(np.percentile(scores, 95)),
        "positive_rate": float((trues == 1).mean()),
    }

    config = {
        "tiers": {
            "auto_approve": {"max_score": float(best_low), "rationale": "score below this -> auto-approve"},
            "human_review": {"min_score": float(best_low), "max_score": float(best_high),
                             "rationale": "scores in range -> route to human"},
            "auto_remove": {"min_score": float(best_high), "rationale": "score above this -> auto-remove"},
        },
        "decision_accuracy": {"precision": float(prec), "recall": float(rec), "f1": float(f1)},
        "drift_baseline": baseline,
        "model": TEXT_MODEL,
    }
    with open(OUT / "hitl_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[P5] tuned approve<={best_low:.2f}, remove>={best_high:.2f}")
    print(f"[P5] decision F1={f1:.3f} P={prec:.3f} R={rec:.3f}")
    print(f"[P5] drift baseline mean={baseline['mean']:.3f} std={baseline['std']:.3f}")
    print(f"[P5] DONE")


if __name__ == "__main__":
    main()
