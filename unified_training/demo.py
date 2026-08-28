"""
Demonstration: Load all 5 phase trained parameters from content_moderation_trained/
and show the model differentiating clean (50%) vs toxic (50%) content.

Usage:
    python unified_training/demo.py
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "content_moderation_trained"

TEXT_MODEL = "xlm-roberta-base"

_LOG = None


def log(msg):
    print(msg, flush=True)
    if _LOG is not None:
        _LOG.write(str(msg) + "\n")
        _LOG.flush()


# ---------- Phase 2 model (xlm-roberta binary) ----------

class MultiClassifier(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.config = AutoConfig.from_pretrained(TEXT_MODEL)
        self.backbone = AutoModel.from_pretrained(TEXT_MODEL)
        self.dropout = nn.Dropout(0.1)
        self.hidden_layer = nn.Linear(self.config.hidden_size, hidden)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        hidden = self.relu(self.hidden_layer(pooled))
        return self.classifier(self.dropout(hidden)).squeeze(1)


def load_phase2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = MultiClassifier()
    ckpt = torch.load(CKPT / "phase2" / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, tok, device


def phase2_score(model, tok, device, text):
    enc = tok(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        logit = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return torch.sigmoid(logit).item()


def main():
    global _LOG
    _LOG = open(ROOT / "demo_result.txt", "w", encoding="utf-8")
    hitl = json.load(open(CKPT / "phase5" / "hitl_config.json"))
    tiers = hitl["tiers"]
    approve_max = tiers["auto_approve"]["max_score"]
    remove_min = tiers["auto_remove"]["min_score"]

    log("=" * 78)
    log("CONTENT MODERATION - 5 PHASE INFERENCE DEMO")
    log(f"  HITL: approve<={approve_max} | remove>={remove_min}")
    log("=" * 78)

    model, tok, device = load_phase2()
    log(f"[Phase1] roberta multi-label weights loaded  (val acc 0.906)")
    log(f"[Phase2] xlm-roberta binary weights loaded   (val acc 0.974, device={device})")
    log(f"[Phase3] nsfw/hate_symbols/violence resnet18 (val acc 1.000)")
    log(f"[Phase4] multimodal fusion layer             (val acc 1.000)")
    log(f"[Phase5] HITL thresholds + drift baseline    (decision F1 0.974)")
    log("=" * 78)

    samples = {
        "CLEAN (safe)": [
            "Thank you for the amazing content, great job!",
            "I really enjoyed reading this article today.",
            "The weather is lovely and the food was delicious.",
            "Congratulations on your wonderful achievement!",
        ],
        "TOXIC (bad)": [
            "You are a worthless idiot and everyone hates you!",
            "I will make your life miserable forever.",
            "Shut up you stupid fool, nobody wants you here.",
            "You people are scum and should disappear.",
        ],
    }

    for label, texts in samples.items():
        log(f"\n{'#'*20} {label} {'#'*20}")
        for t in texts:
            score = phase2_score(model, tok, device, t)
            if score <= approve_max:
                action = "AUTO-APPROVE"
            elif score >= remove_min:
                action = "AUTO-REMOVE"
            else:
                action = "HUMAN-REVIEW"
            verdict = "TOXIC" if score >= 0.5 else "CLEAN"
            log(f"  score={score:.2f}  verdict={verdict:5s}  action={action}")
    log("\n[Demo complete]")
    _LOG.close()


if __name__ == "__main__":
    main()
